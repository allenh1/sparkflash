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

//! The chunk-wide prefill path on the tensor cores (kernels/qwen4_prefill_sm121.cu).
//!
//! A prompt chunk of [GEMM_MIN_BATCH](super::GEMM_MIN_BATCH) tokens or more runs its dense
//! projections, its routed experts and its Gated DeltaNet recurrence here. A decode step and a
//! speculative verify are a handful of rows and never do, so a verified row is computed by
//! exactly the kernels of the decode step that would have produced it.
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

use super::{DeviceSequence, GEMM_MIN_BATCH, GEMM_PARTIAL_FLOATS, GpuMat};
use crate::config::Qwen4ExpConfig;

/// The tensor-core cubin build.rs embeds. Empty on a CPU-only build.
#[cfg(sparkflash_qwen4_cubin)]
const PREFILL_CUBIN: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_PREFILL_CUBIN"));
#[cfg(not(sparkflash_qwen4_cubin))]
const PREFILL_CUBIN: &[u8] = &[];

/// The tensor-core prefill cubin this build embeds, for kernel-level tests.
#[must_use]
pub fn embedded_prefill_cubin() -> &'static [u8] {
    PREFILL_CUBIN
}

/// GLM-5.3-Flash's bulk prefill cubin (block-scaled FP8 dense GEMM and routed experts).
#[cfg(sparkflash_qwen4_cubin)]
const GLM_PREFILL_CUBIN: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_PREFILL_CUBIN"));
#[cfg(not(sparkflash_qwen4_cubin))]
const GLM_PREFILL_CUBIN: &[u8] = &[];

/// The GLM bulk prefill cubin this build embeds.
#[must_use]
pub fn embedded_glm_prefill_cubin() -> &'static [u8] {
    GLM_PREFILL_CUBIN
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
/// The key head width the scan is compiled for (Q4P_GDN_KEY).
pub const GDN_KEY: usize = 128;
/// The value head width the scan is compiled for (Q4P_GDN_VALUE).
pub const GDN_VALUE: usize = 128;
/// Floats per staged (token, key head): q then k, padded (Q4P_GDN_ROW).
pub const GDN_QK_ROW: usize = 288;
/// Blocks the scan aims for: one an SM on GB10's forty-eight.
const GDN_SCAN_TARGET_BLOCKS: usize = 48;
/// The most a head's value columns split: eight columns, one warp, a block.
const GDN_SCAN_MAX_SPLITS: usize = 16;

/// q4p_gdn_scan's and q4p_kda_scan's grid and block width for `heads` heads: the fewest
/// power-of-two splits of a head's value columns that give every SM a block, each split
/// 4 * GDN_VALUE / splits threads. A value column's recurrence reads no other column, so any
/// split computes the same bits; a rank holding eight heads (TP=4) would otherwise run the scan
/// on eight of the forty-eight SMs.
pub fn gdn_scan_shape(heads: usize) -> ((u32, u32, u32), u32) {
    let mut splits = 1;
    while heads * splits < GDN_SCAN_TARGET_BLOCKS && splits < GDN_SCAN_MAX_SPLITS {
        splits *= 2;
    }
    (
        (heads as u32, splits as u32, 1),
        (4 * GDN_VALUE / splits) as u32,
    )
}
/// The conv history q4p_gdn_conv_state carries in registers (Q4P_CONV_HISTORY).
pub const CONV_HISTORY: usize = 8;
/// The dense MMA's token and column tile (OM_BM, OM_BN).
const DENSE_TILE: usize = 128;
/// CTAs a split dense MMA aims for: two per SM on GB10's forty-eight.
const DENSE_TARGET_CTAS: usize = 96;
/// The thinnest K slice a split takes, in 32-column blocks.
const DENSE_MIN_SLICE_BLOCKS: usize = 8;

/// How a prompt is cut and whether its chunks take the tensor cores, fixed when the backend
/// loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    /// Tokens per layer-major chunk.
    pub chunk: usize,
    /// Whether a chunk of GEMM_MIN_BATCH tokens or more takes this image.
    /// `SPARKFLASH_QWEN4_PREFILL_MMA=0` keeps every chunk on the token kernels, for A/B runs.
    pub tensor_cores: bool,
}

impl PrefillPlan {
    /// A plan for `chunk`-token chunks, clamped to the widths the scratch supports and, past
    /// DENSE_ROWS, rounded down to a whole number of them. The tensor cores are only taken when
    /// this build carries the prefill image.
    #[must_use]
    pub fn new(chunk: usize, tensor_cores: bool) -> Self {
        let chunk = chunk.clamp(MIN_PREFILL_CHUNK, MAX_PREFILL_CHUNK);
        Self {
            chunk: if chunk > DENSE_ROWS {
                chunk - chunk % DENSE_ROWS
            } else {
                chunk
            },
            tensor_cores: tensor_cores && !PREFILL_CUBIN.is_empty(),
        }
    }

    /// The forwards a prompt of `tokens` runs as: `chunk` tokens each, except that a tail past the
    /// last whole DENSE_ROWS too short for the tensor cores runs on its own. Every row then
    /// computes as it did when chunks were DENSE_ROWS wide: dense pieces start where those chunks
    /// did, and such a tail was a chunk of its own on the token kernels then too.
    #[must_use]
    pub fn forwards(&self, tokens: usize) -> Vec<std::ops::Range<usize>> {
        let mut forwards = Vec::with_capacity(tokens.div_ceil(self.chunk.max(1)) + 1);
        let mut start = 0;
        while start < tokens {
            let mut end = (start + self.chunk).min(tokens);
            let tail = (end - start) % DENSE_ROWS;
            if end - start > DENSE_ROWS && tail < GEMM_MIN_BATCH {
                end -= tail;
            }
            forwards.push(start..end);
            start = end;
        }
        forwards
    }

    /// The serving default, as `SPARKFLASH_QWEN4_PREFILL_MMA` and `SPARKFLASH_QWEN4_PREFILL_CHUNK`
    /// adjust it.
    #[must_use]
    pub fn from_env() -> Self {
        let tensor_cores = !PREFILL_CUBIN.is_empty()
            && std::env::var("SPARKFLASH_QWEN4_PREFILL_MMA")
                .map(|value| value != "0")
                .unwrap_or(true);
        let fallback = if tensor_cores {
            DEFAULT_PREFILL_CHUNK
        } else {
            TOKEN_PREFILL_CHUNK
        };
        let chunk = std::env::var("SPARKFLASH_QWEN4_PREFILL_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(fallback);
        Self::new(chunk, tensor_cores)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PrefillKernels {
    dense_mma: Kernel,
    epilogue: Kernel,
    split_bf16: Kernel,
    moe_plan: Kernel,
    moe_gate_up: Kernel,
    moe_down: Kernel,
    gdn_conv: Kernel,
    gdn_conv_state: Kernel,
    gdn_prep: Kernel,
    gdn_scan: Kernel,
    gdn_gate: Kernel,
}

impl PrefillKernels {
    pub(super) fn load(cuda: &Cuda) -> Result<Self, String> {
        if PREFILL_CUBIN.is_empty() {
            return Err(
                "this build has no Qwen4-Exp prefill cubin: rebuild on a host with nvcc so \
                 build.rs compiles crates/sparkflash-model-qwen4-exp/kernels/qwen4_prefill_sm121.cu"
                    .to_owned(),
            );
        }
        let named = |name: &'static str| -> Result<Kernel, String> {
            let (_, kernel) = cuda.load_named(PREFILL_CUBIN, name)?;
            Ok(kernel)
        };
        Ok(Self {
            dense_mma: named("q4p_dense_mma")?,
            epilogue: named("q4p_epilogue")?,
            split_bf16: named("q4p_split_bf16")?,
            moe_plan: named("q4p_moe_plan")?,
            moe_gate_up: named("q4p_moe_gate_up")?,
            moe_down: named("q4p_moe_down")?,
            gdn_conv: named("q4p_gdn_conv")?,
            gdn_conv_state: named("q4p_gdn_conv_state")?,
            gdn_prep: named("q4p_gdn_prep")?,
            gdn_scan: named("q4p_gdn_scan")?,
            gdn_gate: named("q4p_gdn_gate")?,
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
                (chunk * config.linear_num_key_heads * GDN_QK_ROW * 4).max(4),
                total,
            )?,
            gdn_decay_beta: cuda
                .alloc((chunk * config.linear_num_value_heads * 8).max(4), total)?,
        })
    }

    pub(super) fn pointers(&self) -> [&Deviceptr; 5] {
        [
            &self.expert_input,
            &self.items,
            &self.item_count,
            &self.gdn_qk,
            &self.gdn_decay_beta,
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
    /// Whether a chunk of `batch` rows takes this path. The caller also keeps it off for a
    /// forward that snapshots a row, which only a speculative verify does.
    pub(super) fn chunk_path(&self, batch: usize) -> bool {
        self.backend.prefill.tensor_cores && batch >= GEMM_MIN_BATCH
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
    /// The split is chosen per DENSE_ROWS piece of the chunk, from its first row. A row's sums
    /// depend on the split and nothing else in the launch, so whole pieces go in one launch, as
    /// many as the partials hold.
    pub(super) fn dense_mma(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        batch: usize,
        epilogue: u32,
        scale: f32,
    ) -> Result<bool, BackendError> {
        if !self.chunk_path(batch)
            || !part("dense")
            || !matrix.rows.is_multiple_of(32)
            || !matrix.columns.is_multiple_of(32)
            || ![matrix.pointer, input, output].into_iter().all(aligned)
        {
            return Ok(false);
        }
        let k_blocks = matrix.columns / 32;
        let mut done = 0usize;
        while done < batch {
            let left = batch - done;
            let piece = left.min(DENSE_ROWS);
            let segments = dense_segments(matrix.rows, k_blocks, piece);
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
    /// items next to each other in the grid.
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
        self.launch(
            kernels.moe_plan,
            (1, 1, 1),
            (1024, 1, 1),
            &[
                self.expert_indices.0,
                slots as u64,
                config.num_experts as u64,
                self.expert_perm.0,
                scratch.items.0,
                scratch.item_count.0,
                self.expert_accumulator.0,
                hidden as u64,
            ],
        )?;
        // The grid covers the most items the plan can write; past the count it wrote, a CTA
        // returns before touching anything.
        let items = (slots.div_ceil(MOE_ROWS) + config.num_experts.min(slots)) as u32;
        self.launch(
            kernels.moe_gate_up,
            ((intermediate / MOE_GATE_UP_COLUMNS) as u32, items, 1),
            (256, 1, 1),
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
            kernels.moe_down,
            ((hidden / MOE_DOWN_COLUMNS) as u32, items, 1),
            (256, 1, 1),
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

    /// The chunk's Gated DeltaNet after its projections: the conv over every token at once, the
    /// normalised queries and keys and each row's decay and beta, the scan with the state in
    /// registers, and the gated norm in place. Reads `linear_qkv`, `linear_z`, `linear_a` and
    /// `linear_b`, leaves the gated rows in `linear_out` and advances both states. Returns false,
    /// launching nothing, for a configuration other than the one the scan is compiled for.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_chunk(
        &self,
        conv_weight: Deviceptr,
        conv_state: Deviceptr,
        recurrent_state: Deviceptr,
        a_log: Deviceptr,
        dt_bias: Deviceptr,
        norm: Deviceptr,
        batch: usize,
    ) -> Result<bool, BackendError> {
        let config = &self.backend.config;
        let taps = config.linear_conv_kernel_dim;
        let key_heads = config.linear_num_key_heads;
        let heads = config.linear_num_value_heads;
        let groups = config.value_groups();
        let width = config.linear_conv_width();
        if !part("gdn")
            || config.linear_key_head_dim != GDN_KEY
            || config.linear_value_head_dim != GDN_VALUE
            || !(2..=CONV_HISTORY + 1).contains(&taps)
            || groups == 0
            || groups > GDN_KEY
            || key_heads * groups != heads
        {
            return Ok(false);
        }
        let kernels = &self.backend.kernels.prefill;
        let scratch = &self.prefill;
        let columns = width.div_ceil(256).max(1) as u32;
        // A chunk is never a speculative verify, so it writes no snapshot region.
        let (null, no_rows, no_stride) = (0_u64, 0_u64, 0_u64);
        self.launch(
            kernels.gdn_conv,
            (columns, batch as u32, 1),
            (256, 1, 1),
            &[
                self.linear_qkv.0,
                conv_state.0,
                conv_weight.0,
                self.linear_conv.0,
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
                null,
                no_rows,
                no_stride,
            ],
        )?;
        self.prof_mark("gdn_conv")?;
        let key_width = config.linear_key_width();
        self.launch(
            kernels.gdn_prep,
            (key_heads as u32, batch as u32, 1),
            (GDN_KEY as u32, 1, 1),
            &[
                self.linear_conv.0,
                self.linear_a.0,
                self.linear_b.0,
                a_log.0,
                dt_bias.0,
                scratch.gdn_qk.0,
                scratch.gdn_decay_beta.0,
                key_width as u64,
                key_heads as u64,
                heads as u64,
                GDN_VALUE as u64,
                groups as u64,
            ],
        )?;
        let (scan_grid, scan_threads) = gdn_scan_shape(heads);
        self.launch(
            kernels.gdn_scan,
            scan_grid,
            (scan_threads, 1, 1),
            &[
                scratch.gdn_qk.0,
                scratch.gdn_decay_beta.0,
                self.linear_conv.0,
                recurrent_state.0,
                self.linear_out.0,
                key_width as u64,
                key_heads as u64,
                heads as u64,
                GDN_VALUE as u64,
                groups as u64,
                batch as u64,
                null,
                no_rows,
                no_stride,
            ],
        )?;
        self.launch(
            kernels.gdn_gate,
            (heads as u32, batch as u32, 1),
            (GDN_VALUE as u32, 1, 1),
            &[
                self.linear_out.0,
                self.linear_z.0,
                norm.0,
                self.linear_out.0,
                heads as u64,
                GDN_VALUE as u64,
                u64::from((config.rms_norm_eps as f32).to_bits()),
                u64::from(config.output_gate_is_sigmoid()),
            ],
        )?;
        self.prof_mark("gdn_recur")?;
        Ok(true)
    }
}
