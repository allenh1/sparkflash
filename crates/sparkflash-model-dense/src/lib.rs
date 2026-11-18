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

//! The shared dense device arm: a family no pack transcribes, served on the device.
//!
//! A grouped-query decoder (embedding, RMSNorm, grouped-query attention with or without the
//! per-head output gate and q/k norms, plain rotate-half RoPE, a dense SwiGLU or routed experts,
//! a plain residual and the head) needs no code of its own: this arm reads the shared
//! [Architecture], resolves the checkpoint's own spelling through [sparkflash_ops::naming],
//! uploads every tensor through [DraftUploader] and launches the shared token-image bodies through
//! their typed [sparkflash_ops::calls]. Its prompt projections take the shared wide tile under
//! [WidePolicy::DENSE]; decode rows keep the row GEMV.
//!
//! A projection or expert stored as E4M3 codes under one BF16 inverse scale a 128 x 128 tile (the
//! Qwen3 FP8 releases' form, [Proj::Fp8]) runs the kernels Qwen's and GLM's FP8 weights run: the
//! FP8 tensor-core tile ([Fp8Tile]) for a prompt chunk by [route_fp8], the tiled E4M3 GEMV for
//! decode rows, and the token passes' FP8 form for the experts. Activations stay f32 in all of
//! them, as in the host reference, which applies the same tile scales to the same codes.
//!
//! A routed layer runs the parts the released arms run: the one router body (`q4e_route`, any
//! family's score, top-k, renormalisation and scale), the shared expert plan's run walk
//! ([MoePlanCall::runs]), the token path's expert passes in their BF16 form (one block a run of
//! an expert's rows, Qwen's NVFP4 and FP8 body) and the combine, with the shared expert and its
//! sigmoid gate on the same projections as a dense MLP.
//!
//! A Gated DeltaNet layer (a hybrid family's linear attention) runs the scan image's chunk
//! kernels through [sparkflash_ops::gdn]'s calls, the ones Qwen's prompts take: the conv over a
//! segment's rows and its history advanced past them, the prep, the scan [ScanPlan] resolves for
//! the head pair, and the gated norm. Every row count takes them, decode rows included, so a row's
//! state is the same bits in a prompt, a step alone or a fused step. Each sequence keeps its conv
//! history and delta-rule state in its lane, and its rewind point copies them
//! ([Checkpoint]) when a prompt ends.
//!
//! Which checkpoints it takes is [sparkflash_ops::parts::dense_device_refusal]: nothing the host
//! reference does not execute, and nothing the reference does that these kernels do not. Its
//! gate is a per-layer parity report against that reference (the CLI's `dense_parity` test),
//! not a recorded digest: the arm keeps its KV cache in BF16 where the reference keeps f32, so a
//! row agrees to a tolerance rather than to the bit.
//!
//! It serves on the shared runtime's contract ([DenseLanes], a [FusedArm]): fused lanes whose
//! decode rows share one walk, prompts a chunk at a time between the lanes' steps, kept
//! conversations and a rewind point to the end of each prompt, its caches on the shared lane
//! memory ([SequenceLanes]), and the ledger.
//!
//! It runs on one rank or on a tensor-parallel group ([DenseDeviceBackend::open_rank]): a rank
//! holds its share of every attention layer's query heads (and of the key/value heads, which a
//! world wider than them shares), of every Gated DeltaNet layer's key and value heads, and of
//! every MLP's and expert's width (the gate and up rows, the down columns), as the shared
//! [TpPlan] splits them, and holds the rest whole: the embedding, the norms, the routers and the
//! head. Each branch therefore leaves a partial of its hidden rows, which the ranks sum in rank
//! order ([RowSums]) before the residual takes it, so the residual is the same bits on every rank
//! and so is every replicated decision (the router's picks, the token). Rank 0 serves the group's
//! fused lanes and every other rank replays them ([DenseDeviceBackend::enable_group_fusion],
//! [DenseDeviceBackend::mirror_group]).

use std::sync::{Arc, Mutex};

use sparkflash_kernels::checkpoint::Checkpoint;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::{
    dory_cubin, moe_plan_cubin, qwen4_attn_cubin, qwen4_exp_cubin, qwen4_moe_cubin,
    qwen4_prefill_cubin, qwen4_scan_cubin,
};
use sparkflash_kernels::lane_memory::{Layout, SequenceLane, SequenceLanes};
use sparkflash_model::architecture::{Activation, AttentionKind, Ffn, WeightFormat};
use sparkflash_model::backend::{
    BackendError, ModelBackend, ModelSession, Sampling, Speculation, StepOutput,
};
use sparkflash_model::vision::VisionImage;
use sparkflash_model::{Architecture, Capabilities, ModelDescriptor, TpAxis, TpPlan};
use sparkflash_ops::OpPlan;
use sparkflash_ops::calls::{self, Call, FullAttention, RmsNorm, Rope};
use sparkflash_ops::dflash::{DraftUploader, Part as Run};
use sparkflash_ops::engine_plan::{EnginePlan, PlanError};
use sparkflash_ops::gdn::{self, GdnGeometry, gdn_qk_row};
use sparkflash_ops::gemv::{
    CudaHandle, E4M3_TILE, GEMV_MAX_BATCH, GemvDevice, GpuMat, GpuMatE4m3Tile, TILED_GEMV_BLOCKS,
    TILED_GEMV_PARTIALS, gemv_e4m3_tile,
};
use sparkflash_ops::moe::{Launch, MoePlanCall, combine_grid, router_score, token};
use sparkflash_ops::naming::{AttentionNames, ExpertNames, FfnNames, GdnNames, TensorNames};
use sparkflash_ops::packed::{self, Form, PackedSet, Placement};
use sparkflash_ops::projection::{
    Fp8Tile, Route, WIDE_SHARED, WidePolicy, WideTile, route_bf16, route_fp8,
};
use sparkflash_ops::sampler::Sampler;
use sparkflash_ops::scan::{ScanKind, ScanPlan};
use sparkflash_ops::tp::RowSums;
use sparkflash_runtime::{
    FusedArm, Fusion, Ledger, Mode, Part, Resume, Strategy, TextWire, resume,
};
use sparkflash_weights::tp::TpRuntime;
use sparkflash_weights::{DType, WeightSource};

mod walk;

pub use walk::DenseWalk;

/// The host-facing name of a device refusal.
const DEVICE: &str = "dense_device";

/// The prompt rows one pass of the layer walk takes: two of the wide tile's 128-row blocks.
pub const CHUNK_ROWS: usize = 256;

/// One layer's attention and MLP shape. A Gated DeltaNet layer's attention fields are zero and
/// its geometry is `linear`'s.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Shape {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    gated: bool,
    rotary: usize,
    theta: f32,
    /// The dense MLP's width, or a routed layer's shared expert's (zero without one).
    intermediate: usize,
    /// The SwiGLU limit every MLP of the layer takes (`swiglu_limit`), when the release sets one.
    clamp: Option<f32>,
    /// A routed layer's experts.
    routed: Option<Routed>,
    /// A Gated DeltaNet layer's geometry.
    linear: Option<Linear>,
    /// Where this rank's share of each sharded axis starts.
    cut: Cut,
}

/// The first element of this rank's share of each sharded axis of a layer, in heads or in
/// columns of width: all zero at one rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Cut {
    heads: usize,
    kv_heads: usize,
    /// The dense MLP's (or the shared expert's) first column.
    intermediate: usize,
    /// Each routed expert's first column.
    expert: usize,
    key_heads: usize,
    value_heads: usize,
}

/// Which share of each sharded axis this rank holds, as the shared [TpPlan] splits it.
#[derive(Debug, Clone, Copy)]
struct Shard(TpPlan);

impl Shard {
    /// One rank holding everything.
    fn whole() -> Self {
        Self(TpPlan::new(1, 0).expect("one rank is a world"))
    }

    /// `(first, count)` of `total` split evenly along `axis`, with nothing padded: the kernels
    /// take the rank's share as its own width.
    fn split(self, total: usize, axis: TpAxis, what: &str) -> Result<(usize, usize), String> {
        let split = self
            .0
            .split(total, axis, 1)
            .map_err(|error| format!("{what}: {error}"))?;
        Ok((split.offset, split.per_rank))
    }

    /// `(first, count)` of `total` as whole `tile`-wide runs, the remainder to the first ranks:
    /// a block-scaled FP8 expert's share, whose scales cover whole tiles.
    fn tiles(self, total: usize, tile: usize, what: &str) -> Result<(usize, usize), String> {
        let split = self
            .0
            .split_tiles(total, tile, TpAxis::Intermediate)
            .map_err(|error| format!("{what}: {error}"))?;
        Ok((split.offset, split.per_rank))
    }
}

/// A Gated DeltaNet layer's geometry and its output gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Linear {
    geometry: GdnGeometry,
    /// A sigmoid output gate (else SiLU).
    sigmoid: bool,
}

/// A routed layer's rule: what `q4e_route` and the expert passes take.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Routed {
    experts: usize,
    top_k: usize,
    /// One expert's SwiGLU width.
    intermediate: usize,
    /// The router's score ([router_score]).
    score: u64,
    renormalise: bool,
    scaling: f32,
}

impl Shape {
    /// Layer `index`'s shape as `shard`'s rank holds it, every sharded count and width its
    /// share, and where each share starts.
    fn of(architecture: &Architecture, index: usize, shard: Shard) -> Result<Self, String> {
        let layer = &architecture.layers[index];
        let attention = &layer.attention;
        let what = |axis: &str| format!("layer {index}'s {axis}");
        let mut cut = Cut::default();
        let (heads, kv_heads, head_dim, gated, linear) = match &attention.kind {
            AttentionKind::Gqa {
                kv_heads,
                output_gate,
            } => {
                let (first, heads) =
                    shard.split(attention.heads, TpAxis::Heads, &what("query heads"))?;
                let (kv_first, kv_heads) =
                    shard.split(*kv_heads, TpAxis::KvHeads, &what("key/value heads"))?;
                (cut.heads, cut.kv_heads) = (first, kv_first);
                (heads, kv_heads, attention.head_dim, *output_gate, None)
            }
            AttentionKind::Gdn(gdn) => {
                // Both head counts split evenly: a rank's value heads read its own key heads.
                let (key_first, key_heads) = shard.split(
                    gdn.key_heads,
                    TpAxis::Heads,
                    &what("Gated DeltaNet key heads"),
                )?;
                let (value_first, value_heads) = shard.split(
                    gdn.value_heads,
                    TpAxis::Heads,
                    &what("Gated DeltaNet value heads"),
                )?;
                (cut.key_heads, cut.value_heads) = (key_first, value_first);
                (
                    0,
                    0,
                    0,
                    false,
                    Some(Linear {
                        geometry: GdnGeometry {
                            key_heads,
                            heads: value_heads,
                            key_dim: gdn.key_dim,
                            value_dim: gdn.value_dim,
                            taps: gdn.conv_kernel,
                        },
                        sigmoid: gdn.output_gate == Activation::Sigmoid,
                    }),
                )
            }
            _ => {
                return Err(format!(
                    "layer {index} is neither grouped-query attention nor a Gated DeltaNet"
                ));
            }
        };
        let fp8_experts = matches!(
            architecture.quantization.expert_format,
            WeightFormat::Fp8Tile { .. }
        );
        let (intermediate, routed) = match &layer.ffn {
            Ffn::Dense(dense) => {
                let (first, width) =
                    shard.split(dense.intermediate, TpAxis::Intermediate, &what("MLP width"))?;
                cut.intermediate = first;
                (width, None)
            }
            Ffn::Moe(moe) => {
                let shared = if moe.shared_experts > 0 {
                    let (first, width) = shard.split(
                        moe.shared_intermediate,
                        TpAxis::Intermediate,
                        &what("shared expert's width"),
                    )?;
                    cut.intermediate = first;
                    width
                } else {
                    0
                };
                // A share is whole runs of what the expert passes read at a time (a scale tile
                // of FP8, eight BF16 columns), the remainder to the first ranks: an even split
                // of a 136-wide expert is two 68-wide shares the BF16 passes cannot read.
                let run = if fp8_experts {
                    E4M3_TILE
                } else {
                    sparkflash_ops::parts::DENSE_BF16_EXPERT_RUN
                };
                let (first, width) = shard.tiles(moe.intermediate, run, &what("expert width"))?;
                cut.expert = first;
                (
                    shared,
                    Some(Routed {
                        experts: moe.routed_experts,
                        top_k: moe.active_experts,
                        intermediate: width,
                        score: router_score(moe.score),
                        renormalise: moe.norm_topk,
                        scaling: moe.routed_scaling as f32,
                    }),
                )
            }
        };
        Ok(Self {
            heads,
            kv_heads,
            head_dim,
            gated,
            rotary: attention.rope.rotated_dims.min(head_dim),
            theta: attention.rope.theta.max(1.0) as f32,
            intermediate,
            clamp: match &layer.ffn {
                Ffn::Dense(dense) => dense.clamp,
                Ffn::Moe(moe) => moe.clamp,
            }
            .map(|limit| limit as f32),
            routed,
            linear,
            cut,
        })
    }

    /// The output projection's input: the attended heads, or the linear attention's values.
    const fn attended_width(&self) -> usize {
        match self.linear {
            Some(linear) => linear.geometry.value_width(),
            None => self.query_width(),
        }
    }

    const fn query_width(&self) -> usize {
        self.heads * self.head_dim
    }

    const fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The query projection's rows: the queries, and the interleaved gates when gated.
    const fn query_rows(&self) -> usize {
        if self.gated {
            2 * self.query_width()
        } else {
            self.query_width()
        }
    }
}

/// One layer's device weights.
#[derive(Debug, Clone, Copy)]
struct Layer {
    shape: Shape,
    input_norm: Deviceptr,
    mixer: Mixer,
    /// The attention's output projection, or the linear attention's `out_proj`.
    o: Proj,
    post_norm: Deviceptr,
    ffn: LayerFfn,
}

/// A layer's token mixer on the device.
#[derive(Debug, Clone, Copy)]
enum Mixer {
    /// Grouped-query attention's projections and its q/k norms.
    Gqa {
        q: Proj,
        k: Proj,
        v: Proj,
        q_norm: Option<Deviceptr>,
        k_norm: Option<Deviceptr>,
    },
    Gdn(GdnLayer),
}

/// A Gated DeltaNet layer's weights: the fused q/k/v projection, the output gate's, the beta and
/// decay logits', the conv (BF16 `[conv_width][taps]`), the decay vectors and the gated norm (BF16
/// a value head's width).
#[derive(Debug, Clone, Copy)]
struct GdnLayer {
    qkv: Proj,
    z: Proj,
    b: Proj,
    a: Proj,
    conv: Deviceptr,
    a_log: Deviceptr,
    dt_bias: Deviceptr,
    norm: Deviceptr,
}

impl Layer {
    /// Every BF16 projection the layer's GEMVs read.
    fn bf16_projections(&self) -> Vec<GpuMat> {
        let mut projections = vec![self.o];
        match self.mixer {
            Mixer::Gqa { q, k, v, .. } => projections.extend([q, k, v]),
            Mixer::Gdn(gdn) => projections.extend([gdn.qkv, gdn.z, gdn.b, gdn.a]),
        }
        let mlp = |mlp: Mlp| [mlp.gate, mlp.up, mlp.down];
        match self.ffn {
            LayerFfn::Dense(dense) => projections.extend(mlp(dense)),
            LayerFfn::Routed {
                router,
                shared,
                shared_gate,
                ..
            } => {
                projections.push(router);
                projections.extend(shared.into_iter().flat_map(mlp));
                projections.extend(shared_gate);
            }
        }
        projections
            .into_iter()
            .filter_map(|projection| match projection {
                Proj::Bf16(mat) => Some(mat),
                Proj::Fp8(_) => None,
            })
            .collect()
    }
}

/// One projection's weights on the device: BF16, or E4M3 codes under one BF16 inverse scale a
/// 128 x 128 tile, as the checkpoint stores the tensor.
#[derive(Debug, Clone, Copy)]
enum Proj {
    Bf16(GpuMat),
    Fp8(GpuMatE4m3Tile),
}

impl Proj {
    /// Output rows.
    const fn rows(&self) -> usize {
        match self {
            Self::Bf16(mat) => mat.rows,
            Self::Fp8(mat) => mat.rows,
        }
    }

    /// The token passes' table entries: the weight and its scale grid (none for BF16).
    const fn table(&self) -> [u64; 2] {
        match self {
            Self::Bf16(mat) => [mat.pointer.0, 0],
            Self::Fp8(mat) => [mat.weight.0, mat.scales.0],
        }
    }
}

/// One SwiGLU MLP's projections: a dense layer's, or a routed layer's shared expert.
#[derive(Debug, Clone, Copy)]
struct Mlp {
    gate: Proj,
    up: Proj,
    down: Proj,
}

/// A layer's MLP on the device.
#[derive(Debug, Clone, Copy)]
enum LayerFfn {
    Dense(Mlp),
    Routed {
        /// The router, `[experts][hidden]`.
        router: Proj,
        /// The selection's correction bias, f32 an expert, when selection adds one.
        bias: Option<Deviceptr>,
        /// Six addresses an expert, the token passes' table: gate, its scale grid (unread for
        /// BF16), up, its grid, down, its grid.
        table: Deviceptr,
        shared: Option<Mlp>,
        /// The shared expert's sigmoid gate, one BF16 `[hidden]` row.
        shared_gate: Option<Proj>,
    },
}

/// The routed layers' kernels, loaded when a layer routes.
#[derive(Clone, Copy)]
struct MoeKernels {
    route: Kernel,
    plan: Kernel,
    gate_up: Kernel,
    /// The gate/up pass under a `swiglu_limit`.
    gate_up_clamped: Kernel,
    down: Kernel,
    combine: Kernel,
}

/// The kernels this arm launches, loaded once.
#[derive(Clone, Copy)]
struct Kernels {
    add: Kernel,
    /// A group's row sum ([RowSums]) on the device legs.
    sum_slots: Kernel,
    deinterleave: Kernel,
    embed: Kernel,
    gemv: Kernel,
    packed: packed::Kernels,
    rmsnorm: Kernel,
    rope: Kernel,
    swiglu: Kernel,
    swiglu_clamped: Kernel,
    attention: Kernel,
    combine: Kernel,
    cache: Kernel,
    wide: Kernel,
    moe: Option<MoeKernels>,
    fp8: Option<Fp8Kernels>,
    gdn: Option<GdnKernels>,
}

/// The Gated DeltaNet layers' kernels, loaded when a layer is one: the scan image's chunk path
/// ([sparkflash_ops::gdn]) with the scan the [ScanPlan] resolves, and the zero a reset state takes.
#[derive(Clone, Copy)]
struct GdnKernels {
    conv: Kernel,
    history: Kernel,
    prep: Kernel,
    scan: Kernel,
    /// The plan's scan grid and block.
    scan_launch: ((u32, u32, u32), (u32, u32, u32)),
    gate: Kernel,
    zero: Kernel,
}

/// The block-scaled FP8 projections' kernels, loaded when the checkpoint declares FP8: the
/// tiled E4M3 GEMV for decode rows and the FP8 tensor-core tile for prompt chunks.
#[derive(Clone, Copy)]
struct Fp8Kernels {
    gemv: Kernel,
    tile: Kernel,
}

// SAFETY: a Kernel is an opaque driver function word, and the handle only names it. The context
// that owns the module lives in the shared state, which outlives every session.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

impl Kernels {
    /// Loads every kernel, the routed layers' when `stage` (the expert passes' staging
    /// reservation, [token::stage_bytes]) names one, in the experts' `formats.experts`, the FP8
    /// projections' when `formats.fp8` says the checkpoint declares FP8, and the Gated DeltaNet
    /// layers' when `scan` is their plan.
    fn load(
        cuda: &Cuda,
        stage: Option<u32>,
        formats: Formats,
        scan: Option<&ScanPlan>,
    ) -> Result<Self, String> {
        let named = |image: &'static [u8], name: &'static str| {
            cuda.load_named(image, name).map(|(_, kernel)| kernel)
        };
        let staged = |name: &'static str, bytes: u32| {
            cuda.load_named_shared(qwen4_moe_cubin(), name, bytes)
                .map(|(_, kernel)| kernel)
        };
        let (gate_up, gate_up_clamped, down) = if formats.fp8_experts {
            (
                "q4e_expert_gate_up_fp8",
                "q4e_expert_gate_up_fp8_clamped",
                "q4e_expert_down_fp8",
            )
        } else {
            (
                "q4e_expert_gate_up_bf16",
                "q4e_expert_gate_up_bf16_clamped",
                "q4e_expert_down_bf16",
            )
        };
        let moe = match stage {
            Some(bytes) => Some(MoeKernels {
                route: named(qwen4_moe_cubin(), "q4e_route")?,
                plan: named(moe_plan_cubin(), "moe_plan")?,
                gate_up: staged(gate_up, bytes)?,
                gate_up_clamped: staged(gate_up_clamped, bytes)?,
                down: staged(down, bytes)?,
                combine: named(qwen4_moe_cubin(), "q4e_moe_combine")?,
            }),
            None => None,
        };
        let fp8 = if formats.fp8 {
            Some(Fp8Kernels {
                gemv: named(qwen4_exp_cubin(), "q4e_gemv_tiled_e4m3_tile")?,
                tile: named(qwen4_prefill_cubin(), "q4p_dense_fp8")?,
            })
        } else {
            None
        };
        let gdn = match scan {
            Some(plan) => {
                let launch = plan.launch();
                Some(GdnKernels {
                    conv: named(qwen4_scan_cubin(), "q4p_gdn_conv")?,
                    history: named(qwen4_scan_cubin(), "q4p_gdn_conv_state")?,
                    prep: named(qwen4_scan_cubin(), "q4p_gdn_prep")?,
                    scan: named(qwen4_scan_cubin(), launch.kernel.symbol)?,
                    scan_launch: (launch.grid, launch.block),
                    gate: named(qwen4_scan_cubin(), "q4p_gdn_gate")?,
                    zero: named(qwen4_exp_cubin(), "q4e_zero")?,
                })
            }
            None => None,
        };
        Ok(Self {
            moe,
            fp8,
            gdn,
            add: named(qwen4_exp_cubin(), "q4e_add")?,
            sum_slots: named(qwen4_exp_cubin(), "q4e_sum_slots")?,
            deinterleave: named(qwen4_exp_cubin(), "q4e_deinterleave")?,
            embed: named(qwen4_exp_cubin(), "q4e_embed")?,
            gemv: named(qwen4_exp_cubin(), "q4e_gemv_bf16")?,
            packed: packed::Kernels::load(cuda)?,
            rmsnorm: named(qwen4_exp_cubin(), "q4e_rmsnorm")?,
            rope: named(qwen4_exp_cubin(), "q4e_rope")?,
            swiglu: named(qwen4_exp_cubin(), "q4e_swiglu")?,
            swiglu_clamped: named(qwen4_exp_cubin(), "q4e_swiglu_clamped")?,
            attention: named(qwen4_attn_cubin(), "q4e_attn_full")?,
            combine: named(qwen4_attn_cubin(), "q4e_attn_full_combine")?,
            cache: named(qwen4_attn_cubin(), "q4e_cache_bf16")?,
            wide: cuda
                .load_named_shared(dory_cubin(), "dory_mma_wide", WIDE_SHARED)
                .map(|(_, kernel)| kernel)?,
        })
    }
}

/// The weight formats a checkpoint declares, which pick the kernels the arm loads.
#[derive(Debug, Clone, Copy)]
struct Formats {
    /// Any projection or expert is block-scaled FP8.
    fp8: bool,
    /// The routed experts are block-scaled FP8 (else BF16).
    fp8_experts: bool,
}

impl Formats {
    fn of(architecture: &Architecture) -> Self {
        let tiled = |format: WeightFormat| matches!(format, WeightFormat::Fp8Tile { .. });
        let quantization = &architecture.quantization;
        let fp8_experts = tiled(quantization.expert_format);
        Self {
            fp8: fp8_experts || tiled(quantization.dense_format),
            fp8_experts,
        }
    }
}

/// The weights, kernels and geometry every session reads.
struct Shared {
    cuda: CudaHandle,
    kernels: Kernels,
    embed: GpuMat,
    head: GpuMat,
    norm: Deviceptr,
    layers: Vec<Layer>,
    hidden: usize,
    vocab: usize,
    eps: f32,
    zero_centred: bool,
    policy: Option<WidePolicy>,
    context: usize,
    /// The sequences' key/value caches' lanes, for sequences of `context` positions.
    lanes: Arc<SequenceLanes<LaneParts>>,
    /// Whether a sequence keeps a rewind point at the end of each prompt
    /// ([sparkflash_runtime::REWIND_ENV], on by default).
    rewind: bool,
    /// The recurrent regions of a sequence, in layer order: each Gated DeltaNet layer's conv
    /// history and delta-rule state, in bytes. Its rewind point copies these.
    linear_sizes: Vec<usize>,
    /// Every weight allocation, freed once.
    owned: Vec<Deviceptr>,
    weight_bytes: usize,
    /// The packed copies the token GEMVs read ([sparkflash_ops::packed]); `owned` frees them.
    packed: PackedSet,
    /// The group's sums of each branch's partial rows, on a tensor-parallel rank.
    sums: Option<RowSums>,
    /// Which share of each sharded axis this rank holds.
    shard: Shard,
}

impl Drop for Shared {
    fn drop(&mut self) {
        // Every sequence holds the shared state, so each has given its lane back by now.
        self.lanes.drain(&self.cuda);
        if let Some(sums) = self.sums.as_mut() {
            sums.release(&self.cuda);
        }
        let references: Vec<&Deviceptr> = self.owned.iter().collect();
        self.cuda.free_buffers(&references);
    }
}

/// A dense grouped-query checkpoint on the device.
pub struct DenseDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<Shared>,
    /// The fused engine its sessions open on once [Self::enable_fusion] (or, on rank 0 of a
    /// group, [Self::enable_group_fusion]) has run.
    fused: Option<Fusion<DenseLanes>>,
}

impl DenseDeviceBackend {
    /// Whether this build carries every kernel the arm launches.
    #[must_use]
    pub fn available() -> bool {
        sparkflash_ops::parts::dense_device_images().is_ok()
    }

    /// Opens `architecture`'s checkpoint from `source` under `descriptor`, for sequences of
    /// `context` tokens.
    ///
    /// # Errors
    /// A checkpoint the arm does not execute ([sparkflash_ops::parts::dense_device_refusal]), a
    /// missing kernel, a profile field the arm cannot honour, a [sparkflash_runtime::REWIND_ENV]
    /// that is not a switch, a tensor that is not BF16 at the shape the architecture names, a
    /// zero context or one past the checkpoint's window, a CUDA context that cannot open, or a
    /// device without the virtual memory management the lanes need.
    pub fn open(
        descriptor: ModelDescriptor,
        architecture: &Architecture,
        source: &dyn WeightSource,
        context: usize,
    ) -> Result<Self, String> {
        Self::open_at(descriptor, architecture, source, context, None)
    }

    /// [Self::open] for one rank of a tensor-parallel group over `tp`: the rank's share of each
    /// sharded tensor (as the shared [TpPlan] splits it), every branch's partial rows summed
    /// over the group before the residual takes them ([RowSums]). Every rank opens the same
    /// checkpoint and steps the same walks, which [Self::enable_group_fusion] and
    /// [Self::mirror_group] keep in step.
    ///
    /// # Errors
    /// As [Self::open], a head count or width the world does not split (an attention layer's
    /// key/value heads may be fewer than the ranks, which then share them), a block-scaled FP8
    /// share that splits a scale tile, or a collective whose device leg does not sum on the f32
    /// wire.
    pub fn open_rank(
        descriptor: ModelDescriptor,
        architecture: &Architecture,
        source: &dyn WeightSource,
        context: usize,
        tp: TpRuntime,
    ) -> Result<Self, String> {
        Self::open_at(descriptor, architecture, source, context, Some(tp))
    }

    fn open_at(
        mut descriptor: ModelDescriptor,
        architecture: &Architecture,
        source: &dyn WeightSource,
        context: usize,
        tp: Option<TpRuntime>,
    ) -> Result<Self, String> {
        if let Some(refusal) = sparkflash_ops::parts::dense_device_refusal(architecture) {
            return Err(format!("the shared dense device arm refuses it: {refusal}"));
        }
        sparkflash_ops::parts::dense_device_images()?;
        let window = architecture.max_position_embeddings;
        if context == 0 || (window != 0 && context > window) {
            return Err(format!(
                "context {context} is outside 1..={window} for the checkpoint"
            ));
        }
        let profile = sparkflash_model::profile::active();
        let policy = profile_policy(profile)?;
        let rewind = sparkflash_runtime::rewind()?;
        let names = sparkflash_ops::naming::resolve(architecture, source)?;
        let shard = match &tp {
            Some(tp) => {
                Shard(TpPlan::new(tp.world(), tp.rank()).map_err(|error| error.to_string())?)
            }
            None => Shard::whole(),
        };
        let shapes = (0..architecture.layers.len())
            .map(|index| Shape::of(architecture, index, shard))
            .collect::<Result<Vec<_>, _>>()?;
        let cuda = CudaHandle::new(Cuda::load()?);
        sparkflash_kernels::cuda_driver::enable_forward_async_launches();
        let formats = Formats::of(architecture);
        let kernels = Kernels::load(
            &cuda,
            expert_stage(architecture.hidden_size, &shapes)?,
            formats,
            linear_scan(architecture, shard)?.as_ref(),
        )?;
        let mut owned = Vec::new();
        let uploaded = upload(
            &cuda,
            architecture,
            &names,
            source,
            formats,
            &shapes,
            &mut owned,
        );
        let (embed, head, norm, layers, weight_bytes) = match uploaded {
            Ok(uploaded) => uploaded,
            Err(error) => {
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error);
            }
        };
        // The token GEMVs read a packed copy of every BF16 projection; the wide tiles keep the
        // BF16 rows, so the copy sits beside them.
        let mut packed = PackedSet::new(Form::from_env()?, Placement::Beside);
        let matrices: Vec<_> = std::iter::once(head)
            .chain(layers.iter().flat_map(Layer::bf16_projections))
            .map(|mat| (mat.pointer, mat.rows, mat.columns))
            .collect();
        let packed_bytes = match packed.pack(&cuda, &kernels.packed, &matrices, 0) {
            Ok((_, packed_bytes)) => packed_bytes,
            Err(error) => {
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error);
            }
        };
        owned.extend(packed.allocations());
        let weight_bytes = weight_bytes + packed_bytes;
        eprintln!("[dense] {}", packed.summary());
        let lanes = SequenceLanes::plan(
            &cuda,
            context,
            0,
            profile.runtime.lane_memory_mb,
            |layout| LaneParts::plan(layout, &layers, context, rewind),
        );
        let lanes = match lanes {
            Ok(lanes) => Arc::new(lanes),
            Err(error) => {
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error);
            }
        };
        let sums = match tp
            .map(|tp| RowSums::new(&cuda, tp, CHUNK_ROWS, architecture.hidden_size, DEVICE))
            .transpose()
        {
            Ok(sums) => sums,
            Err(error) => {
                lanes.drain(&cuda);
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error.to_string());
            }
        };
        eprintln!(
            "[dense] sequences of {context} positions back their caches as they grow: {}",
            lanes.describe()
        );
        descriptor.context_tokens = context;
        let linear_sizes = linear_sizes(&layers);
        Ok(Self {
            descriptor,
            fused: None,
            shared: Arc::new(Shared {
                cuda,
                kernels,
                embed,
                head,
                norm,
                layers,
                hidden: architecture.hidden_size,
                vocab: architecture.vocab_size,
                eps: architecture.norm_epsilon as f32,
                zero_centred: architecture.zero_centered_norms(),
                policy,
                context,
                lanes,
                rewind,
                linear_sizes,
                owned,
                weight_bytes,
                packed,
                sums,
                shard,
            }),
        })
    }

    /// The device bytes the weights claimed, for boot diagnostics.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.shared.weight_bytes
    }

    /// A session of its concrete type, for a caller that reads its per-layer trace.
    ///
    /// # Errors
    /// The session buffers or caches do not fit on the device.
    pub fn session(&self) -> Result<DenseDeviceSession, String> {
        DenseDeviceSession::open(Arc::clone(&self.shared))
    }

    /// Serves this backend's sessions on the shared fused engine: `capacity` lanes whose decode
    /// rows share one walk, whose prompts run a part at a time between the lanes' steps, and which
    /// keep their conversations between requests. Call it before the backend is shared with the
    /// serving driver, with the driver's concurrency as the capacity.
    ///
    /// # Errors
    /// A zero capacity, a scratch or lane the device refuses, or a thread that cannot start.
    pub fn enable_fusion(&mut self, capacity: usize) -> Result<(), BackendError> {
        if self.shared.sums.is_some() {
            return Err(BackendError::new(
                DEVICE,
                "a group's rank steps its lanes with its peers: fuse rank 0's with \
                 enable_group_fusion and mirror them on the others with mirror_group",
            ));
        }
        let arm = DenseLanes::new(Arc::clone(&self.shared), capacity)?;
        self.fused = Some(Fusion::local(arm, capacity)?);
        Ok(())
    }

    /// [Self::enable_fusion] on rank 0 of an [Self::open_rank] group: every lane's prompt, step,
    /// give-back and release is broadcast through `hub` before it runs here, and each worker
    /// replays them ([Self::mirror_group]) over its own `capacity` lanes, so lane `n` is the same
    /// sequence on every rank and the sums inside a walk line up. A worker whose socket closes
    /// ends this process, and so does a failure here after a frame went out
    /// ([sparkflash_runtime::watched_hub], [sparkflash_runtime::end_group]).
    ///
    /// # Errors
    /// A backend that is not a group's rank, then as [Self::enable_fusion], or a watch that
    /// cannot start.
    pub fn enable_group_fusion(
        &mut self,
        capacity: usize,
        hub: sparkflash_comm::control::Tp4ControlHub,
    ) -> Result<(), BackendError> {
        self.group_rank()?;
        let arm = DenseLanes::new(Arc::clone(&self.shared), capacity)?;
        self.fused = Some(Fusion::group(arm, hub, self.wire(), capacity)?);
        Ok(())
    }

    /// A worker of an [Self::open_rank] group: replays rank 0's lane frames over `capacity`
    /// lanes of this rank's own until the group can no longer serve, then ends the process
    /// ([sparkflash_runtime::mirror_until_the_group_ends]).
    ///
    /// # Errors
    /// A backend that is not a group's rank, or the lanes' scratch, before any frame.
    pub fn mirror_group(
        &self,
        control: &mut sparkflash_comm::control::Tp4ControlWorker,
        capacity: usize,
    ) -> Result<std::convert::Infallible, String> {
        self.group_rank().map_err(|error| error.to_string())?;
        let lanes = DenseLanes::new(Arc::clone(&self.shared), capacity)
            .map_err(|error| error.to_string())?;
        sparkflash_runtime::mirror_until_the_group_ends(control, &lanes, &self.wire(), capacity)
    }

    /// What the group's frames carry and rank 0 refuses: the arm serves text.
    fn wire(&self) -> TextWire {
        TextWire::new("dense", "shared dense device", self.shared.context)
    }

    /// Refuses a backend that is no group's rank: its lanes have no peers to step with.
    fn group_rank(&self) -> Result<(), BackendError> {
        if self.shared.sums.is_none() {
            return Err(BackendError::new(
                DEVICE,
                "group lanes need a group's rank (open_rank); this backend runs alone, so fuse \
                 its sessions with enable_fusion",
            ));
        }
        Ok(())
    }

    /// The rank and world this backend holds its share at: `(0, 1)` alone.
    #[must_use]
    pub fn rank(&self) -> (usize, usize) {
        (self.shared.shard.0.rank(), self.shared.shard.0.world())
    }
}

impl sparkflash_ops::EngineBuilder for DenseDeviceBackend {
    type Backend = Self;

    fn open(
        plan: &EnginePlan,
        weights: &dyn WeightSource,
        context: usize,
    ) -> Result<Self::Backend, PlanError> {
        plan.single_rank_context()?;
        let context = plan.admitted_context(context)?;
        let architecture = plan.architecture();
        let descriptor = ModelDescriptor::from_architecture(
            architecture.family.clone(),
            architecture,
            context,
            Capabilities::ALL,
        );
        Self::open(descriptor, architecture, weights, context).map_err(PlanError::Invalid)
    }
}

impl ModelBackend for DenseDeviceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        if let Some(fused) = &self.fused {
            return fused.open();
        }
        Ok(Box::new(
            self.session()
                .map_err(|error| BackendError::new(DEVICE, error))?,
        ))
    }
}

/// The wide-tile policy a typed profile asks of the arm, or the field it cannot honour.
///
/// The arm's chunk and its prompt-and-decode split are fixed by its kernels; a field it does not
/// honour is refused by name rather than accepted and dropped. It honours
/// `prefill.tensor_cores = false`, which keeps every projection on the row GEMV; its lanes
/// (`runtime.concurrency`) and lane memory (`runtime.lane_memory_mb`) are the serving engine's.
fn profile_policy(
    profile: &sparkflash_model::profile::ModelProfile,
) -> Result<Option<WidePolicy>, String> {
    use sparkflash_model::profile::LatentCacheMode;
    let refuse = |what: &str| {
        Err(format!(
            "the shared dense device arm {what}; change the profile, or serve the checkpoint on \
             the host reference (SPARKFLASH_DENSE_DEVICE=0)"
        ))
    };
    if let Some(chunk) = profile.prefill.chunk
        && chunk != CHUNK_ROWS
    {
        return refuse(&format!(
            "fixes its prompt chunk at {CHUNK_ROWS} rows, not prefill.chunk = {chunk}"
        ));
    }
    if profile.prefill.enabled == Some(false) {
        return refuse("always runs its prompt as a prefill, but prefill.enabled = false");
    }
    if profile.prefill.exact == Some(true) {
        return refuse(
            "runs its prompt projections on the wide tile and its decode rows on the row GEMV, \
             which are not bit-identical, but prefill.exact = true",
        );
    }
    if profile.decode.tensor_cores == Some(true) {
        return refuse("decodes on the row GEMV, but decode.tensor_cores = true");
    }
    if profile.speculation.enabled == Some(true) {
        return refuse("has no drafter, but speculation.enabled = true");
    }
    if profile.sequence.parallel == Some(true) || profile.sequence.overlap == Some(true) {
        return refuse("has no sequence parallelism or overlapped halves");
    }
    if matches!(
        profile.attention.latent_cache,
        Some(LatentCacheMode::F32 | LatentCacheMode::Fp8)
    ) {
        return refuse("keeps a BF16 key/value cache, not attention.latent_cache's form");
    }
    Ok(if profile.prefill.tensor_cores == Some(false) {
        None
    } else {
        Some(WidePolicy::DENSE)
    })
}

/// The expert passes' staging reservation for the widest routed layer as the rank holds it, or
/// `None` when no layer routes.
fn expert_stage(hidden: usize, shapes: &[Shape]) -> Result<Option<u32>, String> {
    let widest = shapes
        .iter()
        .filter_map(|shape| shape.routed.map(|routed| routed.intermediate))
        .max();
    widest
        .map(|intermediate| token::stage_bytes(hidden, intermediate))
        .transpose()
}

/// The Gated DeltaNet layers' scan, as the rank's plan resolves it for its share of their head
/// pair, or `None` when no layer is one.
fn linear_scan(architecture: &Architecture, shard: Shard) -> Result<Option<ScanPlan>, String> {
    let linear = architecture
        .layers
        .iter()
        .any(|layer| matches!(layer.attention.kind, AttentionKind::Gdn(_)));
    if !linear {
        return Ok(None);
    }
    let (world, rank) = (shard.0.world(), shard.0.rank());
    match ScanPlan::for_architecture(architecture, world, rank, ScanKind::Gdn, CHUNK_ROWS) {
        Ok(Some(plan)) => Ok(Some(plan)),
        Ok(None) => Err("the rank's plan has no scan for its Gated DeltaNet layers".to_owned()),
        Err(error) => Err(format!("the Gated DeltaNet scan: {error}")),
    }
}

/// Each Gated DeltaNet layer's recurrent regions in bytes, in layer order: its conv history, then
/// its delta-rule state.
fn linear_sizes(layers: &[Layer]) -> Vec<usize> {
    layers
        .iter()
        .filter_map(|layer| layer.shape.linear)
        .flat_map(|linear| [linear.geometry.conv_bytes(), linear.geometry.state_bytes()])
        .collect()
}

type Uploaded = (GpuMat, GpuMat, Deviceptr, Vec<Layer>, usize);

/// One upload's tensors: each checked at its architecture shape (BF16, or a projection's
/// block-scaled E4M3), and every allocation in `owned` as it is made, so a failure part way frees
/// what was uploaded.
struct Uploads<'a> {
    uploader: DraftUploader<'a>,
    source: &'a dyn WeightSource,
    formats: Formats,
    owned: &'a mut Vec<Deviceptr>,
    bytes: usize,
}

impl Uploads<'_> {
    fn keep(&mut self, pointer: Deviceptr, bytes: usize) -> Deviceptr {
        self.owned.push(pointer);
        self.bytes += bytes;
        pointer
    }

    fn mat(&mut self, name: &str, rows: usize, columns: usize) -> Result<GpuMat, String> {
        bf16(self.source, name, &[rows, columns])?;
        let mat = self.uploader.mat(name, rows, columns)?;
        self.keep(mat.pointer, rows * columns * 2);
        Ok(mat)
    }

    fn vector(&mut self, name: &str, width: usize) -> Result<Deviceptr, String> {
        bf16(self.source, name, &[width])?;
        let pointer = self.uploader.vector(name, width)?;
        Ok(self.keep(pointer, width * 2))
    }

    /// A projection as the checkpoint stores it: BF16, or E4M3 codes and their 128 x 128 tile
    /// scales when the checkpoint declares block-scaled FP8.
    fn proj(&mut self, name: &str, rows: usize, columns: usize) -> Result<Proj, String> {
        self.proj_runs(name, (rows, columns), &[Run::Rows(0, rows)])
    }

    /// The `runs` of a `[rows, columns]` projection this rank holds, one after another
    /// ([DraftUploader::mat_parts]), in the checkpoint's form.
    fn proj_runs(
        &mut self,
        name: &str,
        (rows, columns): (usize, usize),
        runs: &[Run],
    ) -> Result<Proj, String> {
        let spec = self.source.spec(name).map_err(|error| error.to_string())?;
        if spec.dtype != DType::F8E4M3 {
            bf16(self.source, name, &[rows, columns])?;
            let mat = self.uploader.mat_parts(name, (rows, columns), runs)?;
            self.keep(mat.pointer, mat.rows * mat.columns * 2);
            return Ok(Proj::Bf16(mat));
        }
        if !self.formats.fp8 {
            return Err(format!(
                "{name} is E4M3, but the checkpoint declares no block-scaled FP8 weights, so the \
                 shared dense device arm has no FP8 kernel for it"
            ));
        }
        let mat = self
            .uploader
            .mat_e4m3_tile_parts(name, (rows, columns), runs)?;
        self.keep(mat.weight, mat.rows * mat.columns);
        let grid = mat.rows.div_ceil(E4M3_TILE) * mat.columns.div_ceil(E4M3_TILE);
        self.keep(mat.scales, grid * 2);
        if !mat.columns.is_multiple_of(E4M3_TILE) {
            return Err(format!(
                "{name} is E4M3 over {} columns a rank; the tile kernels read whole \
                 {E4M3_TILE}-column scale tiles",
                mat.columns
            ));
        }
        Ok(Proj::Fp8(mat))
    }

    /// A SwiGLU of `width` columns, of which this rank holds `count` from `first`: the gate and up
    /// rows, the down columns.
    fn mlp(
        &mut self,
        names: &ExpertNames,
        width: usize,
        (first, count): (usize, usize),
        hidden: usize,
    ) -> Result<Mlp, String> {
        let rows = [Run::Rows(first, count)];
        Ok(Mlp {
            gate: self.proj_runs(&names.gate, (width, hidden), &rows)?,
            up: self.proj_runs(&names.up, (width, hidden), &rows)?,
            down: self.proj_runs(&names.down, (hidden, width), &[Run::Columns(first, count)])?,
        })
    }

    /// The rows of a BF16 `[rows][width]` tensor this rank holds, one after another: a
    /// depthwise filter's channels or a per-head vector's heads.
    /// An f32 vector of `elements` on the device, from an F32 tensor or a BF16 one widened (a
    /// router's correction bias, which every rank holds whole).
    fn floats(&mut self, name: &str, elements: usize) -> Result<Deviceptr, String> {
        let spec = self.source.spec(name).map_err(|error| error.to_string())?;
        if spec.elements() != elements as u128 {
            return Err(format!(
                "{name} is {:?}, expected {elements} values",
                spec.shape
            ));
        }
        let bytes = match spec.dtype {
            DType::F32 => self.uploader.read(name, elements * 4)?,
            DType::BF16 => self
                .uploader
                .read(name, elements * 2)?
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|word| {
                    f32::from_bits(u32::from(u16::from_le_bytes(*word)) << 16).to_le_bytes()
                })
                .collect(),
            other => {
                return Err(format!(
                    "{name} is {other:?}; the shared dense device arm reads an F32 or BF16 vector"
                ));
            }
        };
        let pointer = self.uploader.bytes(&bytes)?;
        Ok(self.keep(pointer, elements * 4))
    }

    fn vector_runs(
        &mut self,
        name: &str,
        (rows, width): (usize, usize),
        runs: &[Run],
    ) -> Result<Deviceptr, String> {
        let mat = self.uploader.mat_parts(name, (rows, width), runs)?;
        Ok(self.keep(mat.pointer, mat.rows * width * 2))
    }

    /// A layer's token mixer and its output projection: grouped-query attention's projections and
    /// q/k norms, or a Gated DeltaNet's projections, conv, decay vectors and gated norm.
    fn mixer(
        &mut self,
        index: usize,
        names: &AttentionNames,
        (shape, whole): (&Shape, &Shape),
        hidden: usize,
    ) -> Result<(Mixer, Proj), String> {
        let cut = shape.cut;
        match (names, shape.linear) {
            (
                AttentionNames::Gqa {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    ..
                },
                None,
            ) => {
                // A query head's rows are its query and, when gated, its gate, interleaved.
                let query_head = shape.query_rows() / shape.heads.max(1);
                let head_dim = shape.head_dim;
                let kv = [Run::Rows(cut.kv_heads * head_dim, shape.kv_width())];
                let mixer = Mixer::Gqa {
                    q: self.proj_runs(
                        q,
                        (whole.query_rows(), hidden),
                        &[Run::Rows(cut.heads * query_head, shape.query_rows())],
                    )?,
                    k: self.proj_runs(k, (whole.kv_width(), hidden), &kv)?,
                    v: self.proj_runs(v, (whole.kv_width(), hidden), &kv)?,
                    q_norm: q_norm
                        .as_deref()
                        .map(|name| self.vector(name, shape.head_dim))
                        .transpose()?,
                    k_norm: k_norm
                        .as_deref()
                        .map(|name| self.vector(name, shape.head_dim))
                        .transpose()?,
                };
                let o = self.proj_runs(
                    o,
                    (hidden, whole.query_width()),
                    &[Run::Columns(cut.heads * head_dim, shape.query_width())],
                )?;
                Ok((mixer, o))
            }
            (AttentionNames::Gdn(names), Some(linear)) => {
                let geometry = linear.geometry;
                let GdnNames {
                    qkv,
                    z,
                    b,
                    a,
                    out,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                } = names;
                let Some(all) = whole.linear.map(|linear| linear.geometry) else {
                    return Err(format!(
                        "layer {index} is a Gated DeltaNet at one rank only"
                    ));
                };
                let heads = geometry.heads;
                let (key_dim, value_dim) = (geometry.key_dim, geometry.value_dim);
                let key_width = all.key_heads * key_dim;
                // The fused projection's rows (and the conv's channels) are every query, every
                // key, then every value: a rank's share is a run of each.
                let qkv_runs = [
                    Run::Rows(cut.key_heads * key_dim, geometry.key_heads * key_dim),
                    Run::Rows(
                        key_width + cut.key_heads * key_dim,
                        geometry.key_heads * key_dim,
                    ),
                    Run::Rows(
                        2 * key_width + cut.value_heads * value_dim,
                        geometry.value_width(),
                    ),
                ];
                let values = [Run::Rows(
                    cut.value_heads * value_dim,
                    geometry.value_width(),
                )];
                let value_heads = [Run::Rows(cut.value_heads, heads)];
                let mixer = Mixer::Gdn(GdnLayer {
                    qkv: self.proj_runs(qkv, (all.conv_width(), hidden), &qkv_runs)?,
                    z: self.proj_runs(z, (all.value_width(), hidden), &values)?,
                    b: self.proj_runs(b, (all.heads, hidden), &value_heads)?,
                    a: self.proj_runs(a, (all.heads, hidden), &value_heads)?,
                    conv: self.conv(conv, (all.conv_width(), geometry.taps), &qkv_runs)?,
                    a_log: self.vector_runs(a_log, (all.heads, 1), &value_heads)?,
                    dt_bias: self.vector_runs(dt_bias, (all.heads, 1), &value_heads)?,
                    norm: self.vector(norm, value_dim)?,
                });
                let out = self.proj_runs(
                    out,
                    (hidden, all.value_width()),
                    &[Run::Columns(
                        cut.value_heads * value_dim,
                        geometry.value_width(),
                    )],
                )?;
                Ok((mixer, out))
            }
            _ => Err(format!("layer {index} names another token mixer's tensors")),
        }
    }

    /// A depthwise conv's BF16 `[width][taps]` filter, which a release stores as `[width][1][taps]`
    /// or as `[width][taps]`: the `runs` of its channels this rank holds.
    fn conv(
        &mut self,
        name: &str,
        (width, taps): (usize, usize),
        runs: &[Run],
    ) -> Result<Deviceptr, String> {
        let spec = self.source.spec(name).map_err(|error| error.to_string())?;
        if spec.dtype != DType::BF16
            || (spec.shape != [width, 1, taps] && spec.shape != [width, taps])
        {
            return Err(format!(
                "{name} is {:?} {:?}; the shared dense device arm reads it as a BF16 \
                 [{width}, 1, {taps}] filter",
                spec.dtype, spec.shape
            ));
        }
        self.vector_runs(name, (width, taps), runs)
    }

    /// A routed layer's router, experts (and their table), shared expert and its gate.
    fn routed(
        &mut self,
        index: usize,
        names: &FfnNames,
        (shape, whole): (&Shape, &Shape),
        routed: &Routed,
        hidden: usize,
    ) -> Result<LayerFfn, String> {
        let Some(all) = whole.routed else {
            return Err(format!("layer {index} routes at one rank only"));
        };
        let FfnNames::Moe {
            router,
            bias,
            experts,
            shared,
            shared_gate,
        } = names
        else {
            return Err(format!("layer {index} names a dense MLP's tensors"));
        };
        if experts.len() != routed.experts {
            return Err(format!(
                "layer {index} names {} experts, the architecture {}",
                experts.len(),
                routed.experts
            ));
        }
        let router = self.proj(router, routed.experts, hidden)?;
        let bias = match bias {
            Some(name) => Some(self.floats(name, routed.experts)?),
            None => None,
        };
        let mut table = Vec::with_capacity(routed.experts * 6 * 8);
        let fp8 = self.formats.fp8_experts;
        for (number, expert) in experts.iter().enumerate() {
            let Mlp { gate, up, down } = self.mlp(
                expert,
                all.intermediate,
                (shape.cut.expert, routed.intermediate),
                hidden,
            )?;
            for (proj, name) in [
                (&gate, &expert.gate),
                (&up, &expert.up),
                (&down, &expert.down),
            ] {
                if matches!(proj, Proj::Fp8(_)) != fp8 {
                    return Err(format!(
                        "layer {index}'s expert {number} stores {name} as {}, and the checkpoint \
                         declares {} experts, the one form the expert passes are loaded in",
                        if fp8 { "BF16" } else { "E4M3" },
                        if fp8 { "block-scaled FP8" } else { "BF16" }
                    ));
                }
                for address in proj.table() {
                    table.extend(address.to_le_bytes());
                }
            }
        }
        let table = self.uploader.bytes(&table)?;
        let table = self.keep(table, routed.experts * 6 * 8);
        let shared = match (shared, shape.intermediate) {
            (Some(names), width) if width > 0 => Some(self.mlp(
                names,
                whole.intermediate,
                (shape.cut.intermediate, width),
                hidden,
            )?),
            (None, 0) => None,
            _ => {
                return Err(format!(
                    "layer {index}'s shared expert and the architecture's width {} disagree",
                    shape.intermediate
                ));
            }
        };
        let shared_gate = match shared_gate {
            Some(name) => {
                // A release stores the gate as `[1][hidden]` or as `[hidden]`.
                let spec = self.source.spec(name).map_err(|error| error.to_string())?;
                if spec.dtype != DType::BF16 || spec.elements() != hidden as u128 {
                    return Err(format!(
                        "{name} is {:?} {:?}; the shared dense device arm reads it as a BF16 \
                         {hidden}-wide gate",
                        spec.dtype, spec.shape
                    ));
                }
                let pointer = self.uploader.vector(name, hidden)?;
                Some(Proj::Bf16(GpuMat {
                    pointer: self.keep(pointer, hidden * 2),
                    rows: 1,
                    columns: hidden,
                }))
            }
            None => None,
        };
        if shared_gate.is_some() && shared.is_none() {
            return Err(format!(
                "layer {index} names a shared-expert gate without a shared expert"
            ));
        }
        Ok(LayerFfn::Routed {
            router,
            bias,
            table,
            shared,
            shared_gate,
        })
    }
}

/// Uploads every tensor `names` resolves, each checked at its architecture shape (BF16, or a
/// projection's block-scaled E4M3 under `formats`); every allocation lands in `owned` as it is
/// made, so a failure part way frees what was uploaded.
#[allow(clippy::too_many_arguments)]
fn upload(
    cuda: &Cuda,
    architecture: &Architecture,
    names: &TensorNames,
    source: &dyn WeightSource,
    formats: Formats,
    shapes: &[Shape],
    owned: &mut Vec<Deviceptr>,
) -> Result<Uploaded, String> {
    let mut up = Uploads {
        uploader: DraftUploader::new(cuda, source),
        source,
        formats,
        owned,
        bytes: 0,
    };
    let hidden = architecture.hidden_size;
    let vocab = architecture.vocab_size;
    let embed = up.mat(&names.embed, vocab, hidden)?;
    let head = if names.head_tied() {
        embed
    } else {
        up.mat(&names.head, vocab, hidden)?
    };
    let norm = up.vector(&names.norm, hidden)?;
    let mut layers = Vec::with_capacity(architecture.layers.len());
    if names.layers.len() != shapes.len() {
        return Err(format!(
            "the checkpoint names {} layers, the architecture {}",
            names.layers.len(),
            shapes.len()
        ));
    }
    for (index, (layer_names, shape)) in names.layers.iter().zip(shapes).enumerate() {
        let whole = Shape::of(architecture, index, Shard::whole())?;
        let pair = (shape, &whole);
        let (mixer, o) = up.mixer(index, &layer_names.attention, pair, hidden)?;
        let ffn = match (&shape.routed, &layer_names.ffn) {
            (None, FfnNames::Dense(names)) => LayerFfn::Dense(up.mlp(
                names,
                whole.intermediate,
                (shape.cut.intermediate, shape.intermediate),
                hidden,
            )?),
            (None, _) => return Err(format!("layer {index} names a routed MLP's tensors")),
            (Some(routed), names) => up.routed(index, names, pair, routed, hidden)?,
        };
        layers.push(Layer {
            shape: *shape,
            input_norm: up.vector(&layer_names.input_norm, hidden)?,
            mixer,
            o,
            post_norm: up.vector(&layer_names.post_norm, hidden)?,
            ffn,
        });
    }
    Ok((embed, head, norm, layers, up.bytes))
}

/// Refuses a tensor that is not BF16 at `shape`.
fn bf16(source: &dyn WeightSource, name: &str, shape: &[usize]) -> Result<(), String> {
    let spec = source.spec(name).map_err(|error| error.to_string())?;
    if spec.dtype != DType::BF16 || spec.shape != shape {
        return Err(format!(
            "{name} is {:?} {:?}; the shared dense device arm reads it as BF16 {shape:?}",
            spec.dtype, spec.shape
        ));
    }
    Ok(())
}

impl Shared {
    /// A branch's `rows` partial rows at `partial` summed over the group, in place: nothing at one
    /// rank.
    fn sum(&self, partial: Deviceptr, rows: usize) -> Result<(), BackendError> {
        match &self.sums {
            None => Ok(()),
            Some(sums) => sums.sum(&self.cuda, None, partial, rows, |call| {
                self.call(self.kernels.sum_slots, call)
            }),
        }
    }

    fn call<const N: usize>(&self, kernel: Kernel, call: Call<N>) -> Result<(), BackendError> {
        self.cuda
            .launch_async(kernel, call.grid, call.block, &call.args)
            .map_err(|error| BackendError::new(DEVICE, format!("{}: {error}", kernel.name())))
    }

    /// A grouped-query attention layer over the normed rows in `b.normed`, into `b.attended`: the
    /// row-wise projections and norms over every row, the rotary, cache appends and attention a
    /// segment at a time.
    fn attend(
        &self,
        index: usize,
        layer: &Layer,
        b: &Buffers,
        segments: &[(&mut Sequence, &[u32])],
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        let Mixer::Gqa {
            q,
            k,
            v,
            q_norm,
            k_norm,
        } = &layer.mixer
        else {
            return Err(BackendError::new(
                DEVICE,
                format!("layer {index} is not an attention layer"),
            ));
        };
        let kernels = self.kernels;
        let shape = layer.shape;
        let (query_width, kv_width) = (shape.query_width(), shape.kv_width());
        let projected_query = if shape.gated { b.interleaved } else { b.query };
        self.project(q, b, b.normed.0, projected_query.0, rows, policy)?;
        self.project(k, b, b.normed.0, b.key.0, rows, policy)?;
        self.project(v, b, b.normed.0, b.value.0, rows, policy)?;
        if shape.gated {
            self.call(
                kernels.deinterleave,
                calls::deinterleave(
                    b.interleaved.0,
                    (b.query.0, b.gate.0),
                    (shape.heads, shape.head_dim),
                    rows,
                ),
            )?;
        }
        for (gains, values, width) in [(*q_norm, b.query, query_width), (*k_norm, b.key, kv_width)]
        {
            if let Some(gains) = gains {
                self.norm(
                    RmsNorm::new(values.0, gains.0, values.0, width, self.eps)
                        .groups(shape.head_dim, true),
                    rows,
                )?;
            }
        }
        let mut first = 0usize;
        for (sequence, tokens) in segments.iter() {
            let count = tokens.len();
            let position = sequence.position();
            let row = |base: Deviceptr, width: usize| base.0 + (first * width * 4) as u64;
            if shape.rotary >= 2 {
                for (values, heads, width) in [
                    (b.query, shape.heads, query_width),
                    (b.key, shape.kv_heads, kv_width),
                ] {
                    let rope = Rope {
                        rows: row(values, width),
                        heads,
                        head_dim: shape.head_dim,
                        rotary: shape.rotary,
                        position,
                        theta: shape.theta,
                        row_stride: width,
                        tokens: count,
                    };
                    self.call(kernels.rope, rope.call())?;
                }
            }
            let Cache::Kv(keys, values) = sequence.caches[index] else {
                return Err(BackendError::new(
                    DEVICE,
                    format!("layer {index}'s lane part holds no key/value cache"),
                ));
            };
            let at = (position * kv_width * 2) as u64;
            self.call(
                kernels.cache,
                calls::cache_bf16(keys.0 + at, row(b.key, kv_width), count * kv_width),
            )?;
            self.call(
                kernels.cache,
                calls::cache_bf16(values.0 + at, row(b.value, kv_width), count * kv_width),
            )?;
            // The partials are one segment's: the combine reads them before the next segment's
            // attention writes them, both on the one stream.
            let attention = FullAttention {
                query: row(b.query, query_width),
                keys: keys.0,
                values: values.0,
                partials: b.partials.0,
                position,
                rows: count,
                heads: shape.heads,
                kv_heads: shape.kv_heads,
                head_dim: shape.head_dim,
            };
            self.call(kernels.attention, attention.call())?;
            let gate = if shape.gated {
                row(b.gate, query_width)
            } else {
                0
            };
            self.call(
                kernels.combine,
                attention.combine(gate, row(b.attended, query_width)),
            )?;
            first += count;
        }
        Ok(())
    }

    /// A Gated DeltaNet layer over the normed rows in `b.normed`, into `b.attended`: the
    /// projections over every row; a segment at a time, the conv over its rows from its history
    /// and its history advanced past them; the prep over every row; a segment at a time, the scan
    /// from its state, in place; and the gated norm over every row. Every row count takes these
    /// kernels, so a row's bits do not depend on what it shares a walk with.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        index: usize,
        layer: &Layer,
        gdn: &GdnLayer,
        b: &Buffers,
        segments: &[(&mut Sequence, &[u32])],
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        let refuse = |message: String| Err(BackendError::new(DEVICE, message));
        let (Some(kernels), Some(linear)) = (self.kernels.gdn, layer.shape.linear) else {
            return refuse(format!("layer {index} has no Gated DeltaNet kernels"));
        };
        let geometry = linear.geometry;
        let width = geometry.conv_width();
        let conv = (width, geometry.taps);
        self.project(&gdn.qkv, b, b.normed.0, b.gdn_raw.0, rows, policy)?;
        self.project(&gdn.z, b, b.normed.0, b.gdn_z.0, rows, policy)?;
        self.project(&gdn.b, b, b.normed.0, b.gdn_b.0, rows, policy)?;
        self.project(&gdn.a, b, b.normed.0, b.gdn_a.0, rows, policy)?;
        let mut first = 0usize;
        for (sequence, tokens) in segments {
            let Cache::Linear { conv: history, .. } = sequence.caches[index] else {
                return refuse(format!(
                    "layer {index}'s lane part holds no recurrent state"
                ));
            };
            let count = tokens.len();
            let at = (first * width * 4) as u64;
            self.call(
                kernels.conv,
                gdn::conv_rows(
                    b.gdn_raw.0 + at,
                    history.0,
                    gdn.conv.0,
                    b.gdn_mixed.0 + at,
                    conv,
                    count,
                ),
            )?;
            self.call(
                kernels.history,
                gdn::conv_history(b.gdn_raw.0 + at, history.0, conv, count, (0, 0, 0)),
            )?;
            first += count;
        }
        self.call(
            kernels.prep,
            geometry.prep(
                b.gdn_mixed.0,
                b.gdn_a.0,
                b.gdn_b.0,
                gdn.a_log.0,
                gdn.dt_bias.0,
                b.gdn_qk.0,
                b.gdn_decay.0,
                rows,
            ),
        )?;
        let (grid, block) = kernels.scan_launch;
        let qk_row = geometry.key_heads * gdn_qk_row(geometry.key_dim) * 4;
        let mut first = 0usize;
        for (sequence, tokens) in segments {
            let Cache::Linear { state, .. } = sequence.caches[index] else {
                return refuse(format!(
                    "layer {index}'s lane part holds no recurrent state"
                ));
            };
            let count = tokens.len();
            let args = geometry.scan_args(
                (
                    b.gdn_qk.0 + (first * qk_row) as u64,
                    b.gdn_decay.0 + (first * geometry.heads * 8) as u64,
                    b.gdn_mixed.0 + (first * width * 4) as u64,
                ),
                (state.0, state.0),
                b.attended.0 + (first * geometry.value_width() * 4) as u64,
                count,
            );
            self.cuda
                .launch_async(kernels.scan, grid, block, &args)
                .map_err(|error| {
                    BackendError::new(DEVICE, format!("{}: {error}", kernels.scan.name()))
                })?;
            first += count;
        }
        self.call(
            kernels.gate,
            gdn::gated_norm(
                (b.attended.0, b.gdn_z.0, gdn.norm.0, b.attended.0),
                (geometry.heads, geometry.value_dim),
                self.eps,
                linear.sigmoid,
                rows,
            ),
        )
    }

    /// `output[rows][proj.rows] = input[rows][proj.columns] x projᵀ` in the projection's format,
    /// on the kernel [route_bf16] or [route_fp8] picks under `policy` (`None` keeps the row GEMV);
    /// an FP8 GEMV splits its sums in `b`'s tiled scratch.
    fn project(
        &self,
        proj: &Proj,
        b: &Buffers,
        input: u64,
        output: u64,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        match proj {
            Proj::Bf16(mat) => self.project_bf16(mat, input, output, rows, policy),
            Proj::Fp8(mat) => self.project_fp8(mat, b, input, output, rows, policy),
        }
    }

    /// A block-scaled FP8 projection: the FP8 tensor-core tile ([Fp8Tile]) where [route_fp8]
    /// takes it and the operands are 16-byte aligned, else the tiled E4M3 GEMV a group of rows at
    /// a time. Both keep the activations f32 and apply each tile's scale to its sums, the host
    /// reference's arithmetic to a tolerance.
    fn project_fp8(
        &self,
        mat: &GpuMatE4m3Tile,
        b: &Buffers,
        input: u64,
        output: u64,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        let Some(kernels) = self.kernels.fp8 else {
            return Err(BackendError::new(
                DEVICE,
                "an FP8 projection runs without the FP8 kernels",
            ));
        };
        let tiled = Tiled {
            shared: self,
            partials: b.tiled_partials,
            counters: b.tiled_counters,
        };
        let gemv = |input: u64, output: u64, batch: usize| {
            gemv_e4m3_tile(
                &tiled,
                kernels.gemv,
                mat,
                Deviceptr(input),
                Deviceptr(output),
                batch,
            )
        };
        let batched = || {
            for start in (0..rows).step_by(GEMV_MAX_BATCH) {
                gemv(
                    input + (start * mat.columns * 4) as u64,
                    output + (start * mat.rows * 4) as u64,
                    GEMV_MAX_BATCH.min(rows - start),
                )?;
            }
            Ok(())
        };
        let aligned = [mat.weight.0, input, output]
            .iter()
            .all(|address| address.is_multiple_of(16));
        match route_fp8(policy, mat.rows, mat.columns, rows) {
            Route::RowGemv => gemv(input, output, rows),
            Route::Wide if aligned => {
                let tile = Fp8Tile::call(
                    mat.weight.0,
                    mat.scales.0,
                    input,
                    output,
                    mat.rows,
                    mat.columns,
                    rows,
                )
                .map_err(|error| BackendError::new(DEVICE, error))?;
                self.cuda
                    .launch_async(kernels.tile, tile.grid, tile.block, &tile.args)
                    .map_err(|error| BackendError::new(DEVICE, format!("q4p_dense_fp8: {error}")))
            }
            Route::Wide | Route::BatchedGemv => batched(),
            route @ (Route::Skinny | Route::Refuse) => Err(BackendError::new(
                DEVICE,
                format!(
                    "the dense policy routed an FP8 {} x {} projection to {route:?}",
                    mat.rows, mat.columns
                ),
            )),
        }
    }

    /// `output[rows][mat.rows] = input[rows][mat.columns] x matᵀ` on the kernel
    /// [route_bf16] picks under `policy` (`None` keeps the row GEMV).
    fn project_bf16(
        &self,
        mat: &GpuMat,
        input: u64,
        output: u64,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        // The packed copy where there is one and the activations are a 16-byte load: the same
        // bits from about three quarters of the bytes.
        let packed = self.packed.get(mat.pointer);
        let gemv = |input: u64, output: u64, batch: usize| match packed
            .filter(|_| input.is_multiple_of(16))
        {
            Some(rows) => self.call(
                self.kernels.packed.gemv,
                packed::gemv(
                    rows,
                    Deviceptr(input),
                    Deviceptr(output),
                    (mat.rows, mat.columns),
                    batch,
                    1,
                )
                .map_err(|error| BackendError::new(DEVICE, error))?,
            ),
            None => self.call(
                self.kernels.gemv,
                calls::gemv_bf16(mat.pointer.0, input, output, (mat.rows, mat.columns), batch),
            ),
        };
        match route_bf16(policy, mat.rows, mat.columns, rows) {
            Route::RowGemv => gemv(input, output, rows),
            Route::BatchedGemv => {
                for start in (0..rows).step_by(GEMV_MAX_BATCH) {
                    gemv(
                        input + (start * mat.columns * 4) as u64,
                        output + (start * mat.rows * 4) as u64,
                        GEMV_MAX_BATCH.min(rows - start),
                    )?;
                }
                Ok(())
            }
            Route::Wide => {
                let tile =
                    WideTile::call(mat.pointer.0, input, output, mat.rows, mat.columns, rows)
                        .map_err(|error| BackendError::new(DEVICE, error))?;
                self.cuda
                    .launch_async(self.kernels.wide, tile.grid, tile.block, &tile.args)
                    .map_err(|error| BackendError::new(DEVICE, format!("dory_mma_wide: {error}")))
            }
            route @ (Route::Skinny | Route::Refuse) => Err(BackendError::new(
                DEVICE,
                format!(
                    "the dense policy routed a {} x {} projection to {route:?}",
                    mat.rows, mat.columns
                ),
            )),
        }
    }

    fn norm(&self, plan: RmsNorm, rows: usize) -> Result<(), BackendError> {
        self.call(
            self.kernels.rmsnorm,
            plan.zero_centred(self.zero_centred).call(rows),
        )
    }

    fn launch(
        &self,
        kernel: Kernel,
        (grid, block): Launch,
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.cuda
            .launch_async(kernel, grid, block, args)
            .map_err(|error| BackendError::new(DEVICE, format!("{}: {error}", kernel.name())))
    }

    /// `output = down(silu(gate(normed)) * up(normed))` over `rows`, in the MLP scratch, the gate
    /// and up clamped first under the layer's `clamp`.
    fn mlp(
        &self,
        mlp: &Mlp,
        clamp: Option<f32>,
        b: &Buffers,
        output: Deviceptr,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        self.project(&mlp.gate, b, b.normed.0, b.ffn_gate.0, rows, policy)?;
        self.project(&mlp.up, b, b.normed.0, b.ffn_up.0, rows, policy)?;
        let count = rows * mlp.gate.rows();
        match clamp {
            None => self.call(
                self.kernels.swiglu,
                calls::swiglu(b.ffn_gate.0, b.ffn_up.0, b.ffn_up.0, count),
            )?,
            Some(limit) => self.call(
                self.kernels.swiglu_clamped,
                calls::swiglu_clamped(b.ffn_gate.0, b.ffn_up.0, b.ffn_up.0, count, limit),
            )?,
        }
        self.project(&mlp.down, b, b.ffn_up.0, output.0, rows, policy)
    }

    /// A routed layer's MLP over `rows` into `b.projected`: the router on the row GEMV (so a
    /// prompt row picks the experts its decode row would), `q4e_route`, the shared plan's runs,
    /// the BF16 token passes, the shared expert and its gate, and the combine.
    fn routed(
        &self,
        layer: &Layer,
        b: &Buffers,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        let (
            Some(routed),
            LayerFfn::Routed {
                router,
                bias,
                table,
                shared,
                shared_gate,
            },
            Some(kernels),
        ) = (&layer.shape.routed, &layer.ffn, self.kernels.moe)
        else {
            return Err(BackendError::new(
                DEVICE,
                "a routed layer runs without its experts or the routed kernels",
            ));
        };
        let hidden = self.hidden;
        let (experts, top_k, intermediate) = (routed.experts, routed.top_k, routed.intermediate);
        self.project(router, b, b.normed.0, b.router.0, rows, None)?;
        self.launch(
            kernels.route,
            ((rows as u32, 1, 1), (256, 1, 1)),
            &[
                b.router.0,
                bias.map_or(0, |bias| bias.0),
                b.indices.0,
                b.weights.0,
                experts as u64,
                top_k as u64,
                routed.score,
                u64::from(routed.renormalise),
                u64::from(routed.scaling.to_bits()),
            ],
        )?;
        let plan = MoePlanCall::runs(
            b.indices.0,
            b.sorted.0,
            b.runs.0,
            (rows * top_k, experts, top_k),
        );
        self.launch(kernels.plan, MoePlanCall::launch(), &plan.args())?;
        let gate_up = [
            table.0,
            b.indices.0,
            b.sorted.0,
            b.runs.0,
            b.normed.0,
            b.activated.0,
            top_k as u64,
            hidden as u64,
            intermediate as u64,
        ];
        let grid = token::gate_up_grid(intermediate, top_k, rows);
        match layer.shape.clamp {
            None => self.launch(kernels.gate_up, grid, &gate_up)?,
            Some(limit) => {
                let mut clamped = gate_up.to_vec();
                clamped.push(u64::from(limit.to_bits()));
                self.launch(kernels.gate_up_clamped, grid, &clamped)?;
            }
        }
        self.launch(
            kernels.down,
            token::down_grid(hidden, top_k, rows),
            &[
                table.0,
                b.indices.0,
                b.sorted.0,
                b.runs.0,
                b.weights.0,
                b.activated.0,
                b.expert_partials.0,
                top_k as u64,
                hidden as u64,
                intermediate as u64,
                top_k as u64,
            ],
        )?;
        match shared {
            Some(mlp) => self.mlp(mlp, layer.shape.clamp, b, b.shared_out, rows, policy)?,
            // The combine always adds a shared row; a layer without one adds zeros.
            None => self
                .cuda
                .memset_d8_async(b.shared_out, 0, rows * hidden * 4)
                .map_err(|error| BackendError::new(DEVICE, format!("the shared row: {error}")))?,
        }
        if let Some(gate) = shared_gate {
            self.project(gate, b, b.normed.0, b.shared_logit.0, rows, None)?;
        }
        self.launch(
            kernels.combine,
            combine_grid(hidden, rows),
            &[
                b.projected.0,
                b.expert_partials.0,
                b.shared_out.0,
                if shared_gate.is_some() {
                    b.shared_logit.0
                } else {
                    0
                },
                0,
                0,
                top_k as u64,
                hidden as u64,
                0,
            ],
        )
    }
}

/// The arm as the shared tiled GEMVs' device ([GemvDevice]): its stream and one walk's split
/// sums and per-tile counters.
struct Tiled<'a> {
    shared: &'a Shared,
    partials: Deviceptr,
    counters: Deviceptr,
}

impl GemvDevice for Tiled<'_> {
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.shared.launch(kernel, (grid, block), args)
    }

    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        self.shared
            .project_bf16(matrix, input.0, output.0, rows, None)
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        (self.partials, self.counters)
    }
}

/// Where each layer's caches lie in a sequence's lane ([SequenceLanes], the lane memory every
/// other arm's sequences use), backed from one pool the backend's sequences share: an attention
/// layer's key and value caches, a `[capacity][kv_width]` BF16 run each backed only as far as the
/// sequence has reached, and a Gated DeltaNet layer's conv history and delta-rule state, f32 the
/// lane backs whole. With a rewind point the lane also holds its [Checkpoint] arena.
#[derive(Debug)]
struct LaneParts {
    caches: Vec<LanePart>,
    arena: Option<usize>,
}

/// One layer's offsets in the lane.
#[derive(Debug, Clone, Copy)]
enum LanePart {
    /// The keys and the values.
    Kv([usize; 2]),
    /// The conv history and the delta-rule state.
    Linear([usize; 2]),
}

impl LaneParts {
    fn plan(layout: &mut Layout, layers: &[Layer], capacity: usize, rewind: bool) -> Self {
        let caches = layers
            .iter()
            .map(|layer| match layer.shape.linear {
                Some(linear) => LanePart::Linear([
                    layout.part(linear.geometry.conv_bytes(), 0, 0),
                    layout.part(linear.geometry.state_bytes(), 0, 0),
                ]),
                None => {
                    let row = layer.shape.kv_width() * 2;
                    LanePart::Kv([
                        layout.part(capacity * row, row, 0),
                        layout.part(capacity * row, row, 0),
                    ])
                }
            })
            .collect();
        let sizes = linear_sizes(layers);
        let arena = (rewind && !sizes.is_empty())
            .then(|| layout.part(Checkpoint::arena_bytes(&sizes), 0, 0));
        Self { caches, arena }
    }
}

/// The f32 scratch one walk of [CHUNK_ROWS] rows takes, sized for the widest layer, and room for
/// `logit_rows` rows of logits.
struct Buffers {
    tokens: Deviceptr,
    x: Deviceptr,
    normed: Deviceptr,
    interleaved: Deviceptr,
    query: Deviceptr,
    gate: Deviceptr,
    key: Deviceptr,
    value: Deviceptr,
    partials: Deviceptr,
    attended: Deviceptr,
    projected: Deviceptr,
    ffn_gate: Deviceptr,
    ffn_up: Deviceptr,
    logits: Deviceptr,
    /// A routed layer's router logits, `[rows][experts]`.
    router: Deviceptr,
    /// The picked experts (i32) and their weights, `[rows][top_k]` each.
    indices: Deviceptr,
    weights: Deviceptr,
    /// The slots in expert order (i32), and each slot's run (two i32 a slot).
    sorted: Deviceptr,
    runs: Deviceptr,
    /// Every slot's SwiGLU row, `[rows * top_k][intermediate]`.
    activated: Deviceptr,
    /// Every slot's weighted down row, `[rows * top_k][hidden]`.
    expert_partials: Deviceptr,
    /// The shared expert's rows (zeros without one), and its gate's logit a row.
    shared_out: Deviceptr,
    shared_logit: Deviceptr,
    /// The tiled E4M3 GEMV's split sums, and its per-tile counters (zero between launches: the
    /// kernel resets each counter it uses).
    tiled_partials: Deviceptr,
    tiled_counters: Deviceptr,
    /// A Gated DeltaNet layer's fused q/k/v projection and its convolved rows, `[rows][conv_width]`
    /// each; its output gate's projection, `[rows][value_width]`; its beta and decay logits,
    /// `[rows][heads]` each; the prep's normalised queries and keys and its decay and beta pairs.
    gdn_raw: Deviceptr,
    gdn_mixed: Deviceptr,
    gdn_z: Deviceptr,
    gdn_b: Deviceptr,
    gdn_a: Deviceptr,
    gdn_qk: Deviceptr,
    gdn_decay: Deviceptr,
    logit_rows: usize,
}

impl Buffers {
    fn all(&self) -> [&Deviceptr; 32] {
        [
            &self.tokens,
            &self.x,
            &self.normed,
            &self.interleaved,
            &self.query,
            &self.gate,
            &self.key,
            &self.value,
            &self.partials,
            &self.attended,
            &self.projected,
            &self.ffn_gate,
            &self.ffn_up,
            &self.logits,
            &self.router,
            &self.indices,
            &self.weights,
            &self.sorted,
            &self.runs,
            &self.activated,
            &self.expert_partials,
            &self.shared_out,
            &self.shared_logit,
            &self.tiled_partials,
            &self.tiled_counters,
            &self.gdn_raw,
            &self.gdn_mixed,
            &self.gdn_z,
            &self.gdn_b,
            &self.gdn_a,
            &self.gdn_qk,
            &self.gdn_decay,
        ]
    }
}

/// A walk's scratch on the device, freed with it. Nothing in it outlives a walk but the logits the
/// walk's caller reads back, so the slots of one engine share one.
struct Scratch {
    shared: Arc<Shared>,
    buffers: Buffers,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        self.shared.cuda.free_buffers(&self.buffers.all());
    }
}

impl Scratch {
    fn new(shared: Arc<Shared>, logit_rows: usize) -> Result<Self, String> {
        let widest = |of: fn(&Shape) -> usize| {
            shared
                .layers
                .iter()
                .map(|layer| of(&layer.shape))
                .max()
                .unwrap_or(0)
        };
        let rows = CHUNK_ROWS;
        let hidden = shared.hidden;
        let logit_rows = logit_rows.clamp(1, CHUNK_ROWS);
        let routed = |of: fn(&Routed) -> usize| {
            shared
                .layers
                .iter()
                .filter_map(|layer| layer.shape.routed.as_ref())
                .map(of)
                .max()
                .unwrap_or(0)
        };
        let (experts, slots) = (routed(|r| r.experts), rows * routed(|r| r.top_k));
        let any_routed = slots > 0;
        let fp8 = shared.kernels.fp8.is_some();
        let linear = |of: fn(&GdnGeometry) -> usize| {
            shared
                .layers
                .iter()
                .filter_map(|layer| layer.shape.linear.as_ref())
                .map(|linear| of(&linear.geometry))
                .max()
                .unwrap_or(0)
        };
        let floats = [
            rows,
            rows * hidden,
            rows * hidden,
            rows * widest(Shape::query_rows),
            rows * widest(Shape::query_width),
            rows * widest(Shape::query_width),
            rows * widest(Shape::kv_width),
            rows * widest(Shape::kv_width),
            widest(|shape| FullAttention::partial_floats(CHUNK_ROWS, shape.heads, shape.head_dim)),
            rows * widest(Shape::attended_width),
            rows * hidden,
            rows * widest(|shape| shape.intermediate),
            rows * widest(|shape| shape.intermediate),
            logit_rows * shared.vocab,
            rows * experts,
            slots,
            slots,
            slots,
            2 * slots,
            slots * routed(|r| r.intermediate),
            slots * hidden,
            if any_routed { rows * hidden } else { 0 },
            if any_routed { rows } else { 0 },
            if fp8 { TILED_GEMV_PARTIALS } else { 0 },
            if fp8 { TILED_GEMV_BLOCKS } else { 0 },
            rows * linear(GdnGeometry::conv_width),
            rows * linear(GdnGeometry::conv_width),
            rows * linear(GdnGeometry::value_width),
            rows * linear(|geometry| geometry.heads),
            rows * linear(|geometry| geometry.heads),
            linear(|geometry| geometry.qk_floats(CHUNK_ROWS)),
            rows * linear(|geometry| 2 * geometry.heads),
        ];
        let mut pointers = Vec::with_capacity(floats.len());
        let mut total = 0usize;
        for count in floats {
            match shared.cuda.alloc(count.max(1) * 4, &mut total) {
                Ok(pointer) => pointers.push(pointer),
                Err(error) => {
                    let references: Vec<&Deviceptr> = pointers.iter().collect();
                    shared.cuda.free_buffers(&references);
                    return Err(error);
                }
            }
        }
        let p = |index: usize| pointers[index];
        let buffers = Buffers {
            tokens: p(0),
            x: p(1),
            normed: p(2),
            interleaved: p(3),
            query: p(4),
            gate: p(5),
            key: p(6),
            value: p(7),
            partials: p(8),
            attended: p(9),
            projected: p(10),
            ffn_gate: p(11),
            ffn_up: p(12),
            logits: p(13),
            router: p(14),
            indices: p(15),
            weights: p(16),
            sorted: p(17),
            runs: p(18),
            activated: p(19),
            expert_partials: p(20),
            shared_out: p(21),
            shared_logit: p(22),
            tiled_partials: p(23),
            tiled_counters: p(24),
            gdn_raw: p(25),
            gdn_mixed: p(26),
            gdn_z: p(27),
            gdn_b: p(28),
            gdn_a: p(29),
            gdn_qk: p(30),
            gdn_decay: p(31),
            logit_rows,
        };
        // The counters start at zero; every tiled launch leaves them so.
        if fp8
            && let Err(error) = shared
                .cuda
                .copy_htod(&buffers.tiled_counters, &[0u8; TILED_GEMV_BLOCKS * 4])
        {
            shared.cuda.free_buffers(&buffers.all());
            return Err(error);
        }
        Ok(Self { shared, buffers })
    }
}

/// One sequence on the device: its lane of BF16 key/value caches and recurrent state, the ids
/// whose rows they hold, the point it can go back to, and its sampler.
///
/// Every cache row is addressed by its position and attention reads only the rows before a
/// query's own, so an attention layer goes back to any prefix of its history by moving the
/// position. A Gated DeltaNet layer's state is the state after the last row, so its rewind point
/// is a copy ([Checkpoint]) of every such layer's state at the end of the last prompt, which a
/// rewind copies back (as the recurrent arms' rewind points do).
pub struct Sequence {
    shared: Arc<Shared>,
    lane: SequenceLane<LaneParts>,
    /// Each layer's caches in the lane.
    caches: Vec<Cache>,
    /// The Gated DeltaNet layers' rewind point, when a layer is one and the sequence keeps one.
    checkpoint: Option<Checkpoint>,
    /// The ids whose rows the caches hold, in position order: the last prompt and every token
    /// stepped since. Its length is the sequence's position.
    history: Vec<u32>,
    /// The end of the last prompt, when [sparkflash_runtime::REWIND_ENV] keeps one.
    point: Option<usize>,
    /// How many of the last prompt's ids the sequence already held.
    cached: usize,
    /// A prompt running a part at a time, and the next id of it to forward.
    prompt: Option<(Vec<u32>, usize)>,
    sampler: Sampler,
    sampling: Sampling,
}

impl Drop for Sequence {
    fn drop(&mut self) {
        self.lane.free(&self.shared.cuda);
    }
}

impl Sequence {
    fn open(shared: Arc<Shared>) -> Result<Self, String> {
        let lane = SequenceLane::reserve(&shared.cuda, &shared.lanes)?;
        let base = lane.base();
        let parts = shared.lanes.parts();
        let caches = parts
            .caches
            .iter()
            .map(|part| match *part {
                LanePart::Kv([keys, values]) => Cache::Kv(base.offset(keys), base.offset(values)),
                LanePart::Linear([conv, state]) => Cache::Linear {
                    conv: base.offset(conv),
                    state: base.offset(state),
                },
            })
            .collect();
        let checkpoint = parts
            .arena
            .map(|arena| Checkpoint::new(base.offset(arena), shared.linear_sizes.clone()));
        Ok(Self {
            shared,
            lane,
            caches,
            checkpoint,
            history: Vec::new(),
            point: None,
            cached: 0,
            prompt: None,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
        })
    }

    fn position(&self) -> usize {
        self.history.len()
    }

    /// The Gated DeltaNet layers' live regions in layer order, as [Shared::linear_sizes] counts
    /// them: each layer's conv history, then its state.
    fn linear_regions(&self) -> Vec<Deviceptr> {
        self.caches
            .iter()
            .filter_map(|cache| match *cache {
                Cache::Linear { conv, state } => Some([conv, state]),
                Cache::Kv(..) => None,
            })
            .flatten()
            .collect()
    }

    /// Sets the recurrent state to the empty sequence's: zero, as a fresh lane's chunks map. A
    /// lane with nothing backed has nothing to clear.
    fn clear_linear(&mut self) -> Result<(), BackendError> {
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint.forget();
        }
        let Some(kernels) = self.shared.kernels.gdn else {
            return Ok(());
        };
        if self.lane.span().backed() == 0 {
            return Ok(());
        }
        let regions = self.linear_regions();
        for (region, bytes) in regions.iter().zip(&self.shared.linear_sizes) {
            self.shared
                .call(kernels.zero, calls::zero(region.0, bytes / 4))?;
        }
        Ok(())
    }

    /// Starts `tokens` as a prompt: past the history or the rewind point it extends
    /// ([sparkflash_runtime::resume]), else from the start, and runs its first part.
    fn begin(
        &mut self,
        b: &Buffers,
        tokens: &[u32],
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Part, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(DEVICE, "the prompt has no tokens"));
        }
        let start = match resume(&self.history, true, self.point, tokens) {
            Some(Resume::Extend(held)) => held,
            Some(Resume::Rewind(point)) => {
                // The attention caches go back by the position; the recurrent state by its copy.
                if let Some(checkpoint) = &self.checkpoint {
                    let live = self.linear_regions();
                    checkpoint
                        .restore(&self.shared.cuda, None, &live, point)
                        .map_err(|error| {
                            BackendError::new(DEVICE, format!("the rewind point: {error}"))
                        })?;
                }
                point
            }
            None => 0,
        };
        self.history.truncate(start);
        if start == 0 {
            self.point = None;
            self.clear_linear()?;
        }
        self.cached = start;
        self.prompt = Some((tokens.to_vec(), start));
        self.part(b, trace)
    }

    /// Runs the next [CHUNK_ROWS] ids of the prompt [Self::begin] started; after the last, the
    /// prompt's first step.
    fn part(
        &mut self,
        b: &Buffers,
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Part, BackendError> {
        let Some((tokens, next)) = self.prompt.take() else {
            return Err(BackendError::new(
                DEVICE,
                "the sequence has no prompt in parts",
            ));
        };
        let end = (next + CHUNK_ROWS).min(tokens.len());
        let shared = Arc::clone(&self.shared);
        walk(
            &shared,
            b,
            &mut [(&mut *self, &tokens[next..end])],
            shared.policy,
            trace,
        )?;
        if end < tokens.len() {
            self.prompt = Some((tokens, end));
            return Ok(Part::More);
        }
        let logits = head(&shared, b, end - next - 1, 1)?;
        self.sampler.reset();
        self.sampler.record_prompt(&tokens);
        let token = self.sampler.choose(&logits, &self.sampling);
        self.point = shared.rewind.then_some(tokens.len());
        let live = self.linear_regions();
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint
                .save(&shared.cuda, None, &live, tokens.len())
                .map_err(|error| BackendError::new(DEVICE, format!("the rewind point: {error}")))?;
        }
        Ok(Part::Done(StepOutput {
            next_token: Some(token),
            logits,
            draft: Vec::new(),
        }))
    }

    /// The whole of `tokens` as a prompt: [Self::begin], then every part.
    fn prefill(
        &mut self,
        b: &Buffers,
        tokens: &[u32],
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<StepOutput, BackendError> {
        let mut part = self.begin(b, tokens, trace.as_deref_mut())?;
        loop {
            match part {
                Part::Done(step) => return Ok(step),
                Part::More => part = self.part(b, trace.as_deref_mut())?,
            }
        }
    }
}

/// One layer's caches in a sequence's lane.
#[derive(Debug, Clone, Copy)]
enum Cache {
    /// An attention layer's BF16 keys and values, `[capacity][kv_width]` each.
    Kv(Deviceptr, Deviceptr),
    /// A Gated DeltaNet layer's conv history and delta-rule state.
    Linear { conv: Deviceptr, state: Deviceptr },
}

/// One walk over the segments' rows, laid out one after another: each sequence's ids at its own
/// position. The row-wise stages (norms, projections, the MLP, the residual adds, the Gated
/// DeltaNet prep and gated norm) take every row at once and the per-sequence stages (rotary, the
/// cache appends, attention, the conv and the scan) each segment's rows, so a row computes as it
/// would alone wherever the projections take the same kernel: a fused step passes no wide policy,
/// and a row GEMV row is the same sum at any batch.
///
/// The residual of every row is left in `b.x`; each sequence's history takes its ids.
fn walk(
    shared: &Shared,
    b: &Buffers,
    segments: &mut [(&mut Sequence, &[u32])],
    policy: Option<WidePolicy>,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), BackendError> {
    let refuse = |message: String| Err(BackendError::new(DEVICE, message));
    let lengths: Vec<usize> = segments.iter().map(|(_, tokens)| tokens.len()).collect();
    let rows = walk::row_plan(&lengths).rows;
    if rows == 0 || rows > CHUNK_ROWS {
        return refuse(format!("a walk takes 1..={CHUNK_ROWS} rows, not {rows}"));
    }
    let mut ids = Vec::with_capacity(rows * 4);
    for (sequence, tokens) in segments.iter_mut() {
        if tokens.is_empty() {
            return refuse("a segment has no tokens".to_owned());
        }
        if let Some(token) = tokens.iter().find(|token| **token as usize >= shared.vocab) {
            return refuse(format!(
                "token {token} is outside the checkpoint vocabulary {}",
                shared.vocab
            ));
        }
        let reach = sequence.position() + tokens.len();
        if reach > shared.context {
            return refuse(format!(
                "the sequence is past the served context {}",
                shared.context
            ));
        }
        sequence
            .lane
            .back(&shared.cuda, reach)
            .map_err(|error| BackendError::new(DEVICE, format!("the lane memory: {error}")))?;
        ids.extend(tokens.iter().flat_map(|token| token.to_le_bytes()));
    }
    let kernels = shared.kernels;
    let hidden = shared.hidden;
    let device = |error: String| BackendError::new(DEVICE, error);
    shared.cuda.copy_htod(&b.tokens, &ids).map_err(device)?;
    shared.call(
        kernels.embed,
        calls::embed_rows(
            shared.embed.pointer.0,
            b.x.0,
            b.tokens.0,
            (hidden, hidden),
            rows,
        ),
    )?;
    for (index, layer) in shared.layers.iter().enumerate() {
        shared.norm(
            RmsNorm::new(b.x.0, layer.input_norm.0, b.normed.0, hidden, shared.eps),
            rows,
        )?;
        match &layer.mixer {
            Mixer::Gqa { .. } => shared.attend(index, layer, b, segments, rows, policy)?,
            Mixer::Gdn(gdn) => shared.linear(index, layer, gdn, b, segments, rows, policy)?,
        }
        shared.project(&layer.o, b, b.attended.0, b.projected.0, rows, policy)?;
        shared.sum(b.projected, rows)?;
        shared.call(kernels.add, calls::add(b.x.0, b.projected.0, rows * hidden))?;
        shared.norm(
            RmsNorm::new(b.x.0, layer.post_norm.0, b.normed.0, hidden, shared.eps),
            rows,
        )?;
        match &layer.ffn {
            LayerFfn::Dense(mlp) => {
                shared.mlp(mlp, layer.shape.clamp, b, b.projected, rows, policy)?;
            }
            LayerFfn::Routed { .. } => shared.routed(layer, b, rows, policy)?,
        }
        shared.sum(b.projected, rows)?;
        shared.call(kernels.add, calls::add(b.x.0, b.projected.0, rows * hidden))?;
        if let Some(trace) = trace.as_deref_mut() {
            let mut bytes = vec![0u8; rows * hidden * 4];
            shared.cuda.copy_dtoh(&mut bytes, &b.x).map_err(device)?;
            trace[index].extend(floats(&bytes));
        }
    }
    for (sequence, tokens) in segments.iter_mut() {
        sequence.history.extend_from_slice(tokens);
    }
    Ok(())
}

/// The logits of `count` residual rows of the last walk from row `first`, row after row: the
/// final norm and the head on the row GEMV.
fn head(
    shared: &Shared,
    b: &Buffers,
    first: usize,
    count: usize,
) -> Result<Vec<f32>, BackendError> {
    if count == 0 || count > b.logit_rows {
        return Err(BackendError::new(
            DEVICE,
            format!(
                "the scratch holds 1..={} rows of logits, not {count}",
                b.logit_rows
            ),
        ));
    }
    let hidden = shared.hidden;
    shared.norm(
        RmsNorm::new(
            b.x.0 + (first * hidden * 4) as u64,
            shared.norm.0,
            b.normed.0,
            hidden,
            shared.eps,
        ),
        count,
    )?;
    shared.project_bf16(&shared.head, b.normed.0, b.logits.0, count, None)?;
    let mut bytes = vec![0u8; count * shared.vocab * 4];
    shared
        .cuda
        .copy_dtoh(&mut bytes, &b.logits)
        .map_err(|error| BackendError::new(DEVICE, format!("the logits: {error}")))?;
    Ok(floats(&bytes))
}

/// One fused step: a row a member, each its own token, and each member's next.
fn step(
    shared: &Shared,
    b: &Buffers,
    members: &mut [(&mut Sequence, u32)],
    trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<Vec<StepOutput>, BackendError> {
    let tokens: Vec<u32> = members.iter().map(|(_, token)| *token).collect();
    let mut segments: Vec<(&mut Sequence, &[u32])> = members
        .iter_mut()
        .zip(tokens.chunks(1))
        .map(|((sequence, _), token)| (&mut **sequence, token))
        .collect();
    walk(shared, b, &mut segments, None, trace)?;
    let logits = head(shared, b, 0, segments.len())?;
    Ok(segments
        .iter_mut()
        .zip(logits.chunks(shared.vocab))
        .map(|((sequence, token), logits)| {
            sequence.sampler.record(token[0]);
            let chosen = sequence.sampler.choose(logits, &sequence.sampling);
            StepOutput {
                next_token: Some(chosen),
                logits: logits.to_vec(),
                draft: Vec::new(),
            }
        })
        .collect())
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// One sequence with its own scratch, outside the fused engine: the parity gate's session, which
/// records each layer's residual.
pub struct DenseDeviceSession {
    sequence: Sequence,
    scratch: Scratch,
    /// Each layer's residual after it, every row in position order, while tracing.
    trace: Option<Vec<Vec<f32>>>,
}

impl DenseDeviceSession {
    fn open(shared: Arc<Shared>) -> Result<Self, String> {
        Ok(Self {
            scratch: Scratch::new(Arc::clone(&shared), 1)?,
            sequence: Sequence::open(shared)?,
            trace: None,
        })
    }

    /// Starts (or stops) recording each layer's residual after it, every row.
    pub fn trace(&mut self, on: bool) {
        self.trace = on.then(|| vec![Vec::new(); self.sequence.shared.layers.len()]);
    }

    /// The recorded residuals, a `[positions][hidden]` row-major run a layer, and a fresh record.
    #[must_use]
    pub fn take_trace(&mut self) -> Vec<Vec<f32>> {
        let layers = self.sequence.shared.layers.len();
        self.trace
            .as_mut()
            .map(|trace| std::mem::replace(trace, vec![Vec::new(); layers]))
            .unwrap_or_default()
    }

    /// Runs `tokens` from the session's position, [CHUNK_ROWS] rows a walk, and returns the last
    /// row's logits. The sampler does not see them.
    ///
    /// # Errors
    /// A token outside the vocabulary, a sequence past the served context, or a device failure.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, BackendError> {
        let shared = Arc::clone(&self.sequence.shared);
        let b = &self.scratch.buffers;
        let mut last = 0;
        for chunk in tokens.chunks(CHUNK_ROWS) {
            walk(
                &shared,
                b,
                &mut [(&mut self.sequence, chunk)],
                shared.policy,
                self.trace.as_mut(),
            )?;
            last = chunk.len();
        }
        if last == 0 {
            return Err(BackendError::new(DEVICE, "the step has no tokens"));
        }
        head(&shared, b, last - 1, 1)
    }
}

impl ModelSession for DenseDeviceSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // Penalties, temperature, top-k and top-p run on the host over the row every step copies
        // back, so a sampled request is served rather than approximated by the greedy pick.
        self.sequence.sampling = sampling;
        self.sequence.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        // A session outside the engine starts every prompt over.
        self.sequence.history.clear();
        self.sequence.point = None;
        self.sequence
            .prefill(&self.scratch.buffers, tokens, self.trace.as_mut())
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let shared = Arc::clone(&self.sequence.shared);
        let mut steps = step(
            &shared,
            &self.scratch.buffers,
            &mut [(&mut self.sequence, token)],
            self.trace.as_mut(),
        )?;
        Ok(steps.remove(0))
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        // Kept conversations are the fused engine's ([DenseLanes]).
        None
    }
}

/// The arm behind the shared fused engine ([sparkflash_runtime::FusedEngine]): a slot is one
/// [Sequence], a step is one walk over every member's row, and a prompt runs a [CHUNK_ROWS]-row
/// part at a time between the members' steps. The engine thread runs one job at a time, so the
/// slots share one scratch.
pub struct DenseLanes {
    shared: Arc<Shared>,
    scratch: Mutex<Scratch>,
    /// The most members one walk holds.
    rows: usize,
}

impl DenseLanes {
    fn new(shared: Arc<Shared>, capacity: usize) -> Result<Self, BackendError> {
        let rows = capacity.clamp(1, CHUNK_ROWS);
        let scratch = Scratch::new(Arc::clone(&shared), rows)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(Self {
            shared,
            scratch: Mutex::new(scratch),
            rows,
        })
    }

    fn scratch(&self) -> Result<std::sync::MutexGuard<'_, Scratch>, BackendError> {
        self.scratch
            .lock()
            .map_err(|_| BackendError::new(DEVICE, "the walk's scratch is poisoned"))
    }
}

impl FusedArm for DenseLanes {
    type Slot = Sequence;

    fn open_slot(&self) -> Result<Sequence, BackendError> {
        Sequence::open(Arc::clone(&self.shared)).map_err(|error| BackendError::new(DEVICE, error))
    }

    fn holds_request(&self, slot: &Sequence) -> bool {
        !slot.history.is_empty() || slot.prompt.is_some()
    }

    fn renew(&self, slot: &mut Sequence) -> Result<(), BackendError> {
        let mut fresh = self.open_slot()?;
        // The request's sampling reached the old sequence before its prompt replaced it.
        fresh.sampling = slot.sampling;
        *slot = fresh;
        Ok(())
    }

    fn set_sampling(&self, slot: &mut Sequence, sampling: Sampling) {
        slot.sampling = sampling;
        slot.sampler.reset();
    }

    fn prefill(&self, slot: &mut Sequence, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        let scratch = self.scratch()?;
        slot.prefill(&scratch.buffers, tokens, None)
    }

    /// A prompt's [CHUNK_ROWS]-row chunks run one a part; the arm serves text, so a prompt with
    /// images is refused as [FusedArm::prefill_images] refuses it.
    fn begin_prompt(
        &self,
        slot: &mut Sequence,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<Part, BackendError> {
        if !images.is_empty() {
            return self.prefill_images(slot, tokens, images).map(Part::Done);
        }
        let scratch = self.scratch()?;
        slot.begin(&scratch.buffers, tokens, None)
    }

    fn next_part(&self, slot: &mut Sequence) -> Result<Part, BackendError> {
        let scratch = self.scratch()?;
        slot.part(&scratch.buffers, None)
    }

    fn step(&self, members: &mut [(Sequence, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        let scratch = self.scratch()?;
        let mut members: Vec<(&mut Sequence, u32)> = members
            .iter_mut()
            .map(|(sequence, token)| (sequence, *token))
            .collect();
        step(&self.shared, &scratch.buffers, &mut members, None)
    }

    fn speculation(&self, _slot: &Sequence) -> Speculation {
        Speculation::default()
    }

    /// A slot keeps its sequence between requests: the next prompt that extends it forwards only
    /// the rest ([Sequence::begin]).
    fn history<'s>(&self, slot: &'s Sequence) -> Option<&'s [u32]> {
        Some(&slot.history)
    }

    /// The end of the last prompt, which a prompt that parts from the reply goes back to.
    fn rewind_prefix<'s>(&self, slot: &'s Sequence) -> Option<&'s [u32]> {
        slot.point
            .filter(|point| *point <= slot.history.len())
            .map(|point| &slot.history[..point])
    }

    fn cached_tokens(&self, slot: &Sequence) -> usize {
        slot.cached
    }

    /// The sequences' caches share one pool: a prompt that does not continue the slot's history
    /// starts over, so it counts the chunks the reset gives back.
    fn room(&self, slot: &Sequence, prompt: Option<&[u32]>) -> bool {
        match prompt {
            Some(tokens) => slot.lane.room(tokens.len(), !self.continues(slot, tokens)),
            None => slot.lane.room(slot.position() + 1, false),
        }
    }

    fn max_batch(&self) -> usize {
        self.rows
    }

    fn ledger(&self) -> Ledger {
        let mut ledger = Ledger::from_declaration(
            sparkflash_model::ledger_for_arm("dense").expect("the dense arm is declared"),
        );
        ledger.set(
            Strategy::PromptParts,
            Mode::native(format!("{CHUNK_ROWS}-row chunks")),
        );
        ledger.drafting(None, false, false);
        let reuse = if self.shared.rewind {
            "a prompt that extends a lane's sequence, or goes back to the last prompt's end"
        } else {
            "a prompt that extends a lane's sequence; no rewind (SPARKFLASH_REWIND=off)"
        };
        ledger.set(Strategy::PrefixReuse, Mode::native(reuse));
        ledger.set(
            Strategy::Rewind,
            if self.shared.rewind {
                Mode::native("a prompt that goes back to the last prompt's end")
            } else {
                Mode::off(format!(
                    "{}=off keeps no rewind point",
                    sparkflash_runtime::REWIND_ENV
                ))
            },
        );
        ledger
    }
}

#[cfg(test)]
mod tests {
    use super::{CHUNK_ROWS, profile_policy};
    use sparkflash_model::profile::ModelProfile;
    use sparkflash_ops::projection::WidePolicy;

    #[test]
    fn the_profile_fields_the_arm_does_not_honour_are_refused() {
        assert_eq!(
            profile_policy(&ModelProfile::default()),
            Ok(Some(WidePolicy::DENSE))
        );
        let mut row_gemv = ModelProfile::default();
        row_gemv.prefill.tensor_cores = Some(false);
        assert_eq!(profile_policy(&row_gemv), Ok(None));
        let refused = |edit: fn(&mut ModelProfile)| {
            let mut profile = ModelProfile::default();
            edit(&mut profile);
            profile_policy(&profile).expect_err("the field is refused")
        };
        assert!(refused(|profile| profile.prefill.chunk = Some(CHUNK_ROWS * 2)).contains("chunk"));
        assert!(refused(|profile| profile.prefill.enabled = Some(false)).contains("prefill"));
        assert!(refused(|profile| profile.prefill.exact = Some(true)).contains("exact"));
        assert!(refused(|profile| profile.decode.tensor_cores = Some(true)).contains("decode"));
        assert!(refused(|profile| profile.speculation.enabled = Some(true)).contains("drafter"));
        assert!(
            refused(|profile| profile.sequence.parallel = Some(true)).contains("sequence parallel")
        );
        let mut lanes = ModelProfile::default();
        lanes.runtime.concurrency = Some(4);
        assert!(
            profile_policy(&lanes).is_ok(),
            "the engine's lanes are honoured"
        );
        let mut chunk = ModelProfile::default();
        chunk.prefill.chunk = Some(CHUNK_ROWS);
        assert!(
            profile_policy(&chunk).is_ok(),
            "the arm's own chunk is honoured"
        );
    }
}
