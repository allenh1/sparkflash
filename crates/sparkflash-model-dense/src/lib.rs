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
//! their typed [sparkflash_ops::calls]. Its projections take the shared wide tile under
//! [WidePolicy::DENSE]; decode rows of eight or fewer keep the row GEMV, and a step that batches
//! more takes the tile.
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
//! It serves on the shared runtime's contract ([DenseLanes], a [sparkflash_runtime::Runtime]
//! over [DenseWalk]): the runtime owns the lanes, the prompt parts, kept conversations,
//! the rewind point and the ledger, and the arm supplies only the walk and its head. Its caches
//! stay on the shared lane memory ([SequenceLanes]).
//!
//! It runs on one rank or on a tensor-parallel group ([DenseDeviceBackend::open_rank]): a rank
//! holds its share of every attention layer's query heads (and of the key/value heads, which a
//! world wider than them shares), of every Gated DeltaNet layer's key and value heads, and of
//! every MLP's and expert's width (the gate and up rows, the down columns), as the shared
//! [TpPlan] splits them, and holds the rest whole: the embedding, the norms, the routers and the
//! head. Each branch therefore leaves a partial of its hidden rows, which the ranks sum in rank
//! order ([RowSums]) before the residual takes it, so the residual is the same bits on every rank
//! and so is every replicated decision (the router's picks, the token). Rank 0 serves the group's
//! lanes and every other rank replays them ([DenseDeviceBackend::enable_group_fusion],
//! [DenseDeviceBackend::mirror_group]).

use std::sync::Arc;

use sparkflash_kernels::checkpoint::Checkpoint;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::{
    Image, dory_cubin, moe_plan_cubin, qwen4_attn_cubin, qwen4_exp_cubin, qwen4_moe_cubin,
    qwen4_prefill_cubin, qwen4_scan_cubin,
};
use sparkflash_kernels::lane_memory::{Layout, SequenceLane, SequenceLanes};
use sparkflash_model::architecture::{Activation, AttentionKind, Ffn, WeightFormat};
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_model::compose::{LayerStep, StepRole};
use sparkflash_model::ledger::State;
use sparkflash_model::{Architecture, Capabilities, ModelDescriptor, TpAxis, TpPlan};
use sparkflash_ops::OpPlan;
use sparkflash_ops::calls::{self, Call, FullAttention, RmsNorm, Rope};
use sparkflash_ops::dflash::{DraftUploader, Part as Run};
use sparkflash_ops::engine_plan::{EnginePlan, PlanError};
use sparkflash_ops::formats::{
    Bf16, F32 as F32Format, FormatKind, Fp8E4m3, Mat, NvFp4, PackedBf16, PackedOrder, Scales,
    Tensor, upload as upload_format,
};
use sparkflash_ops::gdn::{self, GdnGeometry, gdn_qk_row};
use sparkflash_ops::gemv::{
    CudaHandle, E4M3_TILE, GEMV_MAX_BATCH, GemvDevice, GpuMat, GpuMatE4m3Tile, TILED_GEMV_BLOCKS,
    TILED_GEMV_PARTIALS, gemv_e4m3_tile,
};
use sparkflash_ops::moe::{Launch, MoePlanCall, combine_grid, router_score, token};
use sparkflash_ops::naming::{AttentionNames, ExpertNames, FfnNames, GdnNames, TensorNames};
use sparkflash_ops::packed::{self, Form, PackedSet, Placement};
use sparkflash_ops::plan::Precision;
use sparkflash_ops::projection::{
    Fp8Tile, Route, WIDE_SHARED, WidePolicy, WideTile, route_bf16, route_fp8,
};
use sparkflash_ops::scan::{ScanKind, ScanPlan};
use sparkflash_ops::tp::RowSums;
use sparkflash_ops::widen::{FormatLedger, LedgerRow, WidenLog, widen};
use sparkflash_runtime::{
    ChunkPolicy, GroupWire, LaneState, Limits, PartBody, Place, PlanBranch, PlanSeq, RankSplit,
    Region, Serving, TextWire, Walk, WalkArm, walk_rows,
};
use sparkflash_weights::tp::TpRuntime;
use sparkflash_weights::{DType, TensorSpec, WeightSource};

pub mod onboard;
pub mod parts;
mod walk;

pub use parts::{Dense, DensePart, DenseParts, OpenCx, PassCx, ScratchSlot, SegmentRows};
pub use walk::DenseWalk;

/// The dense arm's sequence, as the runtime drives it. The runtime owns the prompt parts, the
/// history, reuse, rewind, the lane pool and the ledger; this arm supplies the walk.
pub type DenseLanes = sparkflash_runtime::Runtime<DenseWalk>;

/// The host-facing name of a device refusal.
const DEVICE: &str = "dense_device";

/// The prompt rows one pass of the layer walk takes by default: two of the wide tile's 128-row
/// blocks. This is the arm's legacy input to [sparkflash_runtime::ChunkPolicy], so a profile that
/// names no `prefill.chunk` keeps the 256-row chunks its kernels were chosen for. A profile may
/// name a documented smaller size; [chunk_rows] quantises it and the walk's limits carry it.
pub const CHUNK_ROWS: usize = 256;

/// The dense arm's one-row token step: the row GEMV, not a tensor-core body. A step that batches
/// more than [sparkflash_ops::gemv::GEMV_MAX_BATCH] rows takes the shared wide tile, but the
/// single-row decode the strategy row is about does not. The walk's capability row and the profile
/// check that refuses `decode.tensor_cores = true` read this one value, so the ledger and the
/// refusal cannot disagree.
pub(crate) const TOKEN_DECODE: State = State::Off("row GEMV; decode.tensor_cores=true is refused");

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
            // Every kind is named, no wildcard: a new kind leads the compiler here.
            AttentionKind::Mla { .. } | AttentionKind::Inkling { .. } | AttentionKind::Kda(_) => {
                return Err(format!(
                    "layer {index} is neither grouped-query attention nor a Gated DeltaNet"
                ));
            }
        };
        let fp8_experts = matches!(
            architecture.quantization.expert_format,
            WeightFormat::Fp8Tile { .. }
        );
        let nvfp4_experts = matches!(
            architecture.quantization.expert_format,
            WeightFormat::NvFp4 { .. }
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
                let run = if nvfp4_experts {
                    16
                } else if fp8_experts {
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

/// One layer on the device: its shape and its branches in forward order, the token mixer's and
/// then the feed-forward's. Each branch's part is built from the layer's step
/// ([Architecture::steps]) by the constructor the registry holds for the part's kind
/// ([DenseParts]).
struct Layer {
    shape: Shape,
    branches: Vec<Branch>,
}

/// One branch of a layer: the RMSNorm of the residual, the part over the normed rows, the group's
/// sum of its output where the part leaves a rank's partial, and the residual add
/// ([sparkflash_runtime::PlanBranch]; its slot is [LaneParts::regions]'s index).
type Branch = PlanBranch<Dense, Deviceptr>;

/// Every branch of `layers`, in walk order: the order of [Branch::slot].
fn branches(layers: &[Layer]) -> impl Iterator<Item = &Branch> {
    layers.iter().flat_map(|layer| &layer.branches)
}

/// Grouped-query attention on the device ([sparkflash_model::Part::GqaAttention]): its projections, its q/k norms
/// and its output projection.
#[derive(Debug, Clone)]
struct Gqa {
    shape: Shape,
    q: Proj,
    k: Proj,
    v: Proj,
    q_norm: Option<Deviceptr>,
    k_norm: Option<Deviceptr>,
    /// The attention's output projection.
    o: Proj,
    /// The keys' cache and the values', a `[capacity][kv_width]` BF16 run each.
    state: [Region; 2],
}

/// A Gated DeltaNet layer on the device ([sparkflash_model::Part::GdnScan]): its weights and the linear
/// attention's `out_proj`.
#[derive(Debug, Clone)]
struct Gdn {
    shape: Shape,
    weights: GdnLayer,
    o: Proj,
    /// The conv history and the delta-rule state.
    state: [Region; 2],
}

/// A Gated DeltaNet layer's weights: the fused q/k/v projection, the output gate's, the beta and
/// decay logits', the conv (BF16 `[conv_width][taps]`), the decay vectors and the gated norm (BF16
/// a value head's width).
#[derive(Debug, Clone)]
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

/// One projection's weights on the device: BF16, or E4M3 codes under one BF16 inverse scale a
/// 128 x 128 tile, as the checkpoint stores the tensor. The matrix is the typed upload
/// ([sparkflash_ops::formats::upload]), so its stored dtype and its format's scale count were
/// checked when it was read.
#[derive(Debug, Clone)]
enum Proj {
    Bf16(Mat<Bf16>),
    Fp8(Mat<Fp8E4m3>),
    /// E2M1 codes under one E4M3 scale a 16-value group: the routed experts' form.
    NvFp4(Mat<NvFp4>),
}

impl Proj {
    /// Output rows.
    fn rows(&self) -> usize {
        match self {
            Self::Bf16(mat) => mat.rows(),
            Self::Fp8(mat) => mat.rows(),
            Self::NvFp4(mat) => mat.rows(),
        }
    }

    /// The token passes' table entries: the weight and its scale grid (none for BF16).
    fn table(&self) -> [u64; 2] {
        match self {
            Self::Bf16(mat) => [mat.data().0, 0],
            Self::Fp8(mat) => [
                mat.data().0,
                mat.scales().blocks().map_or(0, |blocks| blocks.0),
            ],
            Self::NvFp4(mat) => [
                mat.data().0,
                mat.scales().blocks().map_or(0, |blocks| blocks.0),
            ],
        }
    }

    /// The per-tensor f32 multiplier an NVFP4 expert's output takes.
    fn global(&self) -> Option<f32> {
        match self {
            Self::NvFp4(mat) => match mat.scales() {
                Scales::NvFp4 { global, .. } => Some(*global),
                _ => None,
            },
            _ => None,
        }
    }

    /// The storage form this projection was uploaded as.
    fn form(&self) -> ExpertForm {
        match self {
            Self::Bf16(_) => ExpertForm::Bf16,
            Self::Fp8(_) => ExpertForm::Fp8,
            Self::NvFp4(_) => ExpertForm::NvFp4,
        }
    }
}

/// One SwiGLU MLP's projections: a dense layer's, or a routed layer's shared expert.
#[derive(Debug, Clone)]
struct Mlp {
    gate: Proj,
    up: Proj,
    down: Proj,
}

/// A layer's dense SwiGLU MLP on the device ([sparkflash_model::Part::DenseMlp]).
#[derive(Debug, Clone)]
struct DenseMlp {
    mlp: Mlp,
    /// The layer's `swiglu_limit`, when the release sets one.
    clamp: Option<f32>,
}

/// A layer's routed experts on the device ([sparkflash_model::Part::Moe]): the router, the experts' table, the
/// shared expert and its gate.
#[derive(Debug, Clone)]
struct RoutedExperts {
    shape: Shape,
    /// The router, `[experts][hidden]`.
    router: Proj,
    /// The selection's correction bias, f32 an expert, when selection adds one.
    bias: Option<Deviceptr>,
    /// The NVFP4 experts' three per-tensor scales an expert (gate, up, down), when they
    /// are stored that way.
    scales: Option<Deviceptr>,
    /// Six addresses an expert, the token passes' table: gate, its scale grid (unread for
    /// BF16), up, its grid, down, its grid.
    table: Deviceptr,
    shared: Option<Mlp>,
    /// The shared expert's sigmoid gate, one BF16 `[hidden]` row.
    shared_gate: Option<Proj>,
}

/// One loaded kernel and the catalogue entry it was resolved from: the image's symbol list and
/// the catalogue's precision mask are checked at load, so a launch site holds a handle rather
/// than a name it looked up.
#[derive(Clone, Copy)]
struct Bound {
    /// The loaded driver function.
    kernel: Kernel,
    /// The arithmetic the catalogue entry accepts this body at.
    precision: Precision,
}

impl Bound {
    /// Whether `image` exports `symbol` and the catalogue carries it at `precision`.
    fn carried(image: Image, symbol: &'static str, precision: Precision) -> Result<(), String> {
        if !image.exported_symbols().contains(&symbol) {
            return Err(format!("{} does not export {symbol}", image.name()));
        }
        let carried = sparkflash_ops::parts::CATALOGUE.iter().any(|entry| {
            entry.image == image
                && entry
                    .symbols
                    .iter()
                    .any(|candidate| candidate.name == symbol && candidate.accepts(precision))
        });
        if !carried {
            return Err(format!(
                "the catalogue carries no {symbol} in {} accepting {}",
                image.name(),
                precision.name()
            ));
        }
        Ok(())
    }

    /// Resolve `symbol` in `image` at `precision` and load it.
    ///
    /// # Errors
    /// A symbol the image does not export, one the catalogue does not carry at this precision,
    /// or the driver's load refusal.
    fn new(
        cuda: &Cuda,
        image: Image,
        cubin: &'static [u8],
        symbol: &'static str,
        precision: Precision,
    ) -> Result<Self, String> {
        Self::carried(image, symbol, precision)?;
        let (_, kernel) = cuda.load_named(cubin, symbol)?;
        Ok(Self { kernel, precision })
    }

    /// [Self::new] for a body loaded with a dynamic shared-memory reservation.
    ///
    /// # Errors
    /// As [Self::new], or the driver's attribute call.
    fn shared(
        cuda: &Cuda,
        image: Image,
        cubin: &'static [u8],
        symbol: &'static str,
        precision: Precision,
        shared_bytes: u32,
    ) -> Result<Self, String> {
        Self::carried(image, symbol, precision)?;
        let (_, kernel) = cuda.load_named_shared(cubin, symbol, shared_bytes)?;
        Ok(Self { kernel, precision })
    }
}

impl From<Bound> for Kernel {
    fn from(bound: Bound) -> Self {
        bound.kernel
    }
}

/// The routed layers' kernels, loaded when a layer routes.
#[derive(Clone, Copy)]
struct MoeKernels {
    route: Bound,
    plan: Bound,
    gate_up: Bound,
    /// The gate/up pass under a `swiglu_limit`; the NVFP4 body has none.
    gate_up_clamped: Option<Bound>,
    down: Bound,
    combine: Bound,
    /// Whether the expert bodies read the NVFP4 scale table (their arguments carry it).
    nvfp4: bool,
}

/// The packed BF16 form's four bodies ([sparkflash_ops::packed]), resolved through the catalogue
/// like the projections.
#[derive(Clone, Copy)]
struct PackedKernels {
    pack: Bound,
    unpack: Bound,
    gemv: Bound,
    gemv4: Bound,
}

impl PackedKernels {
    /// The plain handles [packed::PackedSet] takes.
    fn raw(&self) -> packed::Kernels {
        packed::Kernels {
            pack: self.pack.kernel,
            unpack: self.unpack.kernel,
            gemv: self.gemv.kernel,
            gemv4: self.gemv4.kernel,
        }
    }
}

/// The kernels this arm launches, loaded once and bound to the catalogue entry each runs.
#[derive(Clone, Copy)]
struct Kernels {
    add: Bound,
    /// A group's row sum ([RowSums]) on the device legs.
    sum_slots: Bound,
    deinterleave: Bound,
    embed: Bound,
    gemv: Bound,
    packed: PackedKernels,
    rmsnorm: Bound,
    rope: Bound,
    swiglu: Bound,
    swiglu_clamped: Bound,
    attention: Bound,
    combine: Bound,
    cache: Bound,
    wide: Bound,
    /// The vocabulary-wide argmax the greedy head chooses with (`q4e_argmax_slices`).
    argmax: Bound,
    moe: Option<MoeKernels>,
    fp8: Option<Fp8Kernels>,
    gdn: Option<GdnKernels>,
}

/// The Gated DeltaNet layers' kernels, loaded when a layer is one: the scan image's chunk path
/// ([sparkflash_ops::gdn]) with the scan the [ScanPlan] resolves, and the zero a reset state takes.
#[derive(Clone, Copy)]
struct GdnKernels {
    conv: Bound,
    history: Bound,
    prep: Bound,
    scan: Bound,
    /// The plan's scan grid and block.
    scan_launch: ((u32, u32, u32), (u32, u32, u32)),
    gate: Bound,
    zero: Bound,
}

/// Threads in the block `q4e_argmax_slices` reduces a vocabulary with: the 32 warps its block
/// reduction holds.
const ARGMAX_BLOCK: u32 = 1024;

/// The block-scaled FP8 projections' kernels, loaded when the checkpoint declares FP8: the
/// tiled E4M3 GEMV for decode rows and the FP8 tensor-core tile for prompt chunks.
#[derive(Clone, Copy)]
struct Fp8Kernels {
    gemv: Bound,
    tile: Bound,
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
        let token = |symbol: &'static str, precision: Precision| {
            Bound::new(cuda, Image::Qwen4Exp, qwen4_exp_cubin(), symbol, precision)
        };
        let attention = |symbol: &'static str, precision: Precision| {
            Bound::new(
                cuda,
                Image::Qwen4Attn,
                qwen4_attn_cubin(),
                symbol,
                precision,
            )
        };
        let (gate_up, gate_up_clamped, down, expert_precision) = match formats.expert {
            ExpertForm::Fp8 => (
                "q4e_expert_gate_up_fp8",
                Some("q4e_expert_gate_up_fp8_clamped"),
                "q4e_expert_down_fp8",
                Precision::Fp8E4M3,
            ),
            ExpertForm::NvFp4 => (
                "q4e_expert_gate_up",
                None,
                "q4e_expert_down",
                Precision::NvFp4,
            ),
            ExpertForm::Bf16 => (
                "q4e_expert_gate_up_bf16",
                Some("q4e_expert_gate_up_bf16_clamped"),
                "q4e_expert_down_bf16",
                Precision::Bf16,
            ),
        };
        let staged = |symbol: &'static str, precision: Precision, bytes: u32| {
            Bound::shared(
                cuda,
                Image::Qwen4Moe,
                qwen4_moe_cubin(),
                symbol,
                precision,
                bytes,
            )
        };
        let moe = match stage {
            Some(bytes) => Some(MoeKernels {
                route: Bound::new(
                    cuda,
                    Image::Qwen4Moe,
                    qwen4_moe_cubin(),
                    "q4e_route",
                    Precision::Bf16,
                )?,
                plan: Bound::new(
                    cuda,
                    Image::MoePlan,
                    moe_plan_cubin(),
                    "moe_plan",
                    Precision::Bf16,
                )?,
                gate_up: staged(gate_up, expert_precision, bytes)?,
                gate_up_clamped: gate_up_clamped
                    .map(|symbol| staged(symbol, expert_precision, bytes))
                    .transpose()?,
                down: staged(down, expert_precision, bytes)?,
                nvfp4: formats.expert == ExpertForm::NvFp4,
                combine: Bound::new(
                    cuda,
                    Image::Qwen4Moe,
                    qwen4_moe_cubin(),
                    "q4e_moe_combine",
                    Precision::Bf16,
                )?,
            }),
            None => None,
        };
        let fp8 = if formats.fp8 {
            Some(Fp8Kernels {
                gemv: token("q4e_gemv_tiled_e4m3_tile", Precision::Fp8E4M3)?,
                tile: Bound::new(
                    cuda,
                    Image::Qwen4Prefill,
                    qwen4_prefill_cubin(),
                    "q4p_dense_fp8",
                    Precision::Fp8E4M3,
                )?,
            })
        } else {
            None
        };
        let gdn = match scan {
            Some(plan) => {
                let launch = plan.launch();
                let scan = |symbol: &'static str| {
                    Bound::new(
                        cuda,
                        Image::Qwen4Scan,
                        qwen4_scan_cubin(),
                        symbol,
                        Precision::Bf16,
                    )
                };
                Some(GdnKernels {
                    conv: scan("q4p_gdn_conv")?,
                    history: scan("q4p_gdn_conv_state")?,
                    prep: scan("q4p_gdn_prep")?,
                    scan: scan(launch.kernel.symbol)?,
                    scan_launch: (launch.grid, launch.block),
                    gate: scan("q4p_gdn_gate")?,
                    zero: token("q4e_zero", Precision::Bf16)?,
                })
            }
            None => None,
        };
        debug_assert!(
            fp8.as_ref()
                .is_none_or(|kernels| kernels.gemv.precision == Precision::Fp8E4M3),
            "the FP8 GEMV resolves at FP8 E4M3"
        );
        debug_assert!(
            moe.as_ref().is_none_or(|kernels| {
                kernels.gate_up.precision == expert_precision
                    && kernels.down.precision == expert_precision
            }),
            "the expert bodies resolve at the checkpoint's expert format"
        );
        let packed = PackedKernels {
            pack: token("q4e_pack_bf16", Precision::Bf16)?,
            unpack: token("q4e_unpack_bf16", Precision::Bf16)?,
            gemv: token("q4e_gemv_packed", Precision::Bf16)?,
            gemv4: token("q4e_gemv4_packed", Precision::Bf16)?,
        };
        Ok(Self {
            moe,
            fp8,
            gdn,
            add: token("q4e_add", Precision::Bf16)?,
            sum_slots: token("q4e_sum_slots", Precision::Bf16)?,
            deinterleave: token("q4e_deinterleave", Precision::Bf16)?,
            embed: token("q4e_embed", Precision::Bf16)?,
            gemv: token("q4e_gemv_bf16", Precision::Bf16)?,
            packed,
            rmsnorm: token("q4e_rmsnorm", Precision::Bf16)?,
            rope: token("q4e_rope", Precision::Bf16)?,
            swiglu: token("q4e_swiglu", Precision::Bf16)?,
            swiglu_clamped: token("q4e_swiglu_clamped", Precision::Bf16)?,
            attention: attention("q4e_attn_full", Precision::Bf16)?,
            combine: attention("q4e_attn_full_combine", Precision::Bf16)?,
            cache: attention("q4e_cache_bf16", Precision::Bf16)?,
            argmax: token("q4e_argmax_slices", Precision::Bf16)?,
            wide: Bound::shared(
                cuda,
                Image::Dory,
                dory_cubin(),
                "dory_mma_wide",
                Precision::Bf16,
                WIDE_SHARED,
            )?,
        })
    }
}

/// The weight formats a checkpoint declares, which pick the kernels the arm loads.
#[derive(Debug, Clone, Copy)]
struct Formats {
    /// Any dense projection is block-scaled FP8.
    fp8: bool,
    /// The stored format of the routed experts, which the arm reads at that format.
    expert: ExpertForm,
}

/// The three storage forms the routed experts are read at, and the token bodies that read each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertForm {
    Bf16,
    Fp8,
    /// E2M1 codes with one E4M3 scale a 16-value group and a per-tensor f32 scale.
    NvFp4,
}

impl Formats {
    fn of(architecture: &Architecture) -> Self {
        let tiled = |format: WeightFormat| matches!(format, WeightFormat::Fp8Tile { .. });
        let quantization = &architecture.quantization;
        let expert = match quantization.expert_format {
            WeightFormat::Fp8Tile { .. } => ExpertForm::Fp8,
            WeightFormat::NvFp4 { .. } => ExpertForm::NvFp4,
            _ => ExpertForm::Bf16,
        };
        Self {
            fp8: tiled(quantization.dense_format) || expert == ExpertForm::Fp8,
            expert,
        }
    }
}

/// The weights, kernels and geometry every session reads: the host the shared walk
/// ([sparkflash_runtime::PlanWalk]) drives.
pub struct Shared {
    cuda: CudaHandle,
    kernels: Kernels,
    embed: Mat<Bf16>,
    head: Mat<Bf16>,
    norm: Deviceptr,
    layers: Vec<Layer>,
    hidden: usize,
    vocab: usize,
    eps: f32,
    zero_centred: bool,
    policy: Option<WidePolicy>,
    /// The prompt rows one pass takes, from [chunk_rows].
    chunk: usize,
    context: usize,
    /// The sequences' key/value caches' lanes, for sequences of `context` positions.
    lanes: Arc<SequenceLanes<LaneParts>>,
    /// The recurrent regions of a sequence in bytes, in walk order: each Gated DeltaNet layer's
    /// conv history and delta-rule state, and whatever a plugged-in part declares
    /// ([Region::Recurrent]). Its rewind point copies these.
    linear_sizes: Vec<usize>,
    /// The scratch the parts' constructors reserved ([OpenCx::scratch]), in f32 a row each: every
    /// walk's scratch holds one buffer of [CHUNK_ROWS] rows for each.
    scratch: Vec<usize>,
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
    /// The engine its sessions open on once [Self::enable_fusion] (or, on rank 0 of a group,
    /// [Self::enable_group_fusion]) has run.
    fused: Option<Serving>,
}

/// The arm's group wire at `context`: it serves text.
pub(crate) fn wire(context: usize) -> TextWire {
    TextWire::new("dense", "shared dense device", context)
}

/// What the runtime serves this backend by ([sparkflash_runtime::serve] and its group forms).
impl WalkArm for DenseDeviceBackend {
    fn walk(&self, capacity: usize) -> Result<impl Walk<Seq: Send> + Send + Sync, BackendError> {
        walk::walk(Arc::clone(&self.shared), capacity)
    }

    /// The arm serves text.
    fn wire(&self) -> impl GroupWire {
        wire(self.shared.context)
    }

    /// A rank of an [Self::open_rank] group holds the sums its peers step through, so it steps
    /// only with them; a backend opened alone has no peers to step with.
    fn admit(&self, place: Place) -> Result<(), BackendError> {
        match (self.shared.sums.is_some(), place) {
            (true, Place::Alone) => Err(BackendError::new(
                DEVICE,
                "a group's rank steps its lanes with its peers: fuse rank 0's with \
                 enable_group_fusion and mirror them on the others with mirror_group",
            )),
            (false, Place::Group) => Err(BackendError::new(
                DEVICE,
                "group lanes need a group's rank (open_rank); this backend runs alone, so fuse \
                 its sessions with enable_fusion",
            )),
            _ => Ok(()),
        }
    }
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
        Self::open_with(
            descriptor,
            architecture,
            source,
            context,
            &Dense::catalogue(),
        )
    }

    /// [Self::open] with the executors `parts` holds in place of the catalogue's
    /// ([Dense::catalogue]): a family with a part of its own registers its constructor there,
    /// and the walk runs it wherever a layer's step names its kind.
    ///
    /// # Errors
    /// As [Self::open], or a step whose part has no executor in `parts`, by the part's name.
    pub fn open_with(
        descriptor: ModelDescriptor,
        architecture: &Architecture,
        source: &dyn WeightSource,
        context: usize,
        parts: &DenseParts,
    ) -> Result<Self, String> {
        Self::open_at(descriptor, architecture, source, context, None, parts)
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
        Self::open_rank_with(
            descriptor,
            architecture,
            source,
            context,
            tp,
            &Dense::catalogue(),
        )
    }

    /// [Self::open_rank] with the executors `parts` holds, as [Self::open_with]. Every rank of
    /// the group opens with the same registry.
    ///
    /// # Errors
    /// As [Self::open_rank] and [Self::open_with].
    pub fn open_rank_with(
        descriptor: ModelDescriptor,
        architecture: &Architecture,
        source: &dyn WeightSource,
        context: usize,
        tp: TpRuntime,
        parts: &DenseParts,
    ) -> Result<Self, String> {
        Self::open_at(descriptor, architecture, source, context, Some(tp), parts)
    }

    fn open_at(
        mut descriptor: ModelDescriptor,
        architecture: &Architecture,
        source: &dyn WeightSource,
        context: usize,
        tp: Option<TpRuntime>,
        parts: &DenseParts,
    ) -> Result<Self, String> {
        if let Some(refusal) = sparkflash_ops::parts::dense_device_refusal(architecture) {
            return Err(format!("the shared dense device arm refuses it: {refusal}"));
        }
        // Every part a step names has an executor before anything is loaded: a part with none is
        // refused by name, not skipped.
        let steps = architecture.steps();
        for step in &steps {
            let kinds = sparkflash_runtime::plan_roles(step)?;
            parts
                .require(kinds.into_iter().map(|(_, part)| part))
                .map_err(|error| {
                    format!(
                        "the shared dense device arm cannot run layer {}: {error}",
                        step.index
                    )
                })?;
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
        let chunk = chunk_rows(profile, context)?;
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
            (&names, &steps),
            source,
            formats,
            (&shapes, shard),
            parts,
            &mut owned,
        );
        let Uploaded {
            embed,
            head,
            norm,
            layers,
            bytes: weight_bytes,
            rows: mut ledger_rows,
            widen,
            labels,
            packable,
            scratch,
        } = match uploaded {
            Ok(uploaded) => uploaded,
            Err(error) => {
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error);
            }
        };
        // The token GEMVs read a packed copy of every BF16 projection; the wide tiles keep the
        // BF16 rows, so the copy sits beside them. Each packed copy is a `Mat<Bf16>` too, with
        // the form's word layout as its scales, so the uploader checks the row geometry.
        let mut packed = PackedSet::new(Form::from_env()?, Placement::Beside);
        let matrices: Vec<_> = std::iter::once(head.clone())
            .chain(packable)
            .map(|mat| (mat.data(), mat.rows(), mat.columns()))
            .collect();
        let packed_bytes = match packed.pack(&cuda, &kernels.packed.raw(), &matrices, 0) {
            Ok((_, packed_bytes)) => packed_bytes,
            Err(error) => {
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error);
            }
        };
        for (pointer, rows, columns) in &matrices {
            let Some(copy) = packed.get(*pointer) else {
                continue;
            };
            let planes = copy.planes();
            let name = labels
                .iter()
                .find(|(address, _)| address == &pointer.0)
                .map_or_else(
                    || format!("packed {rows} x {columns} BF16 projection"),
                    |(_, name)| format!("packed {name}"),
                );
            let spec = TensorSpec {
                name,
                dtype: DType::BF16,
                shape: vec![*rows, *columns],
            };
            let typed = match typed_bf16(
                &spec,
                *rows,
                *columns,
                Deviceptr(planes[0]),
                copy.row_bytes(),
                Scales::packed_bf16(PackedBf16 {
                    escapes: Deviceptr(planes[2]),
                    offsets: Deviceptr(planes[1]),
                    row_bytes: copy.row_bytes(),
                    order: PackedOrder::Walk,
                }),
            ) {
                Ok(typed) => typed,
                Err(error) => {
                    let references: Vec<&Deviceptr> = owned.iter().collect();
                    cuda.free_buffers(&references);
                    return Err(error);
                }
            };
            ledger_rows.push(LedgerRow::new(
                typed.spec().name.clone(),
                typed.format(),
                typed.format(),
                typed.bytes_per_token(),
            ));
        }
        owned.extend(packed.allocations());
        let weight_bytes = weight_bytes + packed_bytes;
        let ledger = match FormatLedger::boot(ledger_rows, &widen) {
            Ok(ledger) => ledger,
            Err(error) => {
                let references: Vec<&Deviceptr> = owned.iter().collect();
                cuda.free_buffers(&references);
                return Err(error.to_string());
            }
        };
        eprintln!("[dense] {ledger}");
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
                chunk,
                context,
                lanes,
                linear_sizes,
                scratch,
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

    /// The walk the runtime serves this backend by, for `capacity` lanes: [WalkArm::walk] as its
    /// concrete type, for the generated conformance checks: they open the runtime themselves so
    /// a check can drive a prompt, a continued turn and a rewind.
    ///
    /// # Errors
    /// A device allocation the walk's scratch cannot make.
    pub fn walk_arm(&self, capacity: usize) -> Result<DenseWalk, BackendError> {
        walk::walk(Arc::clone(&self.shared), capacity)
    }

    /// A session of its concrete type, for a caller that reads its per-layer trace.
    ///
    /// # Errors
    /// The session buffers or caches do not fit on the device.
    pub fn session(&self) -> Result<DenseDeviceSession, String> {
        DenseDeviceSession::open(Arc::clone(&self.shared))
    }

    /// Serves this backend's sessions on the tick engine ([sparkflash_runtime::serve]):
    /// `capacity` lanes whose decode rows share one walk, whose prompts run a part at a time
    /// between the lanes' steps, and which keep their conversations between requests. Call it
    /// before the backend is shared with the serving driver, with the driver's concurrency as the
    /// capacity.
    ///
    /// # Errors
    /// A group's rank, a zero capacity, a scratch or lane the device refuses, or a thread that
    /// cannot start.
    pub fn enable_fusion(&mut self, capacity: usize) -> Result<(), BackendError> {
        self.fused = Some(sparkflash_runtime::serve(self, capacity)?);
        Ok(())
    }

    /// [Self::enable_fusion] on rank 0 of an [Self::open_rank] group
    /// ([sparkflash_runtime::serve_group], on the tick group): every lane's prompt, its chunks,
    /// each step's shape, a give-back and a release is broadcast through `hub` before it runs
    /// here, and each worker replays them ([Self::mirror_group]) over its own `capacity` lanes,
    /// so lane `n` is the same sequence on every rank and the sums inside a walk line up.
    ///
    /// # Errors
    /// A backend that is not a group's rank, then as [Self::enable_fusion], or a watch that
    /// cannot start.
    pub fn enable_group_fusion(
        &mut self,
        capacity: usize,
        hub: sparkflash_comm::control::Tp4ControlHub,
    ) -> Result<(), BackendError> {
        self.fused = Some(sparkflash_runtime::serve_group(self, capacity, hub)?);
        Ok(())
    }

    /// A worker of an [Self::open_rank] group: replays rank 0's lane frames over `capacity`
    /// lanes of this rank's own until the group can no longer serve, then ends the process
    /// ([sparkflash_runtime::mirror]).
    ///
    /// # Errors
    /// A backend that is not a group's rank, or the lanes' scratch, before any frame.
    pub fn mirror_group(
        &self,
        control: &mut sparkflash_comm::control::Tp4ControlWorker,
        capacity: usize,
    ) -> Result<std::convert::Infallible, String> {
        sparkflash_runtime::mirror(self, control, capacity)
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

/// The prompt rows one pass takes, from [sparkflash_runtime::ChunkPolicy].
///
/// The default input is [CHUNK_ROWS], dense's legacy chunk, so a profile that names nothing gets
/// the 256 rows its kernels were chosen for. A profile may name a documented
/// [sparkflash_runtime::ChunkPolicy] size at or below [CHUNK_ROWS]; memory and the context are
/// the policy's business, and the answer is always a documented size. A larger size is refused
/// because the arm's scratch is allocated for [CHUNK_ROWS] rows.
fn chunk_rows(
    profile: &sparkflash_model::profile::ModelProfile,
    context: usize,
) -> Result<usize, String> {
    let asked = profile.prefill.chunk.unwrap_or(CHUNK_ROWS);
    if !ChunkPolicy::sizes().contains(&asked) {
        return Err(format!(
            "the shared dense device arm takes a documented prompt chunk (one of {:?}), not \
             prefill.chunk = {asked}; change the profile, or serve the checkpoint on the host \
             reference (SPARKFLASH_DENSE_DEVICE=0)",
            ChunkPolicy::sizes()
        ));
    }
    let rows = ChunkPolicy::for_limits(Limits::new(asked, context)).rows();
    if rows > CHUNK_ROWS {
        return Err(format!(
            "the shared dense device arm's scratch holds {CHUNK_ROWS} prompt rows, not \
             prefill.chunk = {asked}; change the profile, or serve the checkpoint on the host \
             reference (SPARKFLASH_DENSE_DEVICE=0)"
        ));
    }
    Ok(rows)
}

/// The wide-tile policy a typed profile asks of the arm, or the field it cannot honour.
///
/// The arm's prompt-and-decode split is fixed by its kernels; a field it does not honour is
/// refused by name rather than accepted and dropped. The chunk is [chunk_rows]'s. It honours
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
    if profile.prefill.enabled == Some(false) {
        return refuse("always runs its prompt as a prefill, but prefill.enabled = false");
    }
    if profile.prefill.exact == Some(true) {
        return refuse(
            "runs its prompt projections on the wide tile and its decode rows on the row GEMV, \
             which are not bit-identical, but prefill.exact = true",
        );
    }
    if profile.decode.tensor_cores == Some(true)
        && let State::Off(why) = TOKEN_DECODE
    {
        // The walk's capability row and this refusal are the same constant, so a dense arm that
        // ever declares a tensor-core token body stops refusing the switch by the same edit.
        return refuse(&format!("cannot honour decode.tensor_cores = true: {why}"));
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

/// Every recurrent region the parts declare, in bytes and in walk order: a Gated DeltaNet
/// layer's conv history, then its delta-rule state.
fn linear_sizes(layers: &[Layer]) -> Vec<usize> {
    branches(layers)
        .flat_map(|branch| branch.part.state())
        .filter_map(|region| match *region {
            Region::Recurrent { bytes, .. } => Some(bytes),
            Region::Positional { .. } => None,
        })
        .collect()
}

/// What [upload] leaves on the device.
struct Uploaded {
    embed: Mat<Bf16>,
    head: Mat<Bf16>,
    norm: Deviceptr,
    layers: Vec<Layer>,
    /// The device bytes the weights claimed.
    bytes: usize,
    /// The boot format ledger's rows.
    rows: Vec<LedgerRow>,
    widen: WidenLog,
    /// Each uploaded matrix's device address and its checkpoint name.
    labels: Vec<(u64, String)>,
    /// The BF16 projections the token GEMVs read a packed copy of, in upload order.
    packable: Vec<Mat<Bf16>>,
    /// The parts' scratch reservations ([Shared::scratch]).
    scratch: Vec<usize>,
}

/// The typed matrix of a stored BF16 tensor this arm reads: [upload] checks the container's
/// stored dtype and the scale count of the format (none, or the packed word layout).
fn typed_bf16(
    spec: &TensorSpec,
    rows: usize,
    columns: usize,
    data: Deviceptr,
    stride: usize,
    scales: Scales,
) -> Result<Mat<Bf16>, String> {
    let stored = TensorSpec {
        name: spec.name.clone(),
        dtype: spec.dtype,
        shape: vec![rows, columns],
    };
    upload_format::<Bf16>(Tensor {
        stored: &stored,
        shape: &[rows, columns],
        data,
        stride,
        scales,
    })
    .map_err(|error| error.to_string())
}

/// The typed matrix of a stored FP8 E4M3 tensor: [upload] checks the stored dtype and that the
/// inverse-scale plane has exactly one entry a 128 x 128 tile.
#[allow(clippy::too_many_arguments)]
fn typed_fp8(
    spec: &TensorSpec,
    rows: usize,
    columns: usize,
    data: Deviceptr,
    stride: usize,
    blocks: Deviceptr,
    entries: usize,
    scale_dtype: DType,
) -> Result<Mat<Fp8E4m3>, String> {
    let stored = TensorSpec {
        name: spec.name.clone(),
        dtype: spec.dtype,
        shape: vec![rows, columns],
    };
    upload_format::<Fp8E4m3>(Tensor {
        stored: &stored,
        shape: &[rows, columns],
        data,
        stride,
        scales: Scales::fp8_e4m3(blocks, entries, scale_dtype),
    })
    .map_err(|error| error.to_string())
}

/// The typed matrix of a stored NVFP4 expert weight: [upload] checks the packed U8 container
/// against the logical shape and that the per-16 scale plane has one entry a group.
fn typed_nvfp4(
    spec: &TensorSpec,
    rows: usize,
    columns: usize,
    data: Deviceptr,
    blocks: Deviceptr,
    entries: usize,
    global: f32,
) -> Result<Mat<NvFp4>, String> {
    let stored = TensorSpec {
        name: spec.name.clone(),
        dtype: spec.dtype,
        shape: vec![rows, columns.div_ceil(2)],
    };
    upload_format::<NvFp4>(Tensor {
        stored: &stored,
        shape: &[rows, columns],
        data,
        stride: columns.div_ceil(2),
        scales: Scales::nvfp4(blocks, entries, global),
    })
    .map_err(|error| error.to_string())
}

/// The shape `runs` select from a `[rows, columns]` matrix: row runs stacked, or column runs
/// side by side.
fn run_shape(runs: &[Run], (rows, columns): (usize, usize)) -> Result<(usize, usize), String> {
    let stacked = runs.iter().all(|run| matches!(run, Run::Rows(..)));
    if runs.is_empty() {
        return Err("no part of the matrix is named".to_owned());
    }
    if !stacked && !runs.iter().all(|run| matches!(run, Run::Columns(..))) {
        return Err("the parts mix row and column runs".to_owned());
    }
    let mut total = 0;
    for run in runs {
        let (first, count, within) = match *run {
            Run::Rows(first, count) => (first, count, rows),
            Run::Columns(first, count) => (first, count, columns),
        };
        if count == 0 || first + count > within {
            return Err(format!(
                "{run:?} is not a part of a [{rows}, {columns}] matrix"
            ));
        }
        total += count;
    }
    Ok(if stacked {
        (total, columns)
    } else {
        (rows, total)
    })
}

/// The bytes `runs` select from the row-major `bytes` of a `[rows, columns]` matrix stored
/// `unit` logical values to a byte, and the logical shape of the selection.
fn run_slice(
    bytes: &[u8],
    (rows, columns): (usize, usize),
    runs: &[Run],
    unit: usize,
) -> Result<(usize, usize, Vec<u8>), String> {
    if !columns.is_multiple_of(unit) {
        return Err(format!(
            "the [{rows}, {columns}] matrix is not a whole number of {unit}-value groups"
        ));
    }
    let (out_rows, out_columns) = run_shape(runs, (rows, columns))?;
    let stored_columns = columns / unit;
    let row = stored_columns;
    if bytes.len() != rows * row {
        return Err(format!(
            "the tensor is {} bytes, expected {} for a [{rows}, {columns}] matrix {unit} values a byte",
            bytes.len(),
            rows * row
        ));
    }
    if runs.iter().all(|run| matches!(run, Run::Rows(..))) {
        let out = runs
            .iter()
            .flat_map(|run| match *run {
                Run::Rows(first, count) | Run::Columns(first, count) => {
                    &bytes[first * row..(first + count) * row]
                }
            })
            .copied()
            .collect();
        return Ok((out_rows, out_columns, out));
    }
    let mut out = Vec::with_capacity(out_rows * out_columns / unit);
    for index in 0..rows {
        let start = index * row;
        for run in runs {
            let Run::Columns(first, count) = *run else {
                unreachable!("the runs are all columns");
            };
            if !first.is_multiple_of(unit) || !count.is_multiple_of(unit) {
                return Err(format!("{run:?} splits a {unit}-value column group"));
            }
            out.extend_from_slice(&bytes[start + first / unit..start + (first + count) / unit]);
        }
    }
    Ok((out_rows, out_columns, out))
}

/// One upload's tensors: each checked at its architecture shape (BF16, or a projection's
/// block-scaled E4M3), through [upload] so its stored dtype and scale count are checked too, and
/// every allocation in `owned` as it is made, so a failure part way frees what was uploaded.
struct Uploads<'a> {
    uploader: DraftUploader<'a>,
    source: &'a dyn WeightSource,
    formats: Formats,
    owned: &'a mut Vec<Deviceptr>,
    bytes: usize,
    /// Every part's stored and arithmetic format and the bytes it reads for one token, in upload
    /// order: the boot format ledger's rows.
    rows: Vec<LedgerRow>,
    /// Each uploaded matrix's device address and its checkpoint name, so the packed copies built
    /// after the upload can be named in the ledger.
    labels: Vec<(u64, String)>,
    /// The one widening this load path takes (a BF16 router bias read at f32), recorded with its
    /// reason so boot can accept it.
    widen: WidenLog,
    /// The BF16 projections the token GEMVs read a packed copy of, as each part's constructor
    /// names them.
    packable: Vec<Mat<Bf16>>,
}

impl Uploads<'_> {
    fn keep(&mut self, pointer: Deviceptr, bytes: usize) -> Deviceptr {
        self.owned.push(pointer);
        self.bytes += bytes;
        pointer
    }

    /// Names the BF16 ones of `projections` for a packed copy, in this order.
    fn pack(&mut self, projections: &[&Proj]) {
        for projection in projections {
            match projection {
                Proj::Bf16(mat) => self.packable.push(mat.clone()),
                Proj::Fp8(_) | Proj::NvFp4(_) => {}
            }
        }
    }

    /// One ledger row for a typed matrix, with the bytes one token reads from it.
    fn record(&mut self, part: &str, format: FormatKind, bytes: u128, data: Deviceptr) {
        self.rows.push(LedgerRow::new(part, format, format, bytes));
        self.labels.push((data.0, part.to_owned()));
    }

    fn mat(&mut self, name: &str, rows: usize, columns: usize) -> Result<Mat<Bf16>, String> {
        bf16(self.source, name, &[rows, columns])?;
        let spec = self.source.spec(name).map_err(|error| error.to_string())?;
        let mat = self.uploader.mat(name, rows, columns)?;
        let typed = typed_bf16(
            &spec,
            rows,
            columns,
            mat.pointer,
            columns * 2,
            Scales::none(),
        )?;
        self.keep(mat.pointer, rows * columns * 2);
        self.record(name, typed.format(), typed.bytes_per_token(), typed.data());
        Ok(typed)
    }

    /// A BF16 vector (a norm, a rotary table, a conv filter).
    ///
    /// A vector stays a `Deviceptr`: [Mat] is a matrix with a row stride and a format's scale
    /// plane, and these tensors have neither. Its stored and arithmetic formats and its bytes a
    /// token are still ledger rows.
    fn vector(&mut self, name: &str, width: usize) -> Result<Deviceptr, String> {
        bf16(self.source, name, &[width])?;
        let pointer = self.uploader.vector(name, width)?;
        self.rows.push(LedgerRow::new(
            name,
            FormatKind::Bf16,
            FormatKind::Bf16,
            (width * 2) as u128,
        ));
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
        if spec.dtype == DType::U8 {
            if self.formats.expert != ExpertForm::NvFp4 {
                return Err(format!(
                    "{name} is U8, the NVFP4 expert form, and the checkpoint declares {:?} experts",
                    self.formats.expert
                ));
            }
            let codes = self.uploader.read(name, rows * columns.div_ceil(2))?;
            let scales_name = name
                .strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.weight_scale"))
                .ok_or_else(|| format!("{name} is not a .weight, so it names no weight_scale"))?;
            let global_name = name
                .strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.weight_scale_2"))
                .ok_or_else(|| format!("{name} is not a .weight, so it names no weight_scale_2"))?;
            let scales = self
                .uploader
                .read(&scales_name, rows * columns.div_ceil(16))?;
            let global = self.uploader.read(&global_name, 4)?;
            let global = f32::from_le_bytes(
                global
                    .as_chunks::<4>()
                    .0
                    .first()
                    .copied()
                    .ok_or_else(|| format!("{global_name} holds no f32"))?,
            );
            let (out_rows, out_columns, codes) = run_slice(&codes, (rows, columns), runs, 2)?;
            let (_, _, scales) = run_slice(&scales, (rows, columns), runs, 16)?;
            let entries = out_rows * out_columns.div_ceil(16);
            let data = self.keep(self.uploader.bytes(&codes)?, codes.len());
            let blocks = self.keep(self.uploader.bytes(&scales)?, scales.len());
            let typed = typed_nvfp4(&spec, out_rows, out_columns, data, blocks, entries, global)?;
            self.record(name, typed.format(), typed.bytes_per_token(), typed.data());
            return Ok(Proj::NvFp4(typed));
        }
        if spec.dtype != DType::F8E4M3 {
            bf16(self.source, name, &[rows, columns])?;
            let mat = self.uploader.mat_parts(name, (rows, columns), runs)?;
            let typed = typed_bf16(
                &spec,
                mat.rows,
                mat.columns,
                mat.pointer,
                mat.columns * 2,
                Scales::none(),
            )?;
            self.keep(mat.pointer, mat.rows * mat.columns * 2);
            self.record(name, typed.format(), typed.bytes_per_token(), typed.data());
            return Ok(Proj::Bf16(typed));
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
        let typed = typed_fp8(
            &spec,
            mat.rows,
            mat.columns,
            mat.weight,
            mat.columns,
            mat.scales,
            grid,
            DType::BF16,
        )?;
        self.record(name, typed.format(), typed.bytes_per_token(), typed.data());
        Ok(Proj::Fp8(typed))
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
    ///
    /// This is the one widen-line waiver: a vector is not a [Mat], and the stored BF16 tensor is
    /// read at f32 arithmetic. The `widen` record below is what boot accepts, and the ledger row
    /// says what was stored and what is read.
    fn floats(&mut self, name: &str, elements: usize) -> Result<Deviceptr, String> {
        let spec = self.source.spec(name).map_err(|error| error.to_string())?;
        if spec.elements() != elements as u128 {
            return Err(format!(
                "{name} is {:?}, expected {elements} values",
                spec.shape
            ));
        }
        let (stored, bytes) = match spec.dtype {
            DType::F32 => (FormatKind::F32, self.uploader.read(name, elements * 4)?),
            DType::BF16 => {
                // The one widening this arm takes, and it is exact: every BF16 value is an f32.
                widen::<Bf16, F32Format>(
                    name,
                    "the router bias is f32 arithmetic; the release stores bf16, and every bf16 \
                     value is an f32",
                    &mut self.widen,
                )
                .map_err(|error| error.to_string())?;
                (
                    FormatKind::Bf16,
                    self.uploader
                        .read(name, elements * 2)?
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .flat_map(|word| {
                            f32::from_bits(u32::from(u16::from_le_bytes(*word)) << 16).to_le_bytes()
                        })
                        .collect(),
                )
            }
            other => {
                return Err(format!(
                    "{name} is {other:?}; the shared dense device arm reads an F32 or BF16 vector"
                ));
            }
        };
        let pointer = self.uploader.bytes(&bytes)?;
        // The device arithmetic reads the f32 vector; the row says what was stored and what is
        // read, and the record above is what lets boot accept the difference.
        self.rows.push(LedgerRow::new(
            name,
            stored,
            FormatKind::F32,
            (elements * 4) as u128,
        ));
        Ok(self.keep(pointer, elements * 4))
    }

    fn vector_runs(
        &mut self,
        name: &str,
        (rows, width): (usize, usize),
        runs: &[Run],
    ) -> Result<Deviceptr, String> {
        let mat = self.uploader.mat_parts(name, (rows, width), runs)?;
        self.rows.push(LedgerRow::new(
            name,
            FormatKind::Bf16,
            FormatKind::Bf16,
            (mat.rows * width * 2) as u128,
        ));
        Ok(self.keep(mat.pointer, mat.rows * width * 2))
    }

    /// A grouped-query attention layer: its projections, its q/k norms and its output projection.
    fn gqa(
        &mut self,
        index: usize,
        names: &AttentionNames,
        (shape, whole): (&Shape, &Shape),
        hidden: usize,
    ) -> Result<Gqa, String> {
        let cut = shape.cut;
        let (
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
        ) = (names, shape.linear)
        else {
            return Err(format!("layer {index} names another token mixer's tensors"));
        };
        // A query head's rows are its query and, when gated, its gate, interleaved.
        let query_head = shape.query_rows() / shape.heads.max(1);
        let head_dim = shape.head_dim;
        let kv = [Run::Rows(cut.kv_heads * head_dim, shape.kv_width())];
        let q = self.proj_runs(
            q,
            (whole.query_rows(), hidden),
            &[Run::Rows(cut.heads * query_head, shape.query_rows())],
        )?;
        let k = self.proj_runs(k, (whole.kv_width(), hidden), &kv)?;
        let v = self.proj_runs(v, (whole.kv_width(), hidden), &kv)?;
        let q_norm = q_norm
            .as_deref()
            .map(|name| self.vector(name, shape.head_dim))
            .transpose()?;
        let k_norm = k_norm
            .as_deref()
            .map(|name| self.vector(name, shape.head_dim))
            .transpose()?;
        let o = self.proj_runs(
            o,
            (hidden, whole.query_width()),
            &[Run::Columns(cut.heads * head_dim, shape.query_width())],
        )?;
        self.pack(&[&o, &q, &k, &v]);
        // Keys and values, BF16 a value.
        let cache = Region::Positional {
            name: "kv",
            bytes_per_row: shape.kv_width() * 2,
        };
        Ok(Gqa {
            shape: *shape,
            q,
            k,
            v,
            q_norm,
            k_norm,
            o,
            state: [cache, cache],
        })
    }

    /// A Gated DeltaNet layer: its projections, conv, decay vectors, gated norm and `out_proj`.
    fn gdn(
        &mut self,
        index: usize,
        names: &AttentionNames,
        (shape, whole): (&Shape, &Shape),
        hidden: usize,
    ) -> Result<Gdn, String> {
        let cut = shape.cut;
        let (AttentionNames::Gdn(names), Some(linear)) = (names, shape.linear) else {
            return Err(format!("layer {index} names another token mixer's tensors"));
        };
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
        let weights = GdnLayer {
            qkv: self.proj_runs(qkv, (all.conv_width(), hidden), &qkv_runs)?,
            z: self.proj_runs(z, (all.value_width(), hidden), &values)?,
            b: self.proj_runs(b, (all.heads, hidden), &value_heads)?,
            a: self.proj_runs(a, (all.heads, hidden), &value_heads)?,
            conv: self.conv(conv, (all.conv_width(), geometry.taps), &qkv_runs)?,
            a_log: self.vector_runs(a_log, (all.heads, 1), &value_heads)?,
            dt_bias: self.vector_runs(dt_bias, (all.heads, 1), &value_heads)?,
            norm: self.vector(norm, value_dim)?,
        };
        let o = self.proj_runs(
            out,
            (hidden, all.value_width()),
            &[Run::Columns(
                cut.value_heads * value_dim,
                geometry.value_width(),
            )],
        )?;
        self.pack(&[&o, &weights.qkv, &weights.z, &weights.b, &weights.a]);
        Ok(Gdn {
            shape: *shape,
            weights,
            o,
            state: [
                Region::Recurrent {
                    name: "gated-deltanet",
                    bytes: geometry.conv_bytes(),
                },
                Region::Recurrent {
                    name: "gated-deltanet",
                    bytes: geometry.state_bytes(),
                },
            ],
        })
    }

    /// A layer's dense SwiGLU MLP.
    fn dense_mlp(
        &mut self,
        index: usize,
        names: &FfnNames,
        (shape, whole): (&Shape, &Shape),
        hidden: usize,
    ) -> Result<DenseMlp, String> {
        let (None, FfnNames::Dense(names)) = (&shape.routed, names) else {
            return Err(format!("layer {index} names a routed MLP's tensors"));
        };
        let mlp = self.mlp(
            names,
            whole.intermediate,
            (shape.cut.intermediate, shape.intermediate),
            hidden,
        )?;
        self.pack(&[&mlp.gate, &mlp.up, &mlp.down]);
        Ok(DenseMlp {
            mlp,
            clamp: shape.clamp,
        })
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
        hidden: usize,
    ) -> Result<RoutedExperts, String> {
        let Some(routed) = &shape.routed else {
            return Err(format!("layer {index} is not a routed layer"));
        };
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
        let mut scales_table = Vec::with_capacity(routed.experts * 3 * 4);
        let form = self.formats.expert;
        if form == ExpertForm::NvFp4 && shape.clamp.is_some() {
            return Err(format!(
                "layer {index} stores NVFP4 experts under a swiglu_limit, and the NVFP4 token body \
                 has no clamped form"
            ));
        }
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
                if proj.form() != form {
                    return Err(format!(
                        "layer {index}'s expert {number} stores {name} as {:?}, and the checkpoint \
                         declares {form:?} experts, the one form the expert passes are loaded in",
                        proj.form()
                    ));
                }
                for address in proj.table() {
                    table.extend(address.to_le_bytes());
                }
                if let Some(global) = proj.global() {
                    scales_table.extend(global.to_le_bytes());
                }
            }
        }
        let table = self.uploader.bytes(&table)?;
        let table = self.keep(table, routed.experts * 6 * 8);
        let scales = match form {
            ExpertForm::NvFp4 => {
                let bytes = self.uploader.bytes(&scales_table)?;
                Some(self.keep(bytes, routed.experts * 3 * 4))
            }
            ExpertForm::Bf16 | ExpertForm::Fp8 => None,
        };
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
                let spec = self.source.spec(name).map_err(|error| error.to_string())?;
                let typed = typed_bf16(&spec, 1, hidden, pointer, hidden * 2, Scales::none())?;
                self.keep(pointer, hidden * 2);
                self.record(name, typed.format(), typed.bytes_per_token(), typed.data());
                Some(Proj::Bf16(typed))
            }
            None => None,
        };
        if shared_gate.is_some() && shared.is_none() {
            return Err(format!(
                "layer {index} names a shared-expert gate without a shared expert"
            ));
        }
        self.pack(&[&router]);
        if let Some(shared) = &shared {
            self.pack(&[&shared.gate, &shared.up, &shared.down]);
        }
        if let Some(gate) = &shared_gate {
            self.pack(&[gate]);
        }
        Ok(RoutedExperts {
            shape: *shape,
            router,
            bias,
            scales,
            table,
            shared,
            shared_gate,
        })
    }
}

/// Uploads every tensor `names` resolves, each checked at its architecture shape (BF16, or a
/// projection's block-scaled E4M3 under `formats`); every allocation lands in `owned` as it is
/// made, so a failure part way frees what was uploaded. Each layer's branches are the parts its
/// step names, built by `parts`' constructors in forward order.
#[allow(clippy::too_many_arguments)]
fn upload<'a>(
    cuda: &'a Cuda,
    architecture: &'a Architecture,
    (names, steps): (&'a TensorNames, &[LayerStep]),
    source: &'a dyn WeightSource,
    formats: Formats,
    (shapes, shard): (&'a [Shape], Shard),
    parts: &DenseParts,
    owned: &'a mut Vec<Deviceptr>,
) -> Result<Uploaded, String> {
    let mut open = OpenCx {
        up: Uploads {
            uploader: DraftUploader::new(cuda, source),
            source,
            formats,
            owned,
            bytes: 0,
            labels: Vec::new(),
            rows: Vec::new(),
            widen: WidenLog::new(),
            packable: Vec::new(),
        },
        cuda,
        architecture,
        names,
        shapes,
        plan: shard.0,
        index: 0,
        whole: None,
        scratch: Vec::new(),
    };
    let up = &mut open.up;
    let hidden = architecture.hidden_size;
    let vocab = architecture.vocab_size;
    let embed = up.mat(&names.embed, vocab, hidden)?;
    // A token gathers one embedding row, not the whole table.
    if let Some(row) = up.rows.last_mut() {
        row.bytes_per_token = (hidden * 2) as u128;
    }
    let head = if names.head_tied() {
        embed.clone()
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
    if steps.len() != shapes.len() {
        return Err(format!(
            "the architecture walks {} layers and declares {}",
            steps.len(),
            shapes.len()
        ));
    }
    let mut slot = 0usize;
    for (index, (layer_names, shape)) in names.layers.iter().zip(shapes).enumerate() {
        open.index = index;
        open.whole = Some(Shape::of(architecture, index, Shard::whole())?);
        // The parts first, in forward order, then the norms each reads its input through.
        let mut built = Vec::new();
        for (role, kind) in sparkflash_runtime::plan_roles(&steps[index])? {
            let part = parts
                .build(kind, &mut open)
                .map_err(|error| format!("layer {index}'s part {}: {error}", kind.name()))?;
            built.push((role, part));
        }
        let input_norm = open.up.vector(&layer_names.input_norm, hidden)?;
        let post_norm = open.up.vector(&layer_names.post_norm, hidden)?;
        let mut branches = Vec::with_capacity(built.len());
        for (role, part) in built {
            branches.push(Branch {
                norm: match role {
                    StepRole::Attention => input_norm,
                    _ => post_norm,
                },
                summed: part.split(&open.plan) == RankSplit::Summed,
                part,
                slot,
            });
            slot += 1;
        }
        layers.push(Layer {
            shape: *shape,
            branches,
        });
    }
    let OpenCx { up, scratch, .. } = open;
    Ok(Uploaded {
        embed,
        head,
        norm,
        layers,
        bytes: up.bytes,
        rows: up.rows,
        widen: up.widen,
        labels: up.labels,
        packable: up.packable,
        scratch,
    })
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

    fn call<const N: usize>(
        &self,
        kernel: impl Into<Kernel>,
        call: Call<N>,
    ) -> Result<(), BackendError> {
        let kernel = kernel.into();
        self.cuda
            .launch_async(kernel, call.grid, call.block, &call.args)
            .map_err(|error| BackendError::new(DEVICE, format!("{}: {error}", kernel.name())))
    }

    /// A grouped-query attention layer over the normed rows in `b.normed`, into `b.attended`: the
    /// row-wise projections and norms over every row, the rotary, cache appends and attention a
    /// segment at a time.
    fn attend(&self, part: &Gqa, cx: &PassCx<'_>) -> Result<(), BackendError> {
        let &PassCx {
            b,
            segments,
            rows,
            policy,
            index,
            slot,
            ..
        } = cx;
        let Gqa {
            q,
            k,
            v,
            q_norm,
            k_norm,
            ..
        } = part;
        let kernels = self.kernels;
        let shape = part.shape;
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
            let &[keys, values] = sequence.lane.regions[slot].as_slice() else {
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
    fn linear(&self, part: &Gdn, cx: &PassCx<'_>) -> Result<(), BackendError> {
        let &PassCx {
            b,
            segments,
            rows,
            policy,
            index,
            slot,
            ..
        } = cx;
        let gdn = &part.weights;
        let refuse = |message: String| Err(BackendError::new(DEVICE, message));
        let (Some(kernels), Some(linear)) = (self.kernels.gdn, part.shape.linear) else {
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
            let &[history, _] = sequence.lane.regions[slot].as_slice() else {
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
            let &[_, state] = sequence.lane.regions[slot].as_slice() else {
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
                .launch_async(kernels.scan.kernel, grid, block, &args)
                .map_err(|error| {
                    BackendError::new(DEVICE, format!("{}: {error}", kernels.scan.kernel.name()))
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
            Proj::NvFp4(_) => Err(BackendError::new(
                DEVICE,
                "an NVFP4 dense projection: only the routed experts are stored that way",
            )),
        }
    }

    /// A block-scaled FP8 projection: the FP8 tensor-core tile ([Fp8Tile]) where [route_fp8]
    /// takes it and the operands are 16-byte aligned, else the tiled E4M3 GEMV a group of rows at
    /// a time. Both keep the activations f32 and apply each tile's scale to its sums, the host
    /// reference's arithmetic to a tolerance.
    fn project_fp8(
        &self,
        mat: &Mat<Fp8E4m3>,
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
                kernels.gemv.kernel,
                &GpuMatE4m3Tile {
                    weight: mat.data(),
                    scales: mat.scales().blocks().unwrap_or(Deviceptr(0)),
                    rows: mat.rows(),
                    columns: mat.columns(),
                },
                Deviceptr(input),
                Deviceptr(output),
                batch,
            )
        };
        let batched = || {
            for start in (0..rows).step_by(GEMV_MAX_BATCH) {
                gemv(
                    input + (start * mat.columns() * 4) as u64,
                    output + (start * mat.rows() * 4) as u64,
                    GEMV_MAX_BATCH.min(rows - start),
                )?;
            }
            Ok(())
        };
        let aligned = [
            mat.data().0,
            mat.scales().blocks().map_or(0, |blocks| blocks.0),
            input,
            output,
        ]
        .iter()
        .all(|address| address.is_multiple_of(16));
        match route_fp8(policy, mat.rows(), mat.columns(), rows) {
            Route::RowGemv => gemv(input, output, rows),
            Route::Wide if aligned => {
                let tile = Fp8Tile::call(
                    mat.data().0,
                    mat.scales().blocks().map_or(0, |blocks| blocks.0),
                    input,
                    output,
                    mat.rows(),
                    mat.columns(),
                    rows,
                )
                .map_err(|error| BackendError::new(DEVICE, error))?;
                self.cuda
                    .launch_async(kernels.tile.kernel, tile.grid, tile.block, &tile.args)
                    .map_err(|error| BackendError::new(DEVICE, format!("q4p_dense_fp8: {error}")))
            }
            Route::Wide | Route::BatchedGemv => batched(),
            route @ (Route::Skinny | Route::Refuse) => Err(BackendError::new(
                DEVICE,
                format!(
                    "the dense policy routed an FP8 {} x {} projection to {route:?}",
                    mat.rows(),
                    mat.columns()
                ),
            )),
        }
    }

    /// `output[rows][mat.rows] = input[rows][mat.columns] x matᵀ` on the kernel
    /// [route_bf16] picks under `policy` (`None` keeps the row GEMV).
    fn project_bf16(
        &self,
        mat: &Mat<Bf16>,
        input: u64,
        output: u64,
        rows: usize,
        policy: Option<WidePolicy>,
    ) -> Result<(), BackendError> {
        // The packed copy where there is one and the activations are a 16-byte load: the same
        // bits from about three quarters of the bytes.
        let packed = self.packed.get(mat.data());
        let gemv = |input: u64, output: u64, batch: usize| match packed
            .filter(|_| input.is_multiple_of(16))
        {
            Some(rows) => self.call(
                self.kernels.packed.gemv,
                packed::gemv(
                    rows,
                    Deviceptr(input),
                    Deviceptr(output),
                    (mat.rows(), mat.columns()),
                    batch,
                    1,
                )
                .map_err(|error| BackendError::new(DEVICE, error))?,
            ),
            None => self.call(
                self.kernels.gemv,
                calls::gemv_bf16(
                    mat.data().0,
                    input,
                    output,
                    (mat.rows(), mat.columns()),
                    batch,
                ),
            ),
        };
        match route_bf16(policy, mat.rows(), mat.columns(), rows) {
            Route::RowGemv => gemv(input, output, rows),
            Route::BatchedGemv => {
                for start in (0..rows).step_by(GEMV_MAX_BATCH) {
                    gemv(
                        input + (start * mat.columns() * 4) as u64,
                        output + (start * mat.rows() * 4) as u64,
                        GEMV_MAX_BATCH.min(rows - start),
                    )?;
                }
                Ok(())
            }
            Route::Wide => {
                let tile =
                    WideTile::call(mat.data().0, input, output, mat.rows(), mat.columns(), rows)
                        .map_err(|error| BackendError::new(DEVICE, error))?;
                self.cuda
                    .launch_async(self.kernels.wide.kernel, tile.grid, tile.block, &tile.args)
                    .map_err(|error| BackendError::new(DEVICE, format!("dory_mma_wide: {error}")))
            }
            route @ (Route::Skinny | Route::Refuse) => Err(BackendError::new(
                DEVICE,
                format!(
                    "the dense policy routed a {} x {} projection to {route:?}",
                    mat.rows(),
                    mat.columns()
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
        kernel: impl Into<Kernel>,
        (grid, block): Launch,
        args: &[u64],
    ) -> Result<(), BackendError> {
        let kernel = kernel.into();
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
    fn routed(&self, part: &RoutedExperts, cx: &PassCx<'_>) -> Result<(), BackendError> {
        let &PassCx {
            b, rows, policy, ..
        } = cx;
        let (
            Some(routed),
            RoutedExperts {
                shape,
                router,
                bias,
                scales,
                table,
                shared,
                shared_gate,
            },
            Some(kernels),
        ) = (&part.shape.routed, part, self.kernels.moe)
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
        // The NVFP4 body reads the per-tensor scale table as its second argument; the BF16 and
        // FP8 bodies read the same six-address table with no second table.
        let mut gate_up = Vec::with_capacity(11);
        gate_up.push(table.0);
        if kernels.nvfp4 {
            gate_up.push(scales.unwrap_or(Deviceptr(0)).0);
        }
        gate_up.extend([
            b.indices.0,
            b.sorted.0,
            b.runs.0,
            b.normed.0,
            b.activated.0,
            top_k as u64,
            hidden as u64,
            intermediate as u64,
        ]);
        let grid = token::gate_up_grid(intermediate, top_k, rows);
        match shape.clamp {
            None => self.launch(kernels.gate_up, grid, &gate_up)?,
            Some(limit) => {
                let Some(clamped_kernel) = kernels.gate_up_clamped else {
                    return Err(BackendError::new(
                        DEVICE,
                        "an NVFP4 expert pass under a swiglu_limit: the body has no clamped form",
                    ));
                };
                let mut clamped = gate_up.clone();
                clamped.push(u64::from(limit.to_bits()));
                self.launch(clamped_kernel, grid, &clamped)?;
            }
        }
        let mut down = Vec::with_capacity(12);
        down.push(table.0);
        if kernels.nvfp4 {
            down.push(scales.unwrap_or(Deviceptr(0)).0);
        }
        down.extend([
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
        ]);
        self.launch(kernels.down, token::down_grid(hidden, top_k, rows), &down)?;
        match shared {
            Some(mlp) => self.mlp(mlp, shape.clamp, b, b.shared_out, rows, policy)?,
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
        let spec = TensorSpec {
            name: "the tiled GEMV's matrix".to_owned(),
            dtype: DType::BF16,
            shape: vec![matrix.rows, matrix.columns],
        };
        let typed = typed_bf16(
            &spec,
            matrix.rows,
            matrix.columns,
            matrix.pointer,
            matrix.columns * 2,
            Scales::none(),
        )
        .map_err(|error| BackendError::new(DEVICE, error))?;
        self.shared
            .project_bf16(&typed, input.0, output.0, rows, None)
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
///
/// The lane is laid out from what each part declares
/// ([sparkflash_runtime::PartExec::state]), in walk order: a positional region is `capacity`
/// rows of its row, a recurrent region its bytes.
#[derive(Debug)]
struct LaneParts {
    /// Each part's regions' offsets in the lane, by [Branch::slot].
    regions: Vec<Vec<usize>>,
    arena: Option<usize>,
}

impl LaneParts {
    fn plan(layout: &mut Layout, layers: &[Layer], capacity: usize, rewind: bool) -> Self {
        let regions = branches(layers)
            .map(|branch| {
                branch
                    .part
                    .state()
                    .iter()
                    .map(|region| match *region {
                        Region::Positional { bytes_per_row, .. } => {
                            layout.part(capacity * bytes_per_row, bytes_per_row, 0)
                        }
                        Region::Recurrent { bytes, .. } => layout.part(bytes, 0, 0),
                    })
                    .collect()
            })
            .collect();
        let sizes = linear_sizes(layers);
        let arena = (rewind && !sizes.is_empty())
            .then(|| layout.part(Checkpoint::arena_bytes(&sizes), 0, 0));
        Self { regions, arena }
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
    /// The greedy head's pick a row, from `q4e_argmax_slices`.
    picks: Deviceptr,
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
    /// The parts' reserved scratch, by [ScratchSlot]: `[rows][floats]` f32 each.
    extra: Vec<Deviceptr>,
    logit_rows: usize,
}

impl Buffers {
    fn all(&self) -> Vec<&Deviceptr> {
        let mut all = vec![
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
            &self.picks,
        ];
        all.extend(&self.extra);
        all
    }
}

/// A walk's scratch on the device, freed with it. Nothing in it outlives a walk but the logits the
/// walk's caller reads back, so the slots of one engine share one.
pub struct Scratch {
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
            rows,
        ];
        let fixed = floats.len();
        let reserved = shared.scratch.iter().map(|floats| rows * floats);
        let mut pointers = Vec::with_capacity(fixed + shared.scratch.len());
        let mut total = 0usize;
        for count in floats.into_iter().chain(reserved) {
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
            picks: p(32),
            extra: pointers[fixed..].to_vec(),
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
pub struct DenseLane {
    shared: Arc<Shared>,
    memory: SequenceLane<LaneParts>,
    /// Each part's regions in the lane, by [Branch::slot], in the order the part declares them.
    regions: Vec<Vec<Deviceptr>>,
    /// The recurrent ones among them, in walk order, as [Shared::linear_sizes] counts them.
    recurrent: Vec<Deviceptr>,
    /// The Gated DeltaNet layers' rewind point, when a layer is one and the sequence keeps one.
    checkpoint: Option<Checkpoint>,
}

/// One sequence of the dense walk: its lane, and the ids and sampler the shared walk keeps.
pub type Sequence = PlanSeq<DenseLane>;

impl Drop for DenseLane {
    fn drop(&mut self) {
        self.memory.free(&self.shared.cuda);
    }
}

impl DenseLane {
    fn open(shared: Arc<Shared>) -> Result<Self, String> {
        let memory = SequenceLane::reserve(&shared.cuda, &shared.lanes)?;
        let base = memory.base();
        let parts = shared.lanes.parts();
        let regions: Vec<Vec<Deviceptr>> = parts
            .regions
            .iter()
            .map(|part| part.iter().map(|offset| base.offset(*offset)).collect())
            .collect();
        let recurrent = branches(&shared.layers)
            .flat_map(|branch| branch.part.state().iter().zip(&regions[branch.slot]))
            .filter(|(region, _)| !region.is_positional())
            .map(|(_, pointer)| *pointer)
            .collect();
        let checkpoint = parts
            .arena
            .map(|arena| Checkpoint::new(base.offset(arena), shared.linear_sizes.clone()));
        Ok(Self {
            shared,
            memory,
            regions,
            recurrent,
            checkpoint,
        })
    }

    /// The live recurrent regions in walk order, as [Shared::linear_sizes] counts them: each
    /// Gated DeltaNet layer's conv history, then its state.
    fn linear_regions(&self) -> Vec<Deviceptr> {
        self.recurrent.clone()
    }

    /// Sets the recurrent state to the empty sequence's: zero, as a fresh lane's chunks map. A
    /// lane with nothing backed has nothing to clear.
    fn clear_linear(&mut self) -> Result<(), BackendError> {
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint.forget();
        }
        if self.recurrent.is_empty() || self.memory.span().backed() == 0 {
            return Ok(());
        }
        for (region, bytes) in self.recurrent.iter().zip(&self.shared.linear_sizes) {
            match self.shared.kernels.gdn {
                Some(kernels) => self
                    .shared
                    .call(kernels.zero, calls::zero(region.0, bytes / 4))?,
                // A plugged-in part's recurrent state on a checkpoint with no Gated DeltaNet
                // layer: the scan image's zero is not loaded, so the driver clears it.
                None => self
                    .shared
                    .cuda
                    .memset_d8_async(*region, 0, *bytes)
                    .map_err(|error| {
                        BackendError::new(DEVICE, format!("the recurrent state: {error}"))
                    })?,
            }
        }
        Ok(())
    }
}

/// The final norm and the LM head of `count` residual rows from row `first`, leaving their
/// logits in the scratch row after row.
fn head_logits(
    shared: &Shared,
    b: &Buffers,
    first: usize,
    count: usize,
) -> Result<(), BackendError> {
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
    shared.project_bf16(&shared.head, b.normed.0, b.logits.0, count, None)
}

/// The logits of `count` residual rows of the last walk from row `first`, row after row: the
/// final norm and the head on the row GEMV.
fn head(
    shared: &Shared,
    b: &Buffers,
    first: usize,
    count: usize,
) -> Result<Vec<f32>, BackendError> {
    head_logits(shared, b, first, count)?;
    let mut bytes = vec![0u8; count * shared.vocab * 4];
    shared
        .cuda
        .copy_dtoh(&mut bytes, &b.logits)
        .map_err(|error| BackendError::new(DEVICE, format!("the logits: {error}")))?;
    Ok(floats(&bytes))
}

/// The chosen token of each of `count` residual rows from row `first`, on the device: the final
/// norm and head as [head], then the vocabulary-wide argmax (`q4e_argmax_slices`) writes one
/// u32 a row. Only the picks cross to the host, not the logits.
///
/// The kernel is the host [sparkflash_ops::sampler::argmax] for a finite row: the higher value
/// wins and a tie takes the lower index. A NaN never wins. (A row whose first logit is NaN is
/// the kernel's documented exception: it answers 0.) With every logit finite, a greedy request
/// therefore gets the host sampler's token.
fn head_choices(
    shared: &Shared,
    b: &Buffers,
    first: usize,
    count: usize,
) -> Result<Vec<u32>, BackendError> {
    head_logits(shared, b, first, count)?;
    let rows = u32::try_from(count).map_err(|_| {
        BackendError::new(DEVICE, format!("{count} head rows do not fit one launch"))
    })?;
    shared
        .cuda
        .launch_async(
            shared.kernels.argmax.kernel,
            (rows, 1, 1),
            (ARGMAX_BLOCK, 1, 1),
            &[
                b.logits.0,
                b.picks.0,
                1,
                u64::from(rows),
                shared.vocab as u64,
            ],
        )
        .map_err(|error| BackendError::new(DEVICE, format!("q4e_argmax_slices: {error}")))?;
    let mut bytes = vec![0u8; count * 4];
    shared
        .cuda
        .copy_dtoh(&mut bytes, &b.picks)
        .map_err(|error| BackendError::new(DEVICE, format!("the picks: {error}")))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

/// One fused step: a row a member, each its own token, and each member's next.
fn step(
    shared: &Shared,
    scratch: &Scratch,
    members: &mut [(&mut Sequence, u32)],
    trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<Vec<StepOutput>, BackendError> {
    let tokens: Vec<u32> = members.iter().map(|(_, token)| *token).collect();
    let mut segments: Vec<(&mut Sequence, &[u32])> = members
        .iter_mut()
        .zip(tokens.chunks(1))
        .map(|((sequence, _), token)| (&mut **sequence, token))
        .collect();
    walk_rows(shared, scratch, &mut segments, PartBody::Token, None, trace)?;
    let logits = head(shared, &scratch.buffers, 0, segments.len())?;
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

/// One sequence with its own scratch, outside the serving engine: the parity gate's session,
/// which records each layer's residual.
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
            sequence: PlanSeq::new(DenseLane::open(shared)?),
            trace: None,
        })
    }

    /// Starts (or stops) recording each layer's residual after it, every row.
    pub fn trace(&mut self, on: bool) {
        self.trace = on.then(|| vec![Vec::new(); self.sequence.lane.shared.layers.len()]);
    }

    /// The recorded residuals, a `[positions][hidden]` row-major run a layer, and a fresh record.
    #[must_use]
    pub fn take_trace(&mut self) -> Vec<Vec<f32>> {
        let layers = self.sequence.lane.shared.layers.len();
        self.trace
            .as_mut()
            .map(|trace| std::mem::replace(trace, vec![Vec::new(); layers]))
            .unwrap_or_default()
    }

    /// Runs `tokens` from the session's position, the walk's chunk rows a pass (256 by default),
    /// and returns the last row's logits. The sampler does not see them.
    ///
    /// # Errors
    /// A token outside the vocabulary, a sequence past the served context, or a device failure.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, BackendError> {
        let shared = Arc::clone(&self.sequence.lane.shared);
        let mut last = 0;
        for chunk in tokens.chunks(shared.chunk) {
            walk_rows(
                &*shared,
                &self.scratch,
                &mut [(&mut self.sequence, chunk)],
                PartBody::Rows,
                shared.policy,
                self.trace.as_mut(),
            )?;
            last = chunk.len();
        }
        if last == 0 {
            return Err(BackendError::new(DEVICE, "the step has no tokens"));
        }
        head(&shared, &self.scratch.buffers, last - 1, 1)
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
        if tokens.is_empty() {
            return Err(BackendError::new(DEVICE, "the prompt has no tokens"));
        }
        // A session outside the engine starts every prompt over: the caches go back to position
        // zero and the recurrent state to the empty sequence's.
        self.sequence.rewind(0)?;
        self.sequence.sampler.reset();
        let shared = Arc::clone(&self.sequence.lane.shared);
        let Self {
            sequence,
            scratch,
            trace,
        } = self;
        let mut last = 0;
        for chunk in tokens.chunks(shared.chunk) {
            walk_rows(
                &*shared,
                &*scratch,
                &mut [(&mut *sequence, chunk)],
                PartBody::Rows,
                shared.policy,
                trace.as_mut(),
            )?;
            last = chunk.len();
        }
        let logits = head(&shared, &scratch.buffers, last - 1, 1)?;
        sequence.sampler.record_prompt(tokens);
        let token = sequence.sampler.choose(&logits, &sequence.sampling);
        Ok(StepOutput {
            next_token: Some(token),
            logits,
            draft: Vec::new(),
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let shared = Arc::clone(&self.sequence.lane.shared);
        let mut steps = step(
            &shared,
            &self.scratch,
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

#[cfg(test)]
mod tests {
    use super::{
        Bound, CHUNK_ROWS, DenseWalk, TOKEN_DECODE, chunk_rows, profile_policy, run_slice,
        typed_bf16, typed_fp8, typed_nvfp4,
    };
    use sparkflash_kernels::cuda_driver::Deviceptr;
    use sparkflash_kernels::images::Image;
    use sparkflash_model::backend::Sampling;
    use sparkflash_model::ledger::State;
    use sparkflash_model::profile::ModelProfile;
    use sparkflash_ops::dflash::Part as Run;
    use sparkflash_ops::formats::{Bf16, F32 as F32Format, FormatKind, Scales};
    use sparkflash_ops::plan::Precision;
    use sparkflash_ops::projection::WidePolicy;
    use sparkflash_ops::sampler::Sampler;
    use sparkflash_ops::widen::{FormatLedger, LedgerRow, WidenLog, widen};
    use sparkflash_runtime::Walk;
    use sparkflash_weights::{DType, TensorSpec};

    fn spec(name: &str, dtype: DType, shape: &[usize]) -> TensorSpec {
        TensorSpec {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    /// Every symbol `Kernels::load` resolves, with its image and precision. A symbol that drifts
    /// out of the catalogue then fails here, on the CPU, rather than at a device boot.
    const LOADED_SYMBOLS: &[(Image, &str, Precision)] = &[
        (Image::Qwen4Exp, "q4e_add", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_sum_slots", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_deinterleave", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_embed", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_gemv_bf16", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_pack_bf16", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_unpack_bf16", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_gemv_packed", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_gemv4_packed", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_rmsnorm", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_rope", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_swiglu", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_swiglu_clamped", Precision::Bf16),
        (
            Image::Qwen4Exp,
            "q4e_gemv_tiled_e4m3_tile",
            Precision::Fp8E4M3,
        ),
        (Image::Qwen4Exp, "q4e_zero", Precision::Bf16),
        (Image::Qwen4Exp, "q4e_argmax_slices", Precision::Bf16),
        (Image::Qwen4Attn, "q4e_attn_full", Precision::Bf16),
        (Image::Qwen4Attn, "q4e_attn_full_combine", Precision::Bf16),
        (Image::Qwen4Attn, "q4e_cache_bf16", Precision::Bf16),
        (Image::Qwen4Prefill, "q4p_dense_fp8", Precision::Fp8E4M3),
        (Image::Qwen4Moe, "q4e_route", Precision::Bf16),
        (Image::Qwen4Moe, "q4e_moe_combine", Precision::Bf16),
        (Image::Qwen4Moe, "q4e_expert_gate_up_bf16", Precision::Bf16),
        (
            Image::Qwen4Moe,
            "q4e_expert_gate_up_bf16_clamped",
            Precision::Bf16,
        ),
        (Image::Qwen4Moe, "q4e_expert_down_bf16", Precision::Bf16),
        (
            Image::Qwen4Moe,
            "q4e_expert_gate_up_fp8",
            Precision::Fp8E4M3,
        ),
        (
            Image::Qwen4Moe,
            "q4e_expert_gate_up_fp8_clamped",
            Precision::Fp8E4M3,
        ),
        (Image::Qwen4Moe, "q4e_expert_down_fp8", Precision::Fp8E4M3),
        (Image::Qwen4Moe, "q4e_expert_gate_up", Precision::NvFp4),
        (Image::Qwen4Moe, "q4e_expert_down", Precision::NvFp4),
        (Image::Dory, "dory_mma_wide", Precision::Bf16),
        (Image::MoePlan, "moe_plan", Precision::Bf16),
        (Image::Qwen4Scan, "q4p_gdn_conv", Precision::Bf16),
        (Image::Qwen4Scan, "q4p_gdn_conv_state", Precision::Bf16),
        (Image::Qwen4Scan, "q4p_gdn_prep", Precision::Bf16),
        (Image::Qwen4Scan, "q4p_gdn_scan", Precision::Bf16),
        (Image::Qwen4Scan, "q4p_gdn_gate", Precision::Bf16),
    ];

    #[test]
    fn every_kernel_the_arm_loads_is_in_the_catalogue() {
        for (image, symbol, precision) in LOADED_SYMBOLS {
            Bound::carried(*image, symbol, *precision).unwrap_or_else(|error| {
                panic!("{image:?} {symbol} at {}: {error}", precision.name())
            });
        }
    }

    /// A bound handle names a symbol the image exports and the catalogue carries at the
    /// handle's precision; a launch site never looks a kernel up by string.
    #[test]
    fn a_bound_handle_names_a_catalogued_symbol_and_precision() {
        Bound::carried(Image::Qwen4Exp, "q4e_gemv_bf16", Precision::Bf16)
            .expect("the catalogue carries q4e_gemv_bf16 at bf16");
        let error = Bound::carried(Image::Qwen4Exp, "q4e_gemv_bf16", Precision::Fp8E4M3)
            .expect_err("q4e_gemv_bf16 is a bf16 body, not an FP8 one");
        assert!(error.contains("accepting e4m3"), "{error}");
        let error = Bound::carried(Image::Qwen4Exp, "q4e_not_a_kernel", Precision::Bf16)
            .expect_err("the image does not export it");
        assert!(error.contains("does not export"), "{error}");
    }

    /// The load path's two checks are [sparkflash_ops::formats::upload]'s: a tensor of another
    /// stored dtype, or an FP8 tile whose inverse-scale plane is not one entry a 128 x 128 tile,
    /// is refused before the matrix is built.
    #[test]
    fn the_load_path_checks_the_stored_dtype_and_the_scale_count() {
        let bf16 = spec("layers.0.mlp.up_proj.weight", DType::BF16, &[4, 256]);
        let typed = typed_bf16(&bf16, 4, 256, Deviceptr(0x1000), 512, Scales::none())
            .expect("a BF16 projection");
        assert_eq!(typed.format(), FormatKind::Bf16);
        assert_eq!(typed.bytes_per_token(), 4 * 512);

        // The same logical matrix stored as U8 codes is not a BF16 Mat.
        let quantized = spec("layers.0.mlp.up_proj.weight", DType::U8, &[4, 128]);
        let error = typed_bf16(&quantized, 4, 256, Deviceptr(0x1000), 128, Scales::none())
            .expect_err("a U8 tensor is not a BF16 Mat");
        assert!(error.contains("bf16"), "{error}");
        assert!(error.contains("U8"), "{error}");

        let fp8 = spec("layers.0.mlp.up_proj.weight", DType::F8E4M3, &[4, 256]);
        let short = typed_fp8(
            &fp8,
            4,
            256,
            Deviceptr(0x2000),
            256,
            Deviceptr(0x3000),
            1,
            DType::BF16,
        )
        .expect_err("a 4 x 256 FP8 matrix has two 128 x 128 tiles");
        assert!(short.contains("needs 2"), "{short}");
        assert!(short.contains("has 1"), "{short}");
        let typed = typed_fp8(
            &fp8,
            4,
            256,
            Deviceptr(0x2000),
            256,
            Deviceptr(0x3000),
            2,
            DType::BF16,
        )
        .expect("two inverse scales");
        assert_eq!(typed.format(), FormatKind::Fp8E4m3);
        assert_eq!(typed.bytes_per_token(), 4 * 256);
    }

    /// An NVFP4 expert weight is a packed U8 container read at its logical shape, with one
    /// E4M3 scale a 16-value group and the per-tensor f32 the body multiplies in.
    #[test]
    fn the_nvfp4_expert_load_checks_the_packing_and_the_scale_groups() {
        let weight = spec(
            "layers.0.mlp.experts.0.gate_proj.weight",
            DType::U8,
            &[64, 128],
        );
        let typed = typed_nvfp4(
            &weight,
            64,
            256,
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            64 * 16,
            0.25,
        )
        .expect("a 64 x 256 NVFP4 matrix");
        assert_eq!(typed.format(), FormatKind::NvFp4);
        assert_eq!(typed.rows(), 64);
        assert_eq!(typed.columns(), 256);
        assert_eq!(typed.bytes_per_token(), 64 * 128);
        assert!(matches!(
            typed.scales(),
            sparkflash_ops::formats::Scales::NvFp4 { global: 0.25, .. }
        ));

        // One scale a group is 16 an expert row, not one a row.
        let error = typed_nvfp4(
            &weight,
            64,
            256,
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            64,
            0.25,
        )
        .expect_err("the scale plane is short");
        assert!(error.contains("needs 1024"), "{error}");

        // A BF16 container is not the packed E2M1 form.
        let unpacked = spec(
            "layers.0.mlp.experts.0.gate_proj.weight",
            DType::BF16,
            &[64, 256],
        );
        let error = typed_nvfp4(
            &unpacked,
            64,
            256,
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            1024,
            0.25,
        )
        .expect_err("a BF16 container is not the packed E2M1 form");
        assert!(error.contains("nvfp4"), "{error}");
    }

    /// The expert runs select rows or whole 16-column groups, and a run that splits a group is
    /// refused rather than read at another group's scale.
    #[test]
    fn the_nvfp4_runs_slice_whole_scale_groups() {
        let rows = 4;
        let columns = 64;
        let codes: Vec<u8> = (0..rows * columns / 2).map(|at| at as u8).collect();
        let (out_rows, out_columns, selected) =
            run_slice(&codes, (rows, columns), &[Run::Rows(1, 2)], 2).expect("two rows");
        assert_eq!((out_rows, out_columns), (2, 64));
        assert_eq!(selected, codes[32..96]);

        let (out_rows, out_columns, selected) =
            run_slice(&codes, (rows, columns), &[Run::Columns(16, 32)], 2).expect("32 columns");
        assert_eq!((out_rows, out_columns), (4, 32));
        assert_eq!(selected.len(), 4 * 16);

        let scales: Vec<u8> = (0..rows * columns / 16).map(|at| at as u8).collect();
        let error = run_slice(&scales, (rows, columns), &[Run::Columns(8, 16)], 16)
            .expect_err("eight columns split a 16-value group");
        assert!(error.contains("splits a 16-value column group"), "{error}");
    }

    /// Boot refuses a part read at a format its checkpoint does not store unless the load path
    /// recorded the widening; the router bias is the one this arm takes.
    #[test]
    fn boot_refuses_a_widening_the_load_path_did_not_record() {
        let rows = vec![LedgerRow::new(
            "layers.0.ffn.router.bias",
            FormatKind::Bf16,
            FormatKind::F32,
            32,
        )];
        let error = FormatLedger::boot(rows.clone(), &WidenLog::new())
            .expect_err("an unrecorded widen is refused");
        assert!(error.to_string().contains("layers.0.ffn.router.bias"));
        assert!(error.to_string().contains("bf16"));
        assert!(error.to_string().contains("f32"));

        let mut log = WidenLog::new();
        widen::<Bf16, F32Format>(
            "layers.0.ffn.router.bias",
            "every bf16 value is an f32",
            &mut log,
        )
        .expect("the recorded widening");
        let ledger = FormatLedger::boot(rows, &log).expect("a recorded widening boots");
        assert!(ledger.to_string().contains("widened bf16 -> f32"));
    }

    /// The token the device head's kernel chooses, transcribed from `q4e_argmax_slices`: the
    /// higher value wins, a tie takes the lower index, a NaN never wins, and a row whose first
    /// logit is NaN answers 0.
    fn device_argmax(row: &[f32]) -> u32 {
        let mut value = f32::NEG_INFINITY;
        let mut index = u32::MAX;
        for (at, &logit) in row.iter().enumerate() {
            if logit > value || (logit == value && (at as u32) < index) {
                value = logit;
                index = at as u32;
            }
        }
        if row.first().is_some_and(|value| value.is_nan()) {
            0
        } else {
            index
        }
    }

    /// The toy rows: a clear maximum, a tie (the lower index), a negative row and a one-entry
    /// row. A greedy sampler and the device kernel must agree on every one.
    fn toy_rows() -> Vec<Vec<f32>> {
        vec![
            vec![0.5, -1.0, 2.0, 0.0],
            vec![3.0, 3.0, 1.0, -2.0],
            vec![-3.0, -2.0, -1.0, -4.0],
            vec![7.0, -1.0, 0.0, 2.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ]
    }

    #[test]
    fn the_device_choice_is_the_host_samplers_token_on_the_toy_rows() {
        for row in toy_rows() {
            let mut host = Sampler::new();
            let expected = host.choose(&row, &Sampling::default());
            assert_eq!(
                device_argmax(&row),
                expected,
                "the device kernel's argmax on {row:?}"
            );
        }
    }

    /// The bit delta the policy change can make, on a device: a nine-row BF16 projection through
    /// `dory_mma_wide` (the policy's route at nine rows) against the row GEMV the same rows took
    /// before (one row a launch). The test reports the largest absolute and ULP difference rather
    /// than pinning the tile's arithmetic; it is the first dense number a device gate can compare
    /// with the host reference.
    #[test]
    fn the_dense_tile_and_the_row_gemv_move_a_nine_row_projection_in_the_last_bits() {
        let Ok(cuda) = sparkflash_kernels::cuda_driver::Cuda::load() else {
            eprintln!("skipping the dense tile bit delta: no CUDA device");
            return;
        };
        if sparkflash_kernels::images::dory_cubin().is_empty()
            || sparkflash_kernels::images::qwen4_exp_cubin().is_empty()
        {
            eprintln!("skipping the dense tile bit delta: this build has no Dory/Qwen cubin");
            return;
        }
        let (columns, inner, rows) = (256usize, 256usize, 9usize);
        // A deterministic BF16 weight and f32 activation, in the shape the wide tile covers.
        let mut state = 0x5eed_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let weight: Vec<u8> = (0..columns * inner)
            .flat_map(|_| ((next().to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input: Vec<u8> = (0..rows * inner)
            .flat_map(|_| next().to_le_bytes())
            .collect();
        let gemv = Bound::new(
            &cuda,
            Image::Qwen4Exp,
            sparkflash_kernels::images::qwen4_exp_cubin(),
            "q4e_gemv_bf16",
            Precision::Bf16,
        )
        .expect("the Qwen4-Exp image carries q4e_gemv_bf16")
        .kernel;
        let wide = Bound::shared(
            &cuda,
            Image::Dory,
            sparkflash_kernels::images::dory_cubin(),
            "dory_mma_wide",
            Precision::Bf16,
            sparkflash_ops::projection::WIDE_SHARED,
        )
        .expect("the Dory image carries dory_mma_wide")
        .kernel;
        let mut total = 0;
        let w = cuda.alloc(weight.len(), &mut total).expect("the weights");
        let x = cuda.alloc(input.len(), &mut total).expect("the input");
        let row_out = cuda
            .alloc(rows * columns * 4, &mut total)
            .expect("the GEMV output");
        let tile_out = cuda
            .alloc(rows * columns * 4, &mut total)
            .expect("the tile output");
        cuda.copy_htod(&w, &weight).expect("the weights upload");
        cuda.copy_htod(&x, &input).expect("the input upload");
        for row in 0..rows {
            cuda.launch(
                gemv,
                (columns.div_ceil(8) as u32, 1, 1),
                (256, 1, 1),
                &[
                    w.0,
                    x.0 + (row * inner * 4) as u64,
                    row_out.0 + (row * columns * 4) as u64,
                    columns as u64,
                    inner as u64,
                    1,
                ],
            )
            .expect("the row GEMV launches");
        }
        cuda.launch(
            wide,
            (
                columns.div_ceil(sparkflash_ops::projection::WIDE_COLUMNS) as u32,
                rows.div_ceil(sparkflash_ops::projection::WIDE_ROWS) as u32,
                1,
            ),
            (sparkflash_ops::projection::WIDE_THREADS, 1, 1),
            &[
                w.0,
                x.0,
                tile_out.0,
                columns as u64,
                inner as u64,
                rows as u64,
            ],
        )
        .expect("the wide tile launches");
        let mut left = vec![0u8; rows * columns * 4];
        let mut right = vec![0u8; rows * columns * 4];
        cuda.copy_dtoh(&mut left, &row_out)
            .expect("the GEMV output");
        cuda.copy_dtoh(&mut right, &tile_out)
            .expect("the tile output");
        cuda.free_buffers(&[&w, &x, &row_out, &tile_out]);
        let left = super::floats(&left);
        let right = super::floats(&right);
        assert!(left.iter().chain(&right).all(|value| value.is_finite()));
        let mut worst = 0.0f32;
        let mut worst_ulp = 0u32;
        let mut differing = 0usize;
        for (a, b) in left.iter().zip(&right) {
            worst = worst.max((a - b).abs());
            worst_ulp = worst_ulp.max(a.to_bits().abs_diff(b.to_bits()));
            differing += usize::from(a.to_bits() != b.to_bits());
        }
        eprintln!(
            "dense tile vs row GEMV at {rows}x{columns}x{inner}: {differing} of {} values \
             differ, largest |delta| {worst:e}, largest ULP {worst_ulp}",
            rows * columns
        );
        assert!(worst.is_finite());
    }

    /// The same comparison against the real kernel, where this build and host have a CUDA device
    /// and the Qwen4-Exp image. Without either it says so and passes, as the arm's other device
    /// gates do.
    #[test]
    fn the_cuda_head_choice_is_the_host_samplers_token_on_the_toy_rows() {
        let Ok(cuda) = sparkflash_kernels::cuda_driver::Cuda::load() else {
            eprintln!("skipping the device head choice: no CUDA device");
            return;
        };
        let image = sparkflash_kernels::images::qwen4_exp_cubin();
        if image.is_empty() {
            eprintln!("skipping the device head choice: this build has no Qwen4-Exp cubin");
            return;
        }
        let argmax = Bound::new(
            &cuda,
            Image::Qwen4Exp,
            image,
            "q4e_argmax_slices",
            Precision::Bf16,
        )
        .expect("the Qwen4-Exp image carries q4e_argmax_slices")
        .kernel;
        let rows = toy_rows();
        let vocab = rows[0].len();
        assert!(rows.iter().all(|row| row.len() == vocab));
        let flat: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut total = 0;
        let logits = cuda.alloc(flat.len(), &mut total).expect("the toy logits");
        cuda.copy_htod(&logits, &flat)
            .expect("the toy logits upload");
        let picks = cuda
            .alloc(rows.len() * 4, &mut total)
            .expect("the toy picks");
        cuda.launch(
            argmax,
            (rows.len() as u32, 1, 1),
            (1024, 1, 1),
            &[logits.0, picks.0, 1, rows.len() as u64, vocab as u64],
        )
        .expect("the argmax launches");
        let mut bytes = vec![0u8; rows.len() * 4];
        cuda.copy_dtoh(&mut bytes, &picks)
            .expect("the toy picks download");
        cuda.free_buffers(&[&logits, &picks]);
        for (row, word) in rows.iter().zip(bytes.as_chunks::<4>().0) {
            let chosen = u32::from_le_bytes(*word);
            let mut host = Sampler::new();
            assert_eq!(
                chosen,
                host.choose(row, &Sampling::default()),
                "the kernel's argmax on {row:?}"
            );
        }
    }

    /// The walk's token-decode capability row and the profile refusal are the same constant, so
    /// the ledger cannot state one thing while the switch is refused for another.
    #[test]
    fn the_decode_row_and_the_profile_refusal_are_one_source() {
        assert_eq!(
            <DenseWalk as Walk>::CAPABILITIES.compute.token,
            TOKEN_DECODE,
            "the walk states the token row the profile check reads"
        );
        let mut profile = ModelProfile::default();
        profile.decode.tensor_cores = Some(true);
        let State::Off(why) = TOKEN_DECODE else {
            panic!("dense declares a tensor-core token body; delete the refusal with the row");
        };
        let error = profile_policy(&profile).expect_err("the switch is refused");
        assert!(
            error.contains(why),
            "the refusal names the same reason the ledger row does: {error}"
        );
    }

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

    /// The default is dense's legacy 256 through ChunkPolicy; a documented smaller switch moves
    /// it, and an undocumented or larger one is refused rather than silently rounded.
    #[test]
    fn the_chunk_comes_from_the_policy_and_defaults_to_256() {
        assert_eq!(chunk_rows(&ModelProfile::default(), 32_768), Ok(CHUNK_ROWS));
        for size in [1usize, 2, 4, 8, 64, CHUNK_ROWS] {
            let mut profile = ModelProfile::default();
            profile.prefill.chunk = Some(size);
            assert_eq!(
                chunk_rows(&profile, 32_768),
                Ok(size),
                "a documented {size}-row chunk is honoured"
            );
        }
        for size in [0usize, 3, 128, 300, 512, 1_024, 4_096] {
            let mut profile = ModelProfile::default();
            profile.prefill.chunk = Some(size);
            let error = chunk_rows(&profile, 32_768).expect_err("the size is refused");
            assert!(error.contains("prefill.chunk"), "{error}");
        }
    }
}
