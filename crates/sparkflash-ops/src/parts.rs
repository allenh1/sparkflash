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

//! The per-target kernel catalogue: which image and symbols implement a [Part].
//!
//! [sparkflash_model::parts::built] says which parts a *build* claims; this module says what
//! that claim means. [CATALOGUE] names one entry per (part, target, image) triple, so a part
//! served by several model images lists one entry for each. A part is carried for a target when
//! the catalogue has at least one entry for it ([carried]), and [image_available] narrows that
//! to the entries whose image this build actually embeds.
//!
//! # Per-symbol precision and phase
//!
//! An image is not one arithmetic and one pass. The token image exports a bf16 GEMM beside an
//! FP8 quantiser, and the decode image exports a decode body beside a prefill projection. Each
//! [crate::plan::KernelSymbol] therefore carries its own precision mask and its own
//! [crate::plan::KernelPhase]. The resolver selects a symbol only when that symbol's own mask
//! contains the requested precision and its own phase serves the requested pass; an entry whose
//! first symbol is a bf16 GEMM cannot satisfy an NVFP4 request merely because a later symbol in
//! the same image is a quantiser.
//!
//! # Canonical entries
//!
//! [PartKernel::canonical] marks the implementation a target brings up first. Exactly one entry
//! per (part, target) is canonical, and the resolver scores it ahead of the others; it is also
//! listed first. When DeepSeek implements a part, its entry is canonical: DeepSeek's kernels are
//! the proven implementation and a Qwen or GLM entry must never silently become the default for
//! a shared part.
//!
//! # Images
//!
//! Every entry names an existing [Image] variant. [Image::Deepseek] covers the shared PTX set
//! that the DeepSeek arm loads at run time; it is a real kernel set but reports no embedded
//! bytes ([Image::available] is false), unlike [Image::Qwen4Exp], [Image::Qwen4Attn],
//! [Image::Qwen4Moe], [Image::Qwen4Prefill], [Image::Qwen4Scan], [Image::Dflash2],
//! [Image::GlmPrefill], [Image::GlmAttention], [Image::GlmMoe] and [Image::GlmDecode], which
//! this build embeds when nvcc is present.
//!
//! Only [Target::Sm121] has entries today: the kernels under kernels/ are named *_sm121, so no
//! part is claimed for sm100 or sm120 until its kernels are compiled for them.

use sparkflash_kernels::images::Image;
use sparkflash_model::Architecture;
use sparkflash_model::architecture::WeightFormat;
use sparkflash_model::vision_geometry::{
    VisionActivation, VisionDownsample, VisionNorm, VisionPosition,
};

use crate::gemv::E4M3_TILE;
use crate::plan::{ComputeClass, KernelPhase, KernelSymbol, Precision, RowBand, ShapeConstraint};
use sparkflash_model::Part;
use sparkflash_model::Target;
use sparkflash_model::parts::PartSet;

/// One implementation of a [Part]: the target it runs on, the image that carries it and the
/// kernel symbols the image exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartKernel {
    /// The model-side part this entry implements.
    pub part: Part,
    /// The device target the entry is compiled for.
    pub target: Target,
    /// The shared image that carries the symbols.
    pub image: Image,
    /// The image's kernel symbols that implement the part, each with its own precision and pass.
    pub symbols: &'static [KernelSymbol],
    /// Whether this is the implementation a new target brings up first.
    ///
    /// Exactly one entry per (part, target) is canonical. A DeepSeek entry is always the
    /// canonical one for a part DeepSeek implements.
    pub canonical: bool,
    /// The shapes this implementation accepts. [ShapeConstraint::ANY] is an unproven rule.
    pub shape: ShapeConstraint,
}

/// The (key, value) head-width templates the shared GDN scan is compiled for.
///
/// `qwen4_scan_sm121` instantiates the templated body at exactly these three pairs, and the Qwen
/// device selects `q4p_gdn_scan_64_64` / `q4p_gdn_scan_128_256` for the two that are not the
/// released 128/128 body (D98). The token `q4e_gdn_recur` body bounds the same geometries (value
/// head to 1024, key head to 256), so it carries the same list.
const SCAN_HEAD_PAIRS: &[(usize, usize)] = &[(64, 64), (128, 128), (128, 256)];

/// The head-width template the shared KDA scan serves.
///
/// The image now exports `q4p_kda_scan_64_64` and `q4p_kda_scan_128_256` beside `q4p_kda_scan`,
/// but the GLM-5.3 host launches only `q4p_kda_scan` with a 128-wide key head and `GDN_VALUE` = 128
/// value head, so the catalogue claims only the template the host can actually select.
const KDA_SCAN_HEAD_PAIRS: &[(usize, usize)] = &[(128, 128)];

/// The vision patch side the shared tower serves.
///
/// `opt_vision_sm121.cu` exports both `opt_vision_patchify` (14) and `opt_vision_patchify_16`, and
/// `VisionShape::of_with_patch` derives either grid. The catalogue still claims only 14 because
/// the Qwen tower's orchestration and weight loading live in its own arm (D84): the shared
/// catalogue entry would name a tower this tree does not run at patch 16, so the Qwen3.8 release
/// that asks for one is refused rather than booted against a 14-patch tower.
const VISION_PATCH_SIZES: &[usize] = &[14];

/// The vision patch side Inkling-2-Small's tower is released at.
///
/// Inkling runs its own patchify, rope and scale-add on the shared `opt_vision_*` gemm and
/// attention bodies, so its entry must claim its own patch: without this constraint the vision
/// part falls to `ShapeConstraint::ANY` and a Qwen 16-patch plan would resolve against
/// Inkling's entry, which is the refusal D84 relies on.
const INKLING_VISION_PATCH_SIZES: &[usize] = &[20];

/// The width the shared vision image compiles its norms, attention and MLP at.
const VISION_DIM: usize = 1024;
/// The heads and per-head width the shared vision attention compiles.
const VISION_HEADS: usize = 16;
const VISION_HEAD_DIM: usize = 64;
/// The SwiGLU width `opt_vision_swiglu` compiles.
const VISION_MLP: usize = 2816;
/// The blend row `opt_vision_span_rows` compiles.
const VISION_ROW: usize = 5120;
/// The epsilon `opt_vision_rmsnorm` hardcodes.
const VISION_EPSILON_BITS: u64 = 1e-6f64.to_bits();

/// The routed-expert `(hidden, experts, top-k)` shapes the Qwen MoE images are published at.
///
/// The token and grouped expert bodies are runtime-shaped (up to Q4E_MAX_TOPK = 64 active of
/// Q4P_MAX_EXPERTS = 512, on 128-column E4M3 tiles or 16-column NVFP4 groups), but the resolver
/// picks an image by shape alone. An open rule would claim GLM-5.3's 4096-wide FP8 experts here,
/// ahead of the GLM images that are their proven body, so the list is the releases the Qwen arm
/// serves: Qwen3.8 (2560, 512, 10) and Qwen3.5/3.6-35B-A3B (2048, 256, 8).
const QWEN_ROUTED_SHAPES: &[(usize, usize, usize)] = &[(2560, 512, 10), (2048, 256, 8)];

/// The grouped-query attention group sizes this tree has a proven body for.
///
/// q4e_attn is generic, and q4e_attn_group / q4e_attn_group6 specialise four and six. The
/// published list is the set the released arms resolve at.
const GQA_GROUPS: &[usize] = &[1, 2, 3, 4, 6, 8, 12];

/// The Inkling-2-Small text geometry every inkling_* entry is compiled for.
///
/// The released checkpoint is 4,608-wide with 48 query heads of 128 and 8 KV heads (group 6),
/// a kernel-4 short convolution, a 2,304-wide routed expert and a 1,024-token sliding window.
/// The window is per layer (1024 on a sliding layer, 0 on a full one), so it is documented
/// here rather than pinned as an equality: a full layer legitimately carries no window.
const INKLING_HIDDEN: usize = 4608;
const INKLING_HEAD_DIM: usize = 128;
const INKLING_INTERMEDIATE: usize = 2304;

/// The widest short-convolution kernel the body accepts.
///
/// `kernels/inkling_conv_sm121.cu:87` states it: "The kernel width is 4 in the released
/// checkpoint; the entries accept 1..INKLING_MAX_KERNEL so the same body can be brought up
/// against a different release." The body rejects `kernel < 1 || kernel > 8` at `:134`, so the
/// catalogue must not pin the released width and refuse the widths the same body takes.
const INKLING_MAX_KERNEL: usize = 8;

/// The first row a chained block residual takes.
///
/// The chunk walk chains a residual only when it has more than `RESIDUAL_SPLIT_PAIRS` (8) rows
/// (`crates/sparkflash-model-inkling2/src/device.rs`), so the catalogue's band starts one past
/// that. Below it the walk launches the one- or two-launch form.
const INKLING_CHAIN_FIRST_ROWS: usize = 9;

/// The shape rule of one catalogue entry.
///
/// This is the model-independent part of kernel selection: a kernel that only works on whole
/// 16-wide heads, on one hidden width, or on one intermediate width says so here and the
/// resolver refuses the checkpoint before a weight is read. A field named by a rule is a proof
/// obligation: a geometry that leaves it at zero is refused, not skipped.
const fn shape_for(part: Part, image: Image) -> ShapeConstraint {
    match (part, image) {
        (Part::Embedding, Image::Deepseek) => {
            ShapeConstraint::ANY.hidden_equals(5120).vocab_multiple(128)
        }
        // q4e_embed reads the table row at the runtime hidden width and copies it to f32
        // (`kernels/qwen4_exp_sm121.cu:318-331`); it has no vocabulary alignment. Inkling's
        // 201,024-row table is not a multiple of the Qwen release's 128, and the same body
        // embeds it.
        (Part::Embedding, Image::Qwen4Exp) => ShapeConstraint::ANY,
        (Part::RmsNorm, Image::Deepseek) => {
            ShapeConstraint::ANY.hidden_equals(5120).hidden_multiple(2)
        }
        (Part::RmsNorm, Image::Qwen4Exp) => ShapeConstraint::ANY.hidden_multiple(2),
        (Part::LmHead, Image::Deepseek) => {
            ShapeConstraint::ANY.hidden_equals(5120).vocab_multiple(128)
        }
        // q4e_argmax_slices and q4e_argmax_prob_slices take the rank-local vocabulary width
        // as a runtime argument (`kernels/qwen4_exp_sm121.cu:336-397`); neither rounds it to
        // 128. The head screen is not the matrix projection, which is Part::DenseGemm.
        (Part::LmHead, Image::Qwen4Exp) => ShapeConstraint::ANY,
        // CSA2 / grouped-output LoRA MLA: the DeepSeek latent has output groups and a rope
        // segment. The absorbed NoPE latent has neither, and is served by the GLM images.
        (Part::MlaAttention, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .head_dim_multiple(16)
            .groups_between(1, usize::MAX)
            .positions(0, 1 << 21),
        (Part::MlaAttention, Image::GlmAttention | Image::GlmDecode) => ShapeConstraint::ANY
            .hidden_equals(4096)
            .head_dim_multiple(16)
            .groups_between(0, 0)
            .positions(0, 1 << 21),
        (Part::Csa2Compression, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .min_ratio(1)
            .positions(0, 1 << 21),
        (Part::SparseIndexer, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .head_dim_equals(128)
            .top_k_equals(512)
            .max_heads(32),
        (Part::SparseIndexer, Image::Qwen4Attn) => ShapeConstraint::ANY
            .hidden_equals(2560)
            .head_dim_equals(128)
            .max_heads(8),
        (Part::SparseIndexer, Image::GlmDecode) => ShapeConstraint::ANY
            .hidden_equals(4096)
            .head_dim_equals(128)
            .max_heads(32),
        (Part::GqaAttention, Image::Qwen4Attn) => ShapeConstraint::ANY
            .head_dim_multiple(16)
            .gqa_groups(GQA_GROUPS),
        // GDN instantiates the templated scan at all three pairs; KDA's host selects only the
        // 128/128 template it was released against.
        (Part::GdnScan, Image::Qwen4Scan | Image::Qwen4Exp) => ShapeConstraint::ANY
            .head_pairs(SCAN_HEAD_PAIRS)
            .window_between(2, 9),
        (Part::KdaScan, Image::Qwen4Scan | Image::Qwen4Exp) => ShapeConstraint::ANY
            .head_pairs(KDA_SCAN_HEAD_PAIRS)
            .window_between(2, 9),
        (Part::DenseMlp, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .intermediate_multiple(16),
        // GLM's bulk-prefill MLP is compiled for the 4096-wide release, like its DenseGemm
        // entry. A 2560-wide Qwen MLP must not match it and launch glm_prefill_* on a body
        // compiled for another width.
        (Part::DenseMlp, Image::GlmPrefill) => ShapeConstraint::ANY
            .hidden_equals(4096)
            .intermediate_multiple(16),
        (Part::DenseMlp, Image::Qwen4Exp) => ShapeConstraint::ANY
            .hidden_equals(2560)
            .intermediate_multiple(16),
        // Dory is a 2,560-wide normalized-GPT: its custom kernels are compiled for that hidden
        // width, a 256-wide head with four query heads a KV head, and a 7,168-wide MLP.
        (Part::RmsNorm, Image::Dory) => ShapeConstraint::ANY.hidden_equals(2560),
        (Part::LmHead, Image::Dory) => ShapeConstraint::ANY.hidden_equals(2560).vocab_multiple(128),
        (Part::GqaAttention, Image::Dory) => ShapeConstraint::ANY
            .hidden_equals(2560)
            .head_dim_equals(256)
            .gqa_groups(&[4]),
        (Part::DenseMlp, Image::Dory) => ShapeConstraint::ANY
            .hidden_equals(2560)
            .intermediate_equals(7168),
        (Part::DenseMlp, Image::Qwen4Prefill) => ShapeConstraint::ANY
            .hidden_equals(2560)
            .intermediate_multiple(16),
        (Part::DenseMlp, Image::GlmDecode) => ShapeConstraint::ANY
            .hidden_equals(4096)
            .intermediate_multiple(16),
        // DeepSeek's routed experts are the MXFP4 (NVFP4 in the ops vocabulary) shape. The
        // b12x kernel is compiled for 576 real columns, the rank-local width at TP4; TP1 and
        // TP2 are refused rather than launched on padded-wrong arithmetic.
        (Part::Moe, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .intermediate_equals(576)
            .experts_equals(384)
            .top_k_equals(6)
            .positions(0, 1 << 21),
        (Part::Moe, Image::Qwen4Moe | Image::Qwen4Prefill) => ShapeConstraint::ANY
            .routed_shapes(QWEN_ROUTED_SHAPES)
            .intermediate_multiple(16)
            .max_experts(512)
            .max_top_k(64),
        (Part::Moe, Image::GlmMoe | Image::GlmDecode) => ShapeConstraint::ANY
            .hidden_equals(4096)
            .intermediate_multiple(16)
            .experts_equals(288)
            .top_k_equals(8)
            .max_experts(512)
            .max_top_k(64),
        // Inkling's routed experts are a scalar dequantise over runtime columns, so the
        // intermediate width is the only geometry the body fixes; the expert count and the
        // selected slots are the rank's and the router's.
        // The router is the only body here with a bound of its own: it refuses more than
        // INKLING_ROUTER_HELD * blockDim experts (32 * 128 = 4,096) and more than 64 selected
        // experts (`kernels/inkling_moe_sm121.cu:793-808`). The expert bodies take the rank's
        // owned run as a runtime argument (`:291-322`), so the expert count is not an equality:
        // TP4 holds 96 of the 384 experts, and the same body runs either count. The columns are
        // a multiple of 16 (NVFP4's block scale, `:289`).
        (Part::Moe, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(4608)
            .intermediate_multiple(16)
            .max_experts(4096)
            .max_top_k(64),
        // The shared dense projection. The b12x and q4p bodies are generic over the hidden and
        // intermediate widths, so this entry is not pinned to one model's 5120.
        (Part::DenseGemm, Image::Deepseek) => ShapeConstraint::ANY,
        (Part::DenseGemm, Image::Qwen4Prefill) => ShapeConstraint::ANY.hidden_equals(2560),
        (Part::DenseGemm, Image::GlmPrefill) => ShapeConstraint::ANY.hidden_equals(4096),
        // q4e_gemv_bf16 / q4e_gemm_bf16 take rows and columns as runtime arguments: the GEMV
        // walks `row < rows` over a runtime `columns` (`kernels/qwen4_exp_sm121.cu:407-456`)
        // and the GEMM guards only `columns <= 0` and `batch > QG_TOKENS`
        // (`:1220-1229`). Inkling runs this entry at hidden 4,608, not Qwen's 2,560.
        (Part::DenseGemm, Image::Qwen4Exp) => ShapeConstraint::ANY,
        // Dory's prompt projections are dory_mma_wide / dory_mma_wide4: the body takes cols,
        // inner and tokens at run time (`kernels/dory_sm121.cu:62-67`), and the four-matrix
        // form requires each output width to be a multiple of 32 (`:69-81`), a fact the
        // geometry has no field for. It is the same body the Dory MLP entry names.
        (Part::DenseGemm, Image::Dory) => ShapeConstraint::ANY,
        (Part::DenseGemm, Image::GlmDecode) => ShapeConstraint::ANY.hidden_equals(4096),
        (Part::Activation, Image::Deepseek) => ShapeConstraint::ANY.hidden_equals(5120),
        (Part::Activation, Image::Qwen4Exp) => ShapeConstraint::ANY.hidden_equals(2560),
        (Part::Activation, Image::Qwen4Prefill) => ShapeConstraint::ANY.hidden_equals(2560),
        (Part::Activation, Image::GlmMoe | Image::GlmDecode) => {
            ShapeConstraint::ANY.hidden_equals(4096)
        }
        (Part::Rope, Image::Deepseek) => ShapeConstraint::ANY.hidden_equals(5120),
        (Part::Rope, Image::Qwen4Exp) => ShapeConstraint::ANY.hidden_equals(2560),
        // Inkling's four new arithmetic families. The relative-attention entry serves both the
        // 1,024-token sliding layers and the eight full layers, so the window is a documented
        // per-layer fact rather than an equality the full layers would fail.
        // The group count is a runtime argument: `kernels/inkling_attn_sm121.cu:20-21` says
        // so, and the body maps each query head to `head / (heads / kv_heads)` without a
        // compile-time group. Pinning 6 refused the plan's own RelativeAttention geometry,
        // which names heads and head width but leaves kv_heads unset.
        (Part::RelativeAttention, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(INKLING_HIDDEN)
            .head_dim_equals(INKLING_HEAD_DIM)
            .positions(0, 1 << 21),
        (Part::ShortConv, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(INKLING_HIDDEN)
            .window_between(1, INKLING_MAX_KERNEL),
        (Part::BlockAttentionResidual, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(INKLING_HIDDEN)
            .streams_between(1, 16),
        (Part::SoftcappedActivation, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(INKLING_HIDDEN)
            .intermediate_equals(INKLING_INTERMEDIATE),
        (Part::MoeCombine, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(INKLING_HIDDEN)
            .max_top_k(64),
        (Part::Quantize, _) => ShapeConstraint::ANY,
        // Only the GLM decode image exports a LayerNorm body; RMSNorm is a different part and
        // the resolver must never fall back to it.
        (Part::LayerNorm, Image::GlmDecode) => ShapeConstraint::ANY.hidden_equals(4096),
        (Part::MoeRouter, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .max_experts(512)
            .max_top_k(64),
        (Part::MoeRouter, Image::Qwen4Moe) => ShapeConstraint::ANY
            .hidden_equals(2560)
            .max_experts(512)
            .max_top_k(64),
        (Part::MoeRouter, Image::GlmDecode) => ShapeConstraint::ANY
            .hidden_equals(4096)
            .max_experts(512)
            .max_top_k(64),
        (Part::MoeCombine, Image::Deepseek) => {
            ShapeConstraint::ANY.hidden_equals(5120).max_top_k(64)
        }
        // The shared token combine is architecture-independent: it sums a runtime top-k over any
        // hidden width the token image serves. This is the entry the duplicate (Moe, Qwen4Moe)
        // pair was trying to provide.
        (Part::MoeCombine, Image::Qwen4Moe) => {
            ShapeConstraint::ANY.hidden_multiple(128).max_top_k(64)
        }
        (Part::GroupedMoeGemm, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .intermediate_equals(576)
            .experts_equals(384)
            .top_k_equals(6),
        (Part::GroupedMoeGemm, Image::Qwen4Moe | Image::Qwen4Prefill) => ShapeConstraint::ANY
            .routed_shapes(QWEN_ROUTED_SHAPES)
            .intermediate_multiple(16)
            .max_experts(512)
            .max_top_k(64),
        (Part::GroupedMoeGemm, Image::GlmMoe | Image::GlmDecode) => ShapeConstraint::ANY
            .hidden_equals(4096)
            .intermediate_multiple(16)
            .experts_equals(288)
            .top_k_equals(8)
            .max_experts(512)
            .max_top_k(64),
        (Part::MoePlan, Image::MoePlan) => ShapeConstraint::ANY,
        // The one-shot collective is the TP4 kernel the slot ABI pins.
        (Part::Collective, Image::Rocenante) => ShapeConstraint::ANY.world_equals(4),
        (Part::Collective, Image::Qwen4Exp) => ShapeConstraint::ANY.world_min(2),
        (Part::HyperConnection, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .streams_between(4, 4),
        (Part::HyperConnection, Image::Qwen4Exp) => ShapeConstraint::ANY
            .hidden_multiple(256)
            .streams_between(2, 5),
        (Part::ManifoldHyperConnection, Image::GlmDecode) => ShapeConstraint::ANY
            .hidden_multiple(256)
            .streams_between(2, 5),
        (Part::Engram, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .head_dim_equals(256),
        (Part::Ple, Image::Qwen4Exp) => ShapeConstraint::ANY.hidden_multiple(128),
        (Part::Mtp, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(5120)
            .experts_equals(128)
            .top_k_equals(3),
        (Part::Mtp, Image::Dflash2) => ShapeConstraint::ANY.hidden_multiple(256),
        // The shared vision tower's components, each claiming only the shape its kernel source
        // really compiles. The DeepSeek image carries the 14/16 patchify, 1024-wide norm and
        // attention, 2816-wide SwiGLU, 3x3 unfold and 5120-wide aligner; Inkling's image carries
        // its own 20-patch patchify, LayerNorm, axial rope, layer scale and 4096 GELU MLP.
        (Part::VisionPatchEmbed, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .patch_sizes(VISION_PATCH_SIZES)
            .in_channels_equals(3)
            .temporal_patch_equals(1),
        (Part::VisionPatchEmbed, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .patch_sizes(INKLING_VISION_PATCH_SIZES)
            .in_channels_equals(3)
            .temporal_patch_equals(1),
        (Part::VisionPosition, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .head_dim_equals(VISION_HEAD_DIM)
            .vision_position(VisionPosition::Rotary.code()),
        (Part::VisionPosition, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .head_dim_equals(VISION_HEAD_DIM)
            .vision_position(VisionPosition::RotaryAndGrid.code()),
        (Part::VisionNorm, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .vision_norm(VisionNorm::Rms.code())
            .norm_epsilon_bits(VISION_EPSILON_BITS),
        (Part::VisionNorm, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .vision_norm(VisionNorm::Layer.code()),
        (Part::VisionAttention, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .heads_multiple(VISION_HEADS)
            .head_dim_equals(VISION_HEAD_DIM)
            .attention_bias(true)
            .qk_norm(false)
            .vision_position(VisionPosition::Rotary.code()),
        (Part::VisionAttention, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .heads_multiple(VISION_HEADS)
            .head_dim_equals(VISION_HEAD_DIM)
            .attention_bias(true)
            .qk_norm(false)
            .vision_position(VisionPosition::RotaryAndGrid.code()),
        (Part::VisionMlp, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .intermediate_equals(VISION_MLP)
            .vision_activation(VisionActivation::SwiGlu.code()),
        (Part::VisionMlp, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .intermediate_equals(4096)
            .vision_activation(VisionActivation::GeluErf.code()),
        (Part::VisionDownsample, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .spatial_merge_equals(3)
            .vision_downsample(VisionDownsample::Unfold(3).code()),
        (Part::VisionMerger, Image::Deepseek) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .out_hidden_equals(VISION_ROW)
            .merger_intermediate_equals(VISION_ROW)
            .vision_activation(VisionActivation::GeluErf.code()),
        (Part::VisionMerger, Image::Inkling) => ShapeConstraint::ANY
            .hidden_equals(VISION_DIM)
            .out_hidden_equals(4608)
            .merger_intermediate_equals(4096)
            .vision_activation(VisionActivation::GeluErf.code()),
        _ => ShapeConstraint::ANY,
    }
}

/// A symbol accepting exactly these precisions in both passes.
///
/// The unqualified helpers describe elementwise bodies (norms, rope, scans, quantisers, combines).
/// A GEMM body names its class and row band explicitly with the helpers below, so the catalogue
/// assertion can find it.
const fn sym(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(name, precisions, ComputeClass::Elementwise, RowBand::ALL)
}

/// One DeepSeek image symbol in the exact catalogue entry the baseline manifest names.
const fn deepseek_sym(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(name, precisions, ComputeClass::TensorCore, RowBand::ALL)
}

/// A bulk-prefill symbol accepting exactly these precisions.
const fn pre(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::prefill(name, precisions, ComputeClass::Elementwise, RowBand::ALL)
}

/// A token/decode symbol accepting exactly these precisions.
const fn dec(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::decode(name, precisions, ComputeClass::Elementwise, RowBand::ALL)
}

/// One instantiation of the chained block residual.
///
/// `inkling_block_residual_chain_q<queries>_n<next>` is one template in
/// `kernels/inkling_conv_sm121.cu`; the six entries differ only in the compile-time query and
/// next-call counts. Each takes f32 summaries, queries and carry (the signature is all
/// `const float*`), so it is the elementwise residual body the arm plans and loads at BF16, as
/// it loads the two-launch form's statistics and mix. The walk launches it only over more than
/// eight rows, which is the bulk-prefill pass.
const fn chain(name: &'static str) -> KernelSymbol {
    KernelSymbol::prefill(
        name,
        &[Precision::Bf16],
        ComputeClass::Elementwise,
        RowBand::from(INKLING_CHAIN_FIRST_ROWS),
    )
}

/// A row GEMV's band: at most eight token rows a launch.
const ROW_BAND: RowBand = RowBand::at_most(crate::gemv::GEMV_MAX_BATCH);

/// The first row a bulk tile takes on GLM, Inkling and Dory (`MAX_ROWS`/`BATCH_ROWS` = 8).
const TILE_FIRST: usize = crate::gemv::GEMV_MAX_BATCH + 1;

/// The last row Qwen's few-row tensor tile takes (`GEMM_MIN_BATCH` = 64).
const ROWS_TILE_LAST: usize = crate::projection::QWEN_GEMM_MIN_BATCH - 1;

/// A tensor-core body over every row count, in both passes.
const fn tc(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(name, precisions, ComputeClass::TensorCore, RowBand::ALL)
}

/// A tensor-core body from `TILE_FIRST` rows up, in the bulk-prefill pass.
const fn tc_tile(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::prefill(
        name,
        precisions,
        ComputeClass::TensorCore,
        RowBand::from(TILE_FIRST),
    )
}

/// A tensor-core body from Qwen's grouped threshold up, in the bulk-prefill pass.
const fn tc_chunk(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::prefill(
        name,
        precisions,
        ComputeClass::TensorCore,
        RowBand::from(crate::projection::QWEN_GEMM_MIN_BATCH),
    )
}

/// A tensor-core body over Qwen's few-row tile, in both passes.
const fn tc_rows(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(
        name,
        precisions,
        ComputeClass::TensorCore,
        RowBand::at_most(ROWS_TILE_LAST),
    )
}

/// A tensor-core body from `TILE_FIRST` rows up, in both passes.
const fn tc_wide(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(
        name,
        precisions,
        ComputeClass::TensorCore,
        RowBand::from(TILE_FIRST),
    )
}

/// Dory's four-matrix wide body: each output width is a whole 32-column MMA step.
const fn tc_wide4(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    tc_wide(name, precisions).with_shape(ShapeConstraint::ANY.output_multiple(32))
}

/// A tensor-core token body over the one-to-eight row band, in the token pass.
///
/// A decode step, a speculative verify and a short chunk all run here; a row is the prompt
/// tile's, so a continued turn resumes on the bits a cold prefill of it leaves.
const fn tc_token(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::decode(name, precisions, ComputeClass::TensorCore, ROW_BAND)
}

/// A row body over every row count, in both passes (a non-MMA tile).
const fn row_all(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(name, precisions, ComputeClass::Row, RowBand::ALL)
}

/// A row body over at most eight token rows, in both passes.
const fn row(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(name, precisions, ComputeClass::Row, ROW_BAND)
}

/// A non-MMA few-row tile over Qwen's rows band, in both passes.
const fn row_rows(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(
        name,
        precisions,
        ComputeClass::Row,
        RowBand::at_most(ROWS_TILE_LAST),
    )
}

/// A row body over at most eight token rows, in the token pass.
const fn row_dec(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::decode(name, precisions, ComputeClass::Row, ROW_BAND)
}

/// A catalogue entry on the one target the kernels are built for today.
const fn sm121(
    part: Part,
    image: Image,
    symbols: &'static [KernelSymbol],
    canonical: bool,
) -> PartKernel {
    PartKernel {
        part,
        target: Target::Sm121,
        image,
        symbols,
        canonical,
        shape: shape_for(part, image),
    }
}

/// Every (part, target, image) implementation this tree catalogues.
///
/// Entries are grouped by [Part] in declaration order. Within a part the canonical entry comes
/// first, and a DeepSeek entry is canonical for every part DeepSeek implements. Exactly one entry
/// per (part, target, image) is allowed; an operation that needs a different shape rule gets its
/// own [Part] identity rather than a second entry for the same image.
pub const CATALOGUE: &[PartKernel] = &[
    sm121(
        Part::Embedding,
        Image::Deepseek,
        &[
            deepseek_sym("opt_rows_embed_own", &[Precision::Bf16]),
            deepseek_sym("opt_rows_embed_stream", &[Precision::Bf16]),
            deepseek_sym("opt_rows_embed_table", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::Embedding,
        Image::Qwen4Exp,
        &[sym("q4e_embed", &[Precision::Bf16, Precision::Fp8E4M3])],
        false,
    ),
    sm121(
        Part::RmsNorm,
        Image::Deepseek,
        &[
            deepseek_sym("rmsnorm_sm121", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("opt_rmsnorm_bf16", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("opt_k_norm_rows", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("k_norm_f32_to_bf16", &[Precision::Bf16, Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::RmsNorm,
        Image::Qwen4Exp,
        &[
            sym("q4e_rmsnorm", &[Precision::Bf16, Precision::F32]),
            sym("q4e_rmsnorm_rows", &[Precision::Bf16, Precision::F32]),
        ],
        false,
    ),
    // q4e_layer_norm is launched over a prefill chunk's rows as well as a token's, so it serves
    // both passes even though its image is the decode/token cubin.
    sm121(
        Part::LayerNorm,
        Image::GlmDecode,
        &[sym("q4e_layer_norm", &[Precision::Bf16, Precision::F32])],
        true,
    ),
    sm121(
        Part::LmHead,
        Image::Deepseek,
        &[
            deepseek_sym("cert_lm_head_sm121", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_lm_head", &[Precision::Fp8E4M3, Precision::MxFp8]),
            deepseek_sym("opt_rows_lm_head", &[Precision::NvFp4]),
            deepseek_sym("opt_rows_argmax_fold", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_rows_argmax_pick",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_rows_store_u32", &[Precision::NvFp4]),
            deepseek_sym("opt_head_screen_build", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_head_screen_exact",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_head_screen_mma", &[Precision::NvFp4]),
            deepseek_sym("opt_head_screen_rows", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_rows_topc", &[Precision::Fp8E4M3, Precision::MxFp8]),
        ],
        true,
    ),
    sm121(
        Part::LmHead,
        Image::Qwen4Exp,
        &[
            row("q4e_argmax_slices", &[Precision::Bf16, Precision::Fp8E4M3]),
            row(
                "q4e_argmax_prob_slices",
                &[Precision::Bf16, Precision::Fp8E4M3],
            ),
        ],
        false,
    ),
    sm121(
        Part::MlaAttention,
        Image::Deepseek,
        &[
            deepseek_sym(
                "opt_bulk_attn",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym(
                "opt_flash_mla_csa2_seq",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym(
                "kv_append_bf16",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym(
                "opt_rows_kv",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym(
                "opt_rows_attn",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym(
                "opt_rows_attn_scores",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym(
                "opt_rows_attn_softmax",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym(
                "opt_rows_attn_values",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym(
                "flash_mla_csa2_seq_sm121",
                &[Precision::Bf16, Precision::Fp8E4M3, Precision::MxFp8],
            ),
        ],
        true,
    ),
    sm121(
        Part::MlaAttention,
        Image::SPARSE_MLA,
        &[
            pre(
                "glm_prefill_mla_attend",
                &[Precision::Fp8E4M3, Precision::Bf16],
            ),
            pre(
                "glm_prefill_mla_attend_bf16a",
                &[Precision::Fp8E4M3, Precision::Bf16],
            ),
            pre("glm_prefill_heads_bf16_mma", &[Precision::Bf16]),
            pre("glm_prefill_heads_fp8b128_mma", &[Precision::Fp8E4M3]),
            // The chunk indexer's scores over f32 queries and pooled keys.
            pre("glm_prefill_index_scores", &[Precision::F32]),
        ],
        false,
    ),
    sm121(
        Part::MlaAttention,
        Image::DECODE,
        &[
            dec("q4e_glm_mla_cache", &[Precision::Bf16, Precision::Fp8E4M3]),
            dec("q4e_glm_mla_cache_fp8", &[Precision::Fp8E4M3]),
            dec("q4e_glm_mla_absorb", &[Precision::Bf16, Precision::Fp8E4M3]),
            dec("q4e_glm_mla_attend", &[Precision::Bf16, Precision::Fp8E4M3]),
            dec("q4e_glm_mla_scores", &[Precision::Bf16, Precision::Fp8E4M3]),
            dec(
                "q4e_glm_mla_softmax",
                &[Precision::Bf16, Precision::Fp8E4M3],
            ),
            dec("q4e_glm_mla_mix", &[Precision::Bf16, Precision::Fp8E4M3]),
            dec(
                "q4e_glm_mla_project",
                &[Precision::Bf16, Precision::Fp8E4M3],
            ),
        ],
        false,
    ),
    sm121(
        Part::Csa2Compression,
        Image::Deepseek,
        &[
            deepseek_sym(
                "compressed_append_sm121",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "compressed_selected_fill_sm121",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "pool_group_f32",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_pool_rows",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_group_pool",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_group_keys",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_group_append",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16w",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16w_f32o",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_mma_hostw_f32w_f32o",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16a_bf16w",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16a_bf16w_f32o",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16a_f32w_f32o",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_hostw_bf16w",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_hostw_bf16w_f32o",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_hostw_f32w_f32o",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_hostw_bf16w_f32o_ring",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_hostw_f32w_f32o_ring",
                &[Precision::Bf16, Precision::MxFp8, Precision::Fp8E4M3],
            ),
        ],
        true,
    ),
    sm121(
        Part::SparseIndexer,
        Image::Deepseek,
        &[
            deepseek_sym(
                "opt_ref_select_fast",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_mma_index_scores",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_select_scores",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_select_scores_rows",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_select_scores_seg",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_select_scores_segs",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_select_rows",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_select_rows_bf16",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_select_rows_tab",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16w",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16w_f32o",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_mma_hostw_f32w_f32o",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16a_bf16w",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16a_bf16w_f32o",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_mma_hostw_bf16a_f32w_f32o",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_rows_hostw_bf16w",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_rows_hostw_bf16w_f32o",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_rows_hostw_f32w_f32o",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "scale_index_weights_bf16_to_f32",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "ref_select_sm121",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
        ],
        true,
    ),
    sm121(
        Part::SparseIndexer,
        Image::ATTENTION,
        &[
            sym("q4e_qsa_pool", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4e_qsa_scores", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4e_qsa_pick", &[Precision::Bf16, Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::SparseIndexer,
        Image::DECODE,
        // The GLM device launches all three bodies over a prefill chunk's rows as well as a
        // token's; a decode-only index_scores made the resolver refuse a launch the arm makes.
        &[
            sym("q4e_glm_pool", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym(
                "q4e_glm_index_scores",
                &[Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym("q4e_glm_pick", &[Precision::Bf16, Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::GqaAttention,
        Image::ATTENTION,
        &[
            sym("q4e_attn", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4e_attn_group", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4e_attn_group6", &[Precision::Bf16, Precision::Fp8E4M3]),
            // Every causal position, split over the context: a layer with no sparse indexer.
            sym("q4e_attn_full", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym(
                "q4e_attn_full_combine",
                &[Precision::Bf16, Precision::Fp8E4M3],
            ),
            // The BF16 cache a boundary row appends to its lane, and the row-at-a-time form
            // the sparse indexer writes its keys through.
            sym("q4e_cache_bf16", &[Precision::Bf16]),
            sym("q4e_cache_rows_bf16", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::GdnScan,
        Image::SCAN,
        &[
            pre(
                "q4p_gdn_conv",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            pre(
                "q4p_gdn_conv_state",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            pre(
                "q4p_gdn_prep",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            pre(
                "q4p_gdn_scan",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            // The two non-128/128 bodies are the same template instantiated at the other
            // key/value head pairs in [SCAN_HEAD_PAIRS]. Naming them is what lets ScanPlan
            // resolve the symbol the launch actually runs instead of always naming 128/128.
            pre(
                "q4p_gdn_scan_64_64",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            pre(
                "q4p_gdn_scan_128_256",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            pre(
                "q4p_gdn_gate",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
        ],
        true,
    ),
    sm121(
        Part::GdnScan,
        Image::TOKEN,
        &[
            dec(
                "q4e_gdn_conv",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            dec(
                "q4e_gdn_recur",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            // The same recurrence split over a head's value columns, for a KDA head whose
            // whole state does not fit one block.
            dec("q4e_gdn_recur_columns", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::KdaScan,
        Image::SCAN,
        &[
            pre(
                "q4p_kda_prep",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            pre(
                "q4p_kda_scan",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            pre(
                "q4p_gdn_gate",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
        ],
        true,
    ),
    sm121(
        Part::KdaScan,
        Image::TOKEN,
        &[dec(
            "q4e_kda_decay",
            &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
        )],
        false,
    ),
    sm121(
        Part::DenseMlp,
        Image::Deepseek,
        &[
            deepseek_sym(
                "swiglu_mxfp8_quant_sm121",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_mma_bf16a_bf16o",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_mma_f32a_f32o", &[Precision::NvFp4]),
            deepseek_sym("opt_mma_fp8_bf16o", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_mma_bf16a_f32o",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("shared_swiglu_f32", &[Precision::NvFp4]),
        ],
        true,
    ),
    sm121(
        Part::DenseMlp,
        Image::BLOCK_PREFILL,
        &[
            tc_tile("glm_prefill_bf16_mma", &[Precision::Bf16]),
            tc_tile("glm_prefill_bf16_mma_bf16a", &[Precision::Bf16]),
            tc_tile("glm_prefill_bf16_wide", &[Precision::Bf16]),
            tc_tile("glm_prefill_bf16_wide_bf16a", &[Precision::Bf16]),
            tc_tile("glm_prefill_fp8b128_mma", &[Precision::Fp8E4M3]),
            tc_tile("glm_prefill_fp8b128_mma_bf16a", &[Precision::Fp8E4M3]),
            tc_tile("glm_prefill_q8_mma", &[Precision::Fp8E4M3]),
            pre("glm_quantise_rows", &[Precision::F32]),
        ],
        false,
    ),
    // The bulk-prefill MLP body precedes the token image's few-row GEMM: a prefill request
    // should pick q4p_dense_mma, while a decode request still falls through to the token entry
    // because every q4p symbol is prefill-only.
    sm121(
        Part::DenseMlp,
        Image::PREFILL,
        &[
            tc_chunk("q4p_dense_mma", &[Precision::Bf16, Precision::Fp8E4M3]),
            tc_chunk("q4p_dense_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_epilogue", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::DenseMlp,
        Image::TOKEN,
        &[
            row_all("q4e_gemm_bf16", &[Precision::Bf16, Precision::F32]),
            row("q4e_shared_gate_up", &[Precision::Bf16, Precision::F32]),
            // The same body is the Activation part's; it runs over the projected chunk in
            // prefill and over a token's row in decode, so both entries describe one phase.
            sym("q4e_swiglu", &[Precision::Bf16, Precision::F32]),
            dec("q4e_quantize_e4m3_rows", &[Precision::Bf16]),
            row_dec("q4e_gemv_tiled_bf16", &[Precision::Bf16]),
            row_dec("q4e_gemv_tiled_e4m3", &[Precision::Fp8E4M3]),
            // The Qwen3.5/3.6 FP8 release's block-scaled dense projections (BF16 tile scales).
            row_dec("q4e_gemv_tiled_e4m3_tile", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::DenseMlp,
        Image::DECODE,
        &[
            // The FP8 tile's few-row form, over GLM's f32 128 x 128 block scales: one to eight
            // rows, a row the tile's bits.
            tc_token("glm_decode_fp8b128_rows", &[Precision::Fp8E4M3]),
            row_dec(
                "q4e_gemv_tiled_bf16_parts",
                &[Precision::Bf16, Precision::F32],
            ),
            // The same GEMV over the packed form's planes (sparkflash_ops::packed).
            row_dec(
                "q4e_gemv_tiled_packed_parts",
                &[Precision::Bf16, Precision::F32],
            ),
            row_dec("q4e_gemv_tiled_e4m3_block", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    // The shared dense projection, distinct from the MLP part so its b12x body is not pinned to
    // one model's hidden width.
    sm121(
        Part::DenseGemm,
        Image::Deepseek,
        &[
            deepseek_sym(
                "b12x_bf16a_mxfp8w_wo_a_sm121",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "b12x_mxfp8_gemv_m1_sm121",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("b12x_hostw_gemv_m1_bf16w_sm121", &[Precision::NvFp4]),
            deepseek_sym(
                "b12x_hostw_gemv_m1_f32w_f32o_sm121",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_mma_bf16a_bf16o",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_mma_f32a_f32o", &[Precision::NvFp4]),
            deepseek_sym("opt_mma_fp8_bf16o", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "b12x_hostw_gemv_m1_bf16w_f32o_sm121",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("b12x_mxfp8_gemv_m1_f32a_f32o_sm121", &[Precision::NvFp4]),
            deepseek_sym(
                "b12x_mxfp8_gemv_m1_nf_f32a_sm121",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "b12x_mxfp8_gemv_m1_nf_sm121",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_mma_f32a_bf16o", &[Precision::NvFp4]),
            deepseek_sym(
                "opt_gemv_bf16a_bf16o_t1",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_gemv_bf16a_bf16o_t2",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_gemv_bf16a_bf16o_t4", &[Precision::NvFp4]),
            deepseek_sym(
                "opt_gemv_bf16a_bf16o_t8",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_gemv_f32a_bf16o_t1",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_gemv_f32a_bf16o_t2", &[Precision::NvFp4]),
            deepseek_sym("opt_gemv_f32a_bf16o_t4", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_gemv_f32a_bf16o_t8",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_gemv_f32a_f32o_t1", &[Precision::NvFp4]),
            deepseek_sym("opt_gemv_f32a_f32o_t2", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_gemv_f32a_f32o_t4",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_gemv_f32a_f32o_t8", &[Precision::NvFp4]),
            deepseek_sym("opt_gemv_fp8a_bf16o_t1", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_gemv_fp8a_bf16o_t2",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_gemv_fp8a_bf16o_t4", &[Precision::NvFp4]),
            deepseek_sym("opt_gemv_fp8a_bf16o_t8", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_gemv_mma_bf16a_bf16o_m1",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_gemv_mma_bf16a_bf16o_m2", &[Precision::NvFp4]),
            deepseek_sym(
                "opt_gemv_mma_fp8a_bf16o_m1",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_gemv_rows_bf16a_bf16o_r8c1",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_gemv_rows_bf16a_bf16o_r8c2", &[Precision::NvFp4]),
            deepseek_sym(
                "opt_gemv_rows_f32a_f32o_r4c1",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_gemv_rows_f32a_f32o_r4c2",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_gemv_rows_fp8a_bf16o_r4c1", &[Precision::NvFp4]),
            deepseek_sym(
                "opt_gemv_rows_fp8a_bf16o_r8c1",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_gemv_rows_fp8a_bf16o_r8c2",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
        ],
        true,
    ),
    sm121(
        Part::DenseGemm,
        Image::PREFILL,
        &[
            tc_chunk("q4p_dense_mma", &[Precision::Bf16, Precision::Fp8E4M3]),
            tc_chunk("q4p_dense_fp8", &[Precision::Fp8E4M3]),
            // The E4M3 token body (Qwen's bf16 block scales) and the packed BF16 token body,
            // both on the tile's chain: one to eight rows, a row the tile's bits.
            tc_token("q4p_dense_fp8_rows", &[Precision::Fp8E4M3]),
            tc_token("q4p_dense_packed_rows", &[Precision::Bf16]),
            tc_rows("q4p_dense_rows", &[Precision::Bf16]),
            row_rows("q4p_gemm_rows", &[Precision::Bf16]),
            // The hyper connection's up projection and stream mix for a few rows.
            row_rows("q4p_hyper_up_rows", &[Precision::Bf16]),
            pre("q4p_split_bf16", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::DenseGemm,
        Image::BLOCK_PREFILL,
        &[
            tc_tile("glm_prefill_bf16_mma", &[Precision::Bf16]),
            tc_tile("glm_prefill_bf16_mma_bf16a", &[Precision::Bf16]),
            tc_tile("glm_prefill_bf16_wide", &[Precision::Bf16]),
            tc_tile("glm_prefill_bf16_wide_bf16a", &[Precision::Bf16]),
            tc_tile("glm_prefill_fp8b128_mma", &[Precision::Fp8E4M3]),
            tc_tile("glm_prefill_fp8b128_mma_bf16a", &[Precision::Fp8E4M3]),
            tc_tile("glm_prefill_q8_mma", &[Precision::Fp8E4M3]),
            // The exact router and indexer parts, and the skinny BF16 part of at most four
            // rows a block: f32 activations, BF16 weights, the prefill pass.
            KernelSymbol::prefill(
                "glm_prefill_exact_bf16",
                &[Precision::Bf16],
                ComputeClass::Row,
                RowBand::ALL,
            ),
            KernelSymbol::prefill(
                "glm_prefill_skinny_bf16_1",
                &[Precision::Bf16],
                ComputeClass::Row,
                RowBand::ALL,
            ),
            KernelSymbol::prefill(
                "glm_prefill_skinny_bf16_2",
                &[Precision::Bf16],
                ComputeClass::Row,
                RowBand::ALL,
            ),
            KernelSymbol::prefill(
                "glm_prefill_skinny_bf16_3",
                &[Precision::Bf16],
                ComputeClass::Row,
                RowBand::ALL,
            ),
            KernelSymbol::prefill(
                "glm_prefill_skinny_bf16_4",
                &[Precision::Bf16],
                ComputeClass::Row,
                RowBand::ALL,
            ),
        ],
        false,
    ),
    sm121(
        Part::DenseGemm,
        Image::TOKEN,
        &[
            row_all("q4e_gemm_bf16", &[Precision::Bf16, Precision::F32]),
            row_dec("q4e_gemv_bf16", &[Precision::Bf16]),
            row_dec("q4e_gemv4_bf16", &[Precision::Bf16]),
            row_dec("q4e_gemv_tiled_bf16", &[Precision::Bf16]),
            row_dec("q4e_gemv_tiled_e4m3", &[Precision::Fp8E4M3]),
            // The Qwen3.5/3.6 FP8 release's block-scaled dense projections (BF16 tile scales).
            row_dec("q4e_gemv_tiled_e4m3_tile", &[Precision::Fp8E4M3]),
            // The packed BF16 form's row walk and its load-time pack/unpack pair
            // (sparkflash_ops::packed): the dense arm reads the same bits from fewer bytes.
            row_dec("q4e_gemv_packed", &[Precision::Bf16]),
            row_dec("q4e_gemv4_packed", &[Precision::Bf16]),
            sym("q4e_pack_bf16", &[Precision::Bf16]),
            sym("q4e_unpack_bf16", &[Precision::Bf16]),
            // The K-split fold of the token GEMM: an elementwise reduction of f32 partials to
            // the BF16 epilogue, launched beside q4e_gemm_bf16.
            sym("q4e_gemm_reduce", &[Precision::Bf16, Precision::F32]),
        ],
        false,
    ),
    // Dory's prompt-chunk projections: one wide body for q, k, v, g, o, gate, up and down,
    // beside the shared token GEMV and few-row tile the arm also launches. The arm names these
    // launches Part::DenseGemm, so the catalogue carries that identity rather than leaving the
    // body reachable only under the MLP part.
    sm121(
        Part::DenseGemm,
        Image::Dory,
        &[
            tc_wide("dory_mma_wide", &[Precision::Bf16]),
            tc_wide4("dory_mma_wide4", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::DenseGemm,
        Image::DECODE,
        &[
            // The FP8 tile's few-row form, over GLM's f32 128 x 128 block scales: one to eight
            // rows, a row the tile's bits.
            tc_token("glm_decode_fp8b128_rows", &[Precision::Fp8E4M3]),
            row_dec(
                "q4e_gemv_tiled_bf16_parts",
                &[Precision::Bf16, Precision::F32],
            ),
            // The same GEMV over the packed form's planes (sparkflash_ops::packed).
            row_dec(
                "q4e_gemv_tiled_packed_parts",
                &[Precision::Bf16, Precision::F32],
            ),
            row_dec("q4e_gemv_tiled_e4m3_block", &[Precision::Fp8E4M3]),
            // SLM's L2 prefetch hint over the next kernels' weights: no arithmetic, and it
            // reads the same spans the projections do.
            dec("q4e_prefetch_l2", &[Precision::F32]),
        ],
        false,
    ),
    sm121(
        Part::Activation,
        Image::Deepseek,
        &[
            deepseek_sym(
                "swiglu_mxfp8_quant_sm121",
                &[Precision::Bf16, Precision::F32, Precision::MxFp8],
            ),
            deepseek_sym(
                "swiglu_mxfp8_quant_weighted_sm121",
                &[Precision::Bf16, Precision::F32, Precision::MxFp8],
            ),
            deepseek_sym(
                "shared_swiglu_f32",
                &[Precision::Bf16, Precision::F32, Precision::MxFp8],
            ),
            deepseek_sym(
                "opt_rows_shared_swiglu",
                &[Precision::Bf16, Precision::F32, Precision::MxFp8],
            ),
        ],
        true,
    ),
    sm121(
        Part::Activation,
        Image::TOKEN,
        &[
            sym("q4e_swiglu", &[Precision::Bf16, Precision::F32]),
            sym("q4e_swiglu_clamped", &[Precision::Bf16, Precision::F32]),
            // The residual add, the chunk's layout fan-out and the state zero the dense arm
            // launches beside its projections.
            sym("q4e_add", &[Precision::Bf16, Precision::F32]),
            sym("q4e_deinterleave", &[Precision::Bf16, Precision::F32]),
            sym("q4e_zero", &[Precision::Bf16, Precision::F32]),
            // A flat copy of an f32 state or widened-stream region.
            sym("q4e_copy_flat", &[Precision::F32]),
        ],
        false,
    ),
    sm121(
        Part::Activation,
        Image::PREFILL,
        &[
            pre("q4p_epilogue", &[Precision::Bf16]),
            // A Qwen FP8 chunk's MoE SwiGLU is the grouped path's activation, not a BF16
            // epilogue. Its inputs are the gate/up tile's f32 rows (q4p_moe_swiglu_fp8 takes
            // `const float*`), so the body's operand class is F32 even though the plan that
            // launches it is FP8: both precisions name the same kernel.
            pre("q4p_moe_swiglu_fp8", &[Precision::Fp8E4M3, Precision::F32]),
        ],
        false,
    ),
    sm121(
        Part::Activation,
        Image::BLOCK_ROUTED_EXPERTS,
        &[
            pre("glm_moe_swiglu", &[Precision::Bf16]),
            pre("glm_moe_swiglu_q8", &[Precision::Bf16, Precision::Fp8E4M3]),
        ],
        false,
    ),
    // The GLM device fuses this SwiGLU over a chunk's rows in prefill and over a token's row in
    // decode, from the same decode cubin.
    sm121(
        Part::Activation,
        Image::DECODE,
        &[sym("q4e_swiglu_limited", &[Precision::Bf16])],
        false,
    ),
    sm121(
        Part::Rope,
        Image::Deepseek,
        &[
            deepseek_sym("opt_rope_heads_bf16", &[Precision::Bf16]),
            deepseek_sym("rope_rotate_rows_bf16", &[Precision::Bf16]),
            deepseek_sym("rope_rotate_bf16", &[Precision::Bf16]),
            deepseek_sym("opt_rows_inv_rope", &[Precision::Bf16]),
            deepseek_sym("kv_norm_rope_bf16", &[Precision::Bf16]),
            deepseek_sym("opt_kv_norm_rope", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::Rope,
        Image::TOKEN,
        &[sym("q4e_rope", &[Precision::Bf16])],
        false,
    ),
    // Inkling-2-Small. The relative-attention pair is one body with a one-row (decode) and a
    // many-row (prefill) entry; the caches are BF16, which is the precision both name.
    sm121(
        Part::RelativeAttention,
        Image::Inkling,
        &[
            dec("inkling_attn_decode", &[Precision::Bf16]),
            pre("inkling_attn_prefill", &[Precision::Bf16]),
            // The split walks, the clustered window, the tensor-core body and the cache rows.
            sym("inkling_attn_key_scales", &[Precision::Bf16]),
            sym("inkling_attn_scores", &[Precision::Bf16]),
            sym("inkling_attn_mix", &[Precision::Bf16]),
            sym("inkling_attn_part", &[Precision::Bf16]),
            sym("inkling_attn_scored_part", &[Precision::Bf16]),
            sym("inkling_attn_merge", &[Precision::Bf16]),
            sym("inkling_attn_prepare", &[Precision::Bf16]),
            sym("inkling_attn_mma", &[Precision::Bf16]),
            sym("inkling_attn_window", &[Precision::Bf16]),
            sym("inkling_kv_store", &[Precision::Bf16]),
            sym("inkling_summary_push", &[Precision::Bf16]),
            sym("inkling_widen_f32", &[Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::ShortConv,
        Image::Inkling,
        &[
            sym("inkling_short_conv", &[Precision::Bf16, Precision::F32]),
            sym(
                "inkling_short_conv_decode",
                &[Precision::Bf16, Precision::F32],
            ),
            sym(
                "inkling_short_conv_dynamic",
                &[Precision::Bf16, Precision::F32],
            ),
            sym(
                "inkling_short_conv_dynamic_decode",
                &[Precision::Bf16, Precision::F32],
            ),
            // The convolution history's two moves between the decode and chunk layouts.
            sym("inkling_conv_history_copy", &[Precision::F32]),
            sym("inkling_conv_history_restore", &[Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::BlockAttentionResidual,
        Image::Inkling,
        &[
            sym("inkling_block_residual", &[Precision::Bf16, Precision::F32]),
            // The chained form: one template instantiated at each (query, next) pair the chunk
            // walk launches, one symbol a pair the way the GDN scan entry names its
            // q4p_gdn_scan_64_64 / _128_256 instantiations. The catalogue had a hole here for
            // the whole released arm: the walk loaded these six by name and the entry named
            // only the one- and two-launch bodies, so the typed lookup refused the boot.
            chain("inkling_block_residual_chain_q1_n0"),
            chain("inkling_block_residual_chain_q1_n1"),
            chain("inkling_block_residual_chain_q1_n3"),
            chain("inkling_block_residual_chain_q3_n0"),
            chain("inkling_block_residual_chain_q3_n1"),
            chain("inkling_block_residual_chain_q3_n3"),
            // The two-launch form, the plain residual add, the row-split join and the exact f32
            // widening.
            sym("inkling_block_residual_stats", &[Precision::Bf16]),
            sym("inkling_block_residual_mix", &[Precision::Bf16]),
            sym("inkling_residual_add", &[Precision::Bf16]),
            sym("inkling_join_pieces", &[Precision::F32]),
            sym("inkling_widen_f32", &[Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::SoftcappedActivation,
        Image::Inkling,
        &[
            sym(
                "inkling_softcapped_swiglu",
                &[Precision::Bf16, Precision::F32],
            ),
            sym("inkling_logit_softcap", &[Precision::Bf16, Precision::F32]),
            sym("inkling_band_shares", &[Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::Quantize,
        Image::Deepseek,
        &[
            deepseek_sym(
                "opt_bulk_fp4_pack",
                &[Precision::F32, Precision::Bf16, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "w8a8_prologue_e4m3",
                &[Precision::F32, Precision::Bf16, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "opt_rows_prologue_bf16",
                &[Precision::F32, Precision::Bf16, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "window_quant_fp8_bf16",
                &[Precision::F32, Precision::Bf16, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "fp4_round_trip_bf16_to_f32",
                &[Precision::F32, Precision::Bf16, Precision::Fp8E4M3],
            ),
            deepseek_sym(
                "w8a8_prologue_bf16_e4m3",
                &[Precision::F32, Precision::Bf16, Precision::Fp8E4M3],
            ),
        ],
        true,
    ),
    // Inkling's routed-expert activation quantiser: the same packed-E2M1/E4M3-scale storage
    // as the shared NVFP4 path, but with the released four-over-six block-scale choice. It sits
    // beside the canonical DeepSeek quantisers rather than replacing them.
    sm121(
        Part::Quantize,
        Image::Inkling,
        &[sym(
            "inkling_nvfp4_quantize_4_6",
            &[Precision::Bf16, Precision::NvFp4],
        )],
        false,
    ),
    sm121(
        Part::Quantize,
        Image::TOKEN,
        &[dec("q4e_quantize_e4m3_rows", &[Precision::Bf16])],
        false,
    ),
    sm121(
        Part::Quantize,
        Image::BLOCK_PREFILL,
        &[pre("glm_quantise_rows", &[Precision::F32])],
        false,
    ),
    sm121(
        Part::Moe,
        Image::Deepseek,
        &[
            deepseek_sym(
                "b12x_mxfp4_moe_stored_sm121",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_moe_gate_up_sm121",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_moe_gate_up_k3_sm121", &[Precision::NvFp4]),
            deepseek_sym(
                "opt_moe_down_combine_sm121",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_moe_down_combine_k3_sm121",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_mma_moe_gate_up", &[Precision::NvFp4]),
            deepseek_sym("opt_mma_moe_down", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "moe_expert_stage_sm121",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_moe_down_dedup_sm121", &[Precision::NvFp4]),
            deepseek_sym("routed_gate_up_sm121", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_moe_gate_up_dedup_sm121",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_moe_gate_up_dedup_k3_sm121", &[Precision::NvFp4]),
        ],
        true,
    ),
    // Inkling's routed experts read two weight formats the shared plans do not: ModelOpt NVFP4
    // with a per-expert scale, and MXFP8 with a ue8m0 block scale. Because both factors are
    // linear, an expert's scale is applied to its own output, so one body serves every expert.
    // Its router is a sigmoid schedule (score + bias, then route_scale), not the shared softmax.
    sm121(
        Part::Moe,
        Image::Inkling,
        &[
            row_all(
                "inkling_nvfp4_expert_matvec",
                &[Precision::NvFp4, Precision::F32],
            ),
            row_all(
                "inkling_mxfp8_expert_matvec",
                &[Precision::MxFp8, Precision::F32],
            ),
            sym("inkling_router_topk", &[Precision::F32]),
            // The band, pair and item walks over the same stored experts.
            row_all(
                "inkling_nvfp4_expert_band",
                &[Precision::NvFp4, Precision::F32],
            ),
            row_all(
                "inkling_mxfp8_expert_band",
                &[Precision::MxFp8, Precision::F32],
            ),
            row_all(
                "inkling_nvfp4_expert_pairs",
                &[Precision::NvFp4, Precision::F32],
            ),
            row_all(
                "inkling_mxfp8_expert_pairs",
                &[Precision::MxFp8, Precision::F32],
            ),
            row_all(
                "inkling_nvfp4_expert_items",
                &[Precision::NvFp4, Precision::F32],
            ),
            row_all(
                "inkling_mxfp8_expert_items",
                &[Precision::MxFp8, Precision::F32],
            ),
        ],
        false,
    ),
    sm121(
        Part::Moe,
        Image::ROUTED_EXPERTS,
        &[
            // These bodies decode packed E2M1 codes, so their operand class is NVFP4 alone.
            // A BF16 or F32 document has no routed-expert body here.
            row_dec("q4e_expert_gate_up", &[Precision::NvFp4]),
            row_dec("q4e_expert_down", &[Precision::NvFp4]),
            // The token FP8 experts are the short-batch path (a decode step and a chunk below
            // GEMM_MIN_BATCH); the grouped FP8 tile in the PREFILL entry below is the chunk path.
            row("q4e_expert_gate_up_fp8", &[Precision::Fp8E4M3]),
            row("q4e_expert_down_fp8", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    // The genuine FP8 grouped prefill bodies. `q4p_moe_gate_up` and `q4p_moe_down` are the NVFP4
    // grouped bodies, and `q4p_moe_rows_*` are their panel form: those read packed E2M1 codes and
    // claim `NvFp4` only; the body's MXFP8 form is its own `_mx8` symbols. A Qwen FP8
    // chunk resolves the `_fp8` symbols in this image.
    sm121(
        Part::Moe,
        Image::PREFILL,
        &[
            tc_tile("q4p_moe_gate_up", &[Precision::NvFp4]),
            tc_tile("q4p_moe_down", &[Precision::NvFp4]),
            tc_rows("q4p_moe_rows_gate_up", &[Precision::NvFp4]),
            // Inkling's interleaved w13 and softcapped SwiGLU, the same body's arguments.
            tc_rows("q4p_moe_rows_gate_up_capped", &[Precision::NvFp4]),
            tc_rows("q4p_moe_rows_down", &[Precision::NvFp4]),
            tc_tile("q4p_moe_gate_up_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_moe_swiglu_fp8", &[Precision::Fp8E4M3]),
            tc_tile("q4p_moe_down_fp8", &[Precision::Fp8E4M3]),
            // Inkling-2-Small's gate_up pass: the same body over an interleaved w13 with the
            // softcapped SwiGLU; its down pass is q4p_moe_down.
            tc_tile("q4p_moe_gate_up_capped", &[Precision::NvFp4]),
            // The same two passes over Inkling's MXFP8 layers (E4M3 bytes, a ue8m0 code a 32).
            // The MXFP8 grouped tile is also the decode rows' body (Inkling's one-to-eight-row
            // band takes it as it stands; the catalogue has no MXFP8 few-row form).
            tc("q4p_moe_gate_up_capped_mx8", &[Precision::MxFp8]),
            tc("q4p_moe_down_mx8", &[Precision::MxFp8]),
            // The down passes over the row split's gathered runs, with no join before them.
            tc_tile("q4p_moe_down_runs", &[Precision::NvFp4]),
            tc_tile("q4p_moe_down_runs_mx8", &[Precision::MxFp8]),
            // The six passes' head-only forms (SPARKFLASH_INKLING_PLANES=head).
            tc_tile("q4p_moe_gate_up_capped_head", &[Precision::NvFp4]),
            tc_tile("q4p_moe_gate_up_capped_mx8_head", &[Precision::MxFp8]),
            tc_tile("q4p_moe_down_head", &[Precision::NvFp4]),
            tc_tile("q4p_moe_down_mx8_head", &[Precision::MxFp8]),
            tc_tile("q4p_moe_down_runs_head", &[Precision::NvFp4]),
            tc_tile("q4p_moe_down_runs_mx8_head", &[Precision::MxFp8]),
        ],
        false,
    ),
    sm121(
        Part::Moe,
        Image::BLOCK_ROUTED_EXPERTS,
        &[
            tc_tile("glm_moe_fp8b128_mma", &[Precision::Fp8E4M3]),
            // `_bf16a` names the BF16 activations and accumulator, not the stored weights: the
            // body reads E4M3 codes with a 128x128 BF16 tile scale, so its operand class is FP8.
            // A BF16 document therefore has no routed-expert body in either pass.
            tc_tile("glm_moe_fp8b128_mma_bf16a", &[Precision::Fp8E4M3]),
            tc_tile("glm_moe_q8_mma", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::Moe,
        Image::DECODE,
        &[row_dec("q4e_expert_gemv_e4m3_block", &[Precision::Fp8E4M3])],
        false,
    ),
    // Router, combine and grouped GEMM are their own catalogue identities. Sharing Part::Moe
    // would either duplicate a (part, image) entry or apply the MoE block's shape rule to the
    // combine, which is what made GLM's combine unreachable.
    sm121(
        Part::MoeRouter,
        Image::Deepseek,
        &[
            deepseek_sym("moe_router_sm121", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("moe_router_topk_sm121", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("opt_router_sm121", &[Precision::Bf16, Precision::F32]),
            deepseek_sym(
                "opt_router_e128k3_sm121",
                &[Precision::Bf16, Precision::F32],
            ),
            deepseek_sym("opt_mma_router_dots", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("opt_mma_router_topk", &[Precision::Bf16, Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::MoeRouter,
        Image::ROUTED_EXPERTS,
        &[
            sym("q4e_router", &[Precision::Bf16, Precision::F32]),
            // The shared score/top-k/renormalise body the dense arm launches.
            sym("q4e_route", &[Precision::Bf16, Precision::F32]),
        ],
        false,
    ),
    sm121(
        Part::MoeRouter,
        Image::DECODE,
        &[sym("q4e_noaux_router", &[Precision::Bf16, Precision::F32])],
        false,
    ),
    sm121(
        Part::MoeCombine,
        Image::Deepseek,
        &[
            deepseek_sym(
                "moe_routed_combine_sm121",
                &[Precision::Bf16, Precision::F32],
            ),
            deepseek_sym("opt_moe_combine_sm121", &[Precision::Bf16, Precision::F32]),
            deepseek_sym(
                "opt_moe_combine_k3_sm121",
                &[Precision::Bf16, Precision::F32],
            ),
            deepseek_sym("opt_mma_moe_combine", &[Precision::Bf16, Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::MoeCombine,
        Image::ROUTED_EXPERTS,
        &[sym("q4e_moe_combine", &[Precision::Bf16, Precision::F32])],
        false,
    ),
    // The routed experts' mix. The shared combine sums its slots unweighted and adds a shared
    // expert; Inkling's experts carry their share in three different places (an NVFP4 per-expert
    // scale, an MXFP8 body with no per-expert factor, and a BF16 dense body), so the arm hands the
    // mix one scale a slot and the body is a plain scaled sum. That also keeps the host's
    // sequential f32 sum and the kernel identical, which is what the parity case checks.
    sm121(
        Part::MoeCombine,
        Image::Inkling,
        &[
            sym("inkling_expert_mix", &[Precision::F32]),
            sym("inkling_sum_slots", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::Deepseek,
        &[
            deepseek_sym(
                "b12x_mxfp4_moe_stored_sm121",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_moe_gate_up_sm121",
                &[Precision::Fp8E4M3, Precision::MxFp8],
            ),
            deepseek_sym("opt_moe_down_combine_sm121", &[Precision::NvFp4]),
            deepseek_sym("opt_mma_moe_gate_up", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_mma_moe_down", &[Precision::Fp8E4M3, Precision::MxFp8]),
        ],
        true,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::ROUTED_EXPERTS,
        &[
            // These bodies decode packed E2M1 codes, so their operand class is NVFP4 alone.
            // A BF16 or F32 document has no routed-expert body here.
            row_dec("q4e_expert_gate_up", &[Precision::NvFp4]),
            row_dec("q4e_expert_down", &[Precision::NvFp4]),
            // The token FP8 bodies serve a decode step and a short prefill chunk; a real chunk
            // resolves the grouped `q4p_moe_*_fp8` symbols below.
            row("q4e_expert_gate_up_fp8", &[Precision::Fp8E4M3]),
            row("q4e_expert_down_fp8", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    // The BF16 token expert bodies the shared dense arm's routed layers read. They are dense
    // GEMMs an expert run, listed apart from the grouped entries so a BF16 document does not
    // claim a tensor-core grouped body it has none of.
    sm121(
        Part::DenseGemm,
        Image::ROUTED_EXPERTS,
        &[
            row_dec("q4e_expert_gate_up_bf16", &[Precision::Bf16]),
            row_dec("q4e_expert_gate_up_bf16_clamped", &[Precision::Bf16]),
            row_dec("q4e_expert_down_bf16", &[Precision::Bf16]),
            row_dec("q4e_expert_gate_up_fp8_clamped", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::PREFILL,
        &[
            tc_tile("q4p_moe_gate_up", &[Precision::NvFp4]),
            tc_tile("q4p_moe_down", &[Precision::NvFp4]),
            tc_rows("q4p_moe_rows_gate_up", &[Precision::NvFp4]),
            // Inkling's interleaved w13 and softcapped SwiGLU, the same body's arguments.
            tc_rows("q4p_moe_rows_gate_up_capped", &[Precision::NvFp4]),
            tc_rows("q4p_moe_rows_down", &[Precision::NvFp4]),
            tc_tile("q4p_moe_gate_up_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_moe_swiglu_fp8", &[Precision::Fp8E4M3]),
            tc_tile("q4p_moe_down_fp8", &[Precision::Fp8E4M3]),
            // Inkling-2-Small's gate_up pass: the same body over an interleaved w13 with the
            // softcapped SwiGLU; its down pass is q4p_moe_down.
            tc_tile("q4p_moe_gate_up_capped", &[Precision::NvFp4]),
            // The same two passes over Inkling's MXFP8 layers (E4M3 bytes, a ue8m0 code a 32).
            // The MXFP8 grouped tile is also the decode rows' body (Inkling's one-to-eight-row
            // band takes it as it stands; the catalogue has no MXFP8 few-row form).
            tc("q4p_moe_gate_up_capped_mx8", &[Precision::MxFp8]),
            tc("q4p_moe_down_mx8", &[Precision::MxFp8]),
            // The down passes over the row split's gathered runs, with no join before them.
            tc_tile("q4p_moe_down_runs", &[Precision::NvFp4]),
            tc_tile("q4p_moe_down_runs_mx8", &[Precision::MxFp8]),
            // The six passes' head-only forms (SPARKFLASH_INKLING_PLANES=head).
            tc_tile("q4p_moe_gate_up_capped_head", &[Precision::NvFp4]),
            tc_tile("q4p_moe_gate_up_capped_mx8_head", &[Precision::MxFp8]),
            tc_tile("q4p_moe_down_head", &[Precision::NvFp4]),
            tc_tile("q4p_moe_down_mx8_head", &[Precision::MxFp8]),
            tc_tile("q4p_moe_down_runs_head", &[Precision::NvFp4]),
            tc_tile("q4p_moe_down_runs_mx8_head", &[Precision::MxFp8]),
        ],
        false,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::BLOCK_ROUTED_EXPERTS,
        &[
            tc_tile("glm_moe_fp8b128_mma", &[Precision::Fp8E4M3]),
            // `_bf16a` names the BF16 activations and accumulator, not the stored weights: the
            // body reads E4M3 codes with a 128x128 BF16 tile scale, so its operand class is FP8.
            // A BF16 document therefore has no routed-expert body in either pass.
            tc_tile("glm_moe_fp8b128_mma_bf16a", &[Precision::Fp8E4M3]),
            tc_tile("glm_moe_q8_mma", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::DECODE,
        &[row_dec("q4e_expert_gemv_e4m3_block", &[Precision::Fp8E4M3])],
        false,
    ),
    // The draft's own plan builder (opt_moe_plan_sm121) sits beside the shared one; DeepSeek's
    // entry is canonical as it is for every part DeepSeek implements.
    sm121(
        Part::MoePlan,
        Image::Deepseek,
        &[
            deepseek_sym("moe_plan", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("opt_moe_plan_sm121", &[Precision::Bf16, Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::MoePlan,
        Image::MoePlan,
        &[sym("moe_plan", &[Precision::Bf16])],
        false,
    ),
    // DeepSeek's own rank fold and the side-stream prefetch that runs during its all-reduce.
    // The fold is the one-shot exchange's local sum (opt_sum_ranks_bf16); the prefetch is a
    // hint that moves no arithmetic, kept here because it is queued against the collective.
    sm121(
        Part::Collective,
        Image::Deepseek,
        &[
            deepseek_sym("opt_sum_ranks_bf16", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("opt_rows_prefetch_l2", &[Precision::Bf16, Precision::F32]),
            deepseek_sym("opt_exchange_collect", &[Precision::Bf16, Precision::F32]),
            deepseek_sym(
                "opt_exchange_stage_bf16",
                &[Precision::Bf16, Precision::F32],
            ),
        ],
        true,
    ),
    sm121(
        Part::Collective,
        Image::Rocenante,
        &[sym(
            "rocenante_oneshot_allreduce_bf16_tp4",
            &[Precision::Bf16],
        )],
        false,
    ),
    sm121(
        Part::Collective,
        Image::TOKEN,
        &[sym("q4e_sum_slots", &[Precision::Bf16, Precision::F32])],
        false,
    ),
    sm121(
        Part::HyperConnection,
        Image::Deepseek,
        &[
            deepseek_sym("opt_mhc_seam", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_mhc_post", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_bulk_mhc_tail", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_bulk_mhc_partials", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_bulk_mhc_post_partials",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym("mhc_entry_bf16", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("mhc_ffn_seams", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("mhc_attn_residual", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("mhc_ffn_epilogue", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_bulk_seam_derive", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_rows_mhc_post", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_rows_mhc_seam", &[Precision::F32, Precision::Bf16]),
            deepseek_sym("opt_rows_mhc_tail", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_rows_hc_pre_collapse",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym("hc_pre_collapse_bf16", &[Precision::F32, Precision::Bf16]),
            deepseek_sym(
                "opt_rows_mhc_partials_rb1",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_rows_mhc_partials_rb2",
                &[Precision::F32, Precision::Bf16],
            ),
            deepseek_sym(
                "opt_rows_mhc_partials_rb4",
                &[Precision::F32, Precision::Bf16],
            ),
        ],
        true,
    ),
    sm121(
        Part::HyperConnection,
        Image::TOKEN,
        &[
            sym(
                "q4e_hyper_down",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::NvFp4,
                    Precision::F32,
                ],
            ),
            sym(
                "q4e_hyper_up",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::NvFp4,
                    Precision::F32,
                ],
            ),
            sym(
                "q4e_hyper_mix",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::NvFp4,
                    Precision::F32,
                ],
            ),
            sym(
                "q4e_hyper_inject",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::NvFp4,
                    Precision::F32,
                ],
            ),
            sym(
                "q4e_hyper_down_tiled",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::NvFp4,
                    Precision::F32,
                ],
            ),
            sym(
                "q4e_hyper_up_tiled",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::NvFp4,
                    Precision::F32,
                ],
            ),
            // The widened streams themselves are f32: the fold to the branch's width, the
            // widen back out, and the injection the residual takes.
            sym("q4e_fold_streams", &[Precision::F32]),
            sym("q4e_widen_streams", &[Precision::F32]),
            sym("q4e_inject", &[Precision::F32]),
        ],
        false,
    ),
    sm121(
        Part::ManifoldHyperConnection,
        Image::DECODE,
        &[
            sym("q4e_mhc_in", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4e_mhc_post", &[Precision::Bf16, Precision::Fp8E4M3]),
        ],
        true,
    ),
    sm121(
        Part::Engram,
        Image::Deepseek,
        &[
            deepseek_sym(
                "engram_embed_sm121",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "engram_gate_write_sm121",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "engram_wkv_sm121",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "engram_write_bf16",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "engram_narrow_bf16",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "engram_widen_f32",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_bulk_engram_embed",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_bulk_engram_post",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_engram_gate_write",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_engram_wkv",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_rows_engram_post",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_rows_engram_wkv",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_rows_engram_wkv_s",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_mma_fp8_rows_bf16o",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
            deepseek_sym(
                "opt_rows_rank_permute",
                &[
                    Precision::Bf16,
                    Precision::F32,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                ],
            ),
        ],
        true,
    ),
    sm121(
        Part::Ple,
        Image::TOKEN,
        &[
            sym(
                "q4e_ple_gate",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::NvFp4,
                    Precision::F32,
                ],
            ),
            sym(
                "q4e_ple_conv",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::NvFp4,
                    Precision::F32,
                ],
            ),
            sym("q4e_rope", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::Mtp,
        Image::Deepseek,
        &[
            deepseek_sym(
                "opt_draft_fold_prob",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_pick_prob",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_embed_blocks",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_fold_seqs",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_pick_seqs",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_attn",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_logits",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_markov",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_markov_seqs_mma",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_head_fp8",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_head_fp8t",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_logits_fp8",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_taps",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_draft_norm_quant",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_mma_bf16a_fp8row_f32o",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
            deepseek_sym(
                "opt_mma_head_fp8t_f32o",
                &[
                    Precision::Bf16,
                    Precision::Fp8E4M3,
                    Precision::MxFp8,
                    Precision::NvFp4,
                ],
            ),
        ],
        true,
    ),
    sm121(
        Part::Mtp,
        Image::DRAFTER,
        &[
            sym("q4d_capture", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4d_block_embed", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4d_conv", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4d_kv_ring", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4d_attn", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4d_topk_partial", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4d_topk_final", &[Precision::Bf16, Precision::Fp8E4M3]),
            sym("q4d_select", &[Precision::Bf16, Precision::Fp8E4M3]),
            // The Inkling drafter's block attention reuses DFlash's span pair.
            sym("q4d_attn_span_chunk", &[Precision::Bf16]),
            sym("q4d_attn_span_merge", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::VisionPatchEmbed,
        Image::Deepseek,
        &[
            sym("opt_vision_patchify", &[Precision::Bf16]),
            sym("opt_vision_patchify_16", &[Precision::Bf16]),
            sym("opt_vision_gemm", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::VisionPosition,
        Image::Deepseek,
        &[sym("opt_vision_rope", &[Precision::Bf16])],
        true,
    ),
    sm121(
        Part::VisionNorm,
        Image::Deepseek,
        &[sym("opt_vision_rmsnorm", &[Precision::Bf16])],
        true,
    ),
    sm121(
        Part::VisionAttention,
        Image::Deepseek,
        &[sym("opt_vision_attn", &[Precision::Bf16])],
        true,
    ),
    sm121(
        Part::VisionMlp,
        Image::Deepseek,
        &[
            sym("opt_vision_gemm", &[Precision::Bf16]),
            sym("opt_vision_swiglu", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::VisionDownsample,
        Image::Deepseek,
        &[sym("opt_vision_unfold", &[Precision::Bf16])],
        true,
    ),
    sm121(
        Part::VisionMerger,
        Image::Deepseek,
        &[sym("opt_vision_route", &[Precision::Bf16])],
        true,
    ),
    // Inkling's drafter: a DFlash block drafter over its own fused taps.
    sm121(
        Part::Mtp,
        Image::Inkling,
        &[
            sym("inkling_draft_capture", &[Precision::F32]),
            sym("inkling_draft_mix_norm", &[Precision::Bf16]),
            sym("inkling_draft_masks", &[Precision::Bf16]),
            sym("inkling_draft_round", &[Precision::F32]),
            sym("inkling_draft_norm", &[Precision::F32]),
            sym("inkling_draft_add_round", &[Precision::F32]),
            sym("inkling_draft_swiglu", &[Precision::F32]),
            sym("inkling_draft_rope", &[Precision::F32]),
            sym("inkling_draft_context", &[Precision::F32]),
            sym("inkling_draft_markov_input", &[Precision::F32]),
            sym("inkling_draft_markov_gates", &[Precision::F32]),
            sym("inkling_draft_markov_logits", &[Precision::F32]),
            sym("inkling_draft_markov_local", &[Precision::F32]),
            sym("inkling_draft_markov_last", &[Precision::F32]),
        ],
        false,
    ),
    // The Inkling vision tower's components, from the Inkling image. Its patchify, LayerNorm,
    // axial rope and layer scale are bf16 in and out, so the claims name Bf16 as well as F32.
    sm121(
        Part::VisionPatchEmbed,
        Image::Inkling,
        &[
            sym(
                "inkling_vision_patchify",
                &[Precision::Bf16, Precision::F32],
            ),
            sym("opt_vision_gemm", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::VisionPosition,
        Image::Inkling,
        &[sym(
            "inkling_vision_rope",
            &[Precision::Bf16, Precision::F32],
        )],
        false,
    ),
    sm121(
        Part::VisionNorm,
        Image::Inkling,
        &[sym(
            "inkling_vision_layernorm",
            &[Precision::Bf16, Precision::F32],
        )],
        false,
    ),
    sm121(
        Part::VisionAttention,
        Image::Inkling,
        &[
            sym("opt_vision_attn", &[Precision::Bf16]),
            sym(
                "inkling_vision_scale_add",
                &[Precision::Bf16, Precision::F32],
            ),
        ],
        false,
    ),
    sm121(
        Part::VisionMlp,
        Image::Inkling,
        &[
            sym("opt_vision_gemm", &[Precision::Bf16]),
            sym(
                "inkling_vision_scale_add",
                &[Precision::Bf16, Precision::F32],
            ),
        ],
        false,
    ),
    sm121(
        Part::VisionMerger,
        Image::Inkling,
        &[
            sym("opt_vision_gemm", &[Precision::Bf16]),
            sym("inkling_widen_f32", &[Precision::Bf16, Precision::F32]),
        ],
        false,
    ),
    sm121(
        Part::RmsNorm,
        Image::Dory,
        &[
            sym("dory_justnorm", &[Precision::Bf16]),
            sym("dory_slerp", &[Precision::Bf16]),
            sym("dory_loop_combine", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::LmHead,
        Image::Dory,
        &[sym("dory_logit_scale", &[Precision::Bf16])],
        false,
    ),
    sm121(
        Part::GqaAttention,
        Image::Dory,
        &[
            sym("dory_rope_norm", &[Precision::Bf16]),
            sym("dory_rope_norm_lanes", &[Precision::Bf16]),
            sym("dory_attn_split", &[Precision::Bf16]),
            sym("dory_attn_combine", &[Precision::Bf16]),
            sym("dory_attn_mma16", &[Precision::Bf16]),
            sym("dory_attn_mma64", &[Precision::Bf16]),
            sym("dory_attn_mma128", &[Precision::Bf16]),
            sym("dory_attn_mma256", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::DenseMlp,
        Image::Dory,
        &[
            sym("dory_mlp_epilogue", &[Precision::Bf16]),
            tc_wide("dory_mma_wide", &[Precision::Bf16]),
            tc_wide4("dory_mma_wide4", &[Precision::Bf16]),
        ],
        false,
    ),
];

/// One line in [WAIVERS]: a GEMM-class gap the stack may not close yet.
///
/// A waiver names the part, the precision and pass it stands for, the token-row band where the
/// non-tensor-core body runs, and the evidence. `part: None` records a whole-arm property the
/// catalogue cannot express (DeepSeek's three bit dependences). A band is where the waiver
/// applies, not where the missing body would be; the assertion probes one token row and one
/// prompt row, and a waiver covers a probe when the probe falls inside its band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Waiver {
    /// The part the gap belongs to, or `None` for a whole-arm property.
    pub part: Option<Part>,
    /// The precision the gap belongs to, or `None` for every precision.
    pub precision: Option<Precision>,
    /// The pass the gap belongs to.
    pub phase: KernelPhase,
    /// The token-row band where the non-tensor-core body runs.
    pub rows: RowBand,
    /// Why the line stands: the registry entry or document that holds the number.
    pub measurement: &'static str,
}

/// The catalogue's first waiver table.
///
/// Each line is a gap that exists today and is allowed to stand because closing it changes bits,
/// or because no measurement has been recorded. A line citing a document names the line that
/// holds the number; a line with no number says so. The lines below are grouped as the phase-3
/// plan asks: GLM's and Inkling's row-GEMV decode, Dory's, the heads on row GEMVs, and
/// DeepSeek's three bit dependences.
pub const WAIVERS: &[Waiver] = &[
    // The vocabulary head is a row dot product in every arm: the certified head, the per-row
    // screen and Qwen's argmax slices are all row bodies. A tensor-core head exists nowhere yet.
    Waiver {
        part: Some(Part::LmHead),
        precision: Some(Precision::Bf16),
        phase: KernelPhase::Both,
        rows: RowBand::ALL,
        measurement: "heads are row GEMVs in every catalogue entry (cert_lm_head_sm121, \
                      opt_lm_head, opt_rows_lm_head, q4e_argmax_*); no tensor-core head body and \
                      no measurement recorded",
    },
    Waiver {
        part: Some(Part::LmHead),
        precision: Some(Precision::Fp8E4M3),
        phase: KernelPhase::Both,
        rows: RowBand::ALL,
        measurement: "Qwen's FP8 head is the row body q4e_argmax_prob_slices; no tensor-core \
                      head body and no measurement recorded",
    },
    // The BF16 decode rows now split three ways. GLM's BF16 parts keep their row GEMV (its
    // FP8 rows take glm_decode_fp8b128_rows), and Inkling's non-head dense projections keep
    // BATCH_ROWS (8) row GEMV rows before its tile. Dory's projections and Inkling's head take
    // the wide tile's few-row body (q4p_dense_rows), whose row is dory_mma_wide's and
    // glm_prefill_bf16_wide's; the numbers are the row-count thresholds, not a speed
    // measurement, and none is recorded.
    Waiver {
        part: Some(Part::DenseGemm),
        precision: Some(Precision::Bf16),
        phase: KernelPhase::Decode,
        rows: RowBand::at_most(crate::gemv::GEMV_MAX_BATCH),
        measurement: "GLM's BF16 parts keep MAX_ROWS (8) row GEMV rows before \
                      glm_prefill_bf16_*, and Inkling's non-head dense projections keep \
                      BATCH_ROWS (8); Dory's projections and Inkling's head take the wide \
                      tile's few-row body (q4p_dense_rows). Row band from the released \
                      constants; no one-row measurement recorded",
    },
    Waiver {
        part: Some(Part::DenseMlp),
        precision: None,
        phase: KernelPhase::Decode,
        rows: RowBand::at_most(crate::gemv::GEMV_MAX_BATCH),
        measurement: "Inkling's MLP rows keep BATCH_ROWS (8) row GEMV rows; Dory's take the \
                      wide tile's few-row body (q4p_dense_rows, whose row is dory_mma_wide's), \
                      and GLM's MLP rows are FP8 and take the tile's few-row body \
                      (glm_decode_fp8b128_rows). No one-row measurement recorded",
    },
    // DeepSeek's row-GEMV decode bodies are the shared token bodies phase 3b will make common;
    // today the catalogue has the m1 bodies beside the tile bodies.
    Waiver {
        part: Some(Part::DenseGemm),
        precision: Some(Precision::MxFp8),
        phase: KernelPhase::Decode,
        rows: RowBand::at_most(1),
        measurement: "DeepSeek launches b12x_mxfp8_gemv_m1_sm121 on the one-row band \
                      (crates/sparkflash-deepseek-forward); the tile body exists from one row \
                      and no one-row measurement of the tile is recorded",
    },
    // Moe f32 is the routed matvec's activation/arithmetic precision; no tensor-core body claims
    // f32 in either pass.
    Waiver {
        part: Some(Part::Moe),
        precision: Some(Precision::F32),
        phase: KernelPhase::Both,
        rows: RowBand::ALL,
        measurement: "Inkling's routed experts are row matvecs (inkling_*_expert_matvec); no \
                      tensor-core f32 routed body and no measurement recorded",
    },
    // The grouped FP8/MXFP8 tile is prefill-only; the token pass runs the row experts.
    Waiver {
        part: Some(Part::GroupedMoeGemm),
        precision: Some(Precision::Fp8E4M3),
        phase: KernelPhase::Decode,
        rows: RowBand::at_most(crate::gemv::GEMV_MAX_BATCH),
        measurement: "Qwen's q4e_expert_*_fp8 token experts are row bodies below \
                      GEMM_MIN_BATCH = 64; the grouped FP8 tile is prefill-only \
                      (crates/sparkflash-ops/src/parts.rs:1035-1038). No one-row measurement \
                      recorded",
    },
    // Inkling's one-to-eight-row MXFP8 experts take the grouped tile itself
    // (q4p_moe_gate_up_capped_mx8 / q4p_moe_down_mx8, now catalogued on both passes); the
    // steered band above that, up to its 64-row limit, still walks the row matvecs.
    Waiver {
        part: Some(Part::GroupedMoeGemm),
        precision: Some(Precision::MxFp8),
        phase: KernelPhase::Decode,
        // Inkling's steered band covers a decode step, a verify and up to 64 fused rows.
        rows: RowBand::new(crate::gemv::GEMV_MAX_BATCH + 1, 64),
        measurement: "Inkling's nine-to-64-row steered band keeps the row matvecs \
                      (inkling_mxfp8_expert_band / _items); its one-to-eight-row band takes \
                      the grouped MXFP8 tile body (q4p_moe_gate_up_capped_mx8 / \
                      q4p_moe_down_mx8). No one-row measurement recorded",
    },
    Waiver {
        part: Some(Part::Moe),
        precision: Some(Precision::Fp8E4M3),
        phase: KernelPhase::Decode,
        rows: RowBand::at_most(crate::gemv::GEMV_MAX_BATCH),
        measurement: "Qwen's q4e_expert_*_fp8 token experts are row bodies below \
                      GEMM_MIN_BATCH = 64; GLM's grouped FP8 tile is prefill-only. No one-row \
                      measurement recorded",
    },
    Waiver {
        part: Some(Part::Moe),
        precision: Some(Precision::MxFp8),
        phase: KernelPhase::Decode,
        rows: RowBand::new(crate::gemv::GEMV_MAX_BATCH + 1, 64),
        measurement: "Inkling's nine-to-64-row steered band keeps the row matvecs \
                      (inkling_mxfp8_expert_band / _items); its one-to-eight-row band takes \
                      the grouped MXFP8 tile body (q4p_moe_gate_up_capped_mx8 / \
                      q4p_moe_down_mx8). No one-row measurement recorded",
    },
    // DeepSeek's three bit dependences: documented properties of its paths, not missing bodies.
    Waiver {
        part: None,
        precision: None,
        phase: KernelPhase::Both,
        rows: RowBand::ALL,
        measurement: "DeepSeek's bulk path differs from its one-row path in the last bits \
                      (docs/design.md:113)",
    },
    Waiver {
        part: None,
        precision: None,
        phase: KernelPhase::Both,
        rows: RowBand::ALL,
        measurement: "a DeepSeek idle prompt over about 2.6K tokens moves its bits with the \
                      chunk boundary (docs/optimization-strategy.md:497)",
    },
    Waiver {
        part: None,
        precision: None,
        phase: KernelPhase::Both,
        rows: RowBand::ALL,
        measurement: "a DeepSeek request resumed from a reused lane can give different greedy \
                      text than the same request prefilled whole (docs/NUMERICS.md:281)",
    },
];

/// The part classes the catalogue assertion checks.
///
/// A part is GEMM-class when its arithmetic is a matrix product of weights: the dense projection,
/// the dense MLP, the routed experts, the grouped expert GEMM and the vocabulary head.
const fn is_gemm_class(part: Part) -> bool {
    matches!(
        part,
        Part::DenseGemm | Part::DenseMlp | Part::Moe | Part::GroupedMoeGemm | Part::LmHead
    )
}

/// Every precision a kernel can name.
const PRECISIONS: [Precision; 5] = [
    Precision::F32,
    Precision::Bf16,
    Precision::Fp8E4M3,
    Precision::MxFp8,
    Precision::NvFp4,
];

/// The row-count probe for the prompt pass (Qwen's chunk threshold, one bulk row count).
const PROMPT_PROBE_ROWS: usize = crate::projection::QWEN_GEMM_MIN_BATCH;

/// The row-count probe for the token pass.
const TOKEN_PROBE_ROWS: usize = 1;

const fn same_part(left: Part, right: Part) -> bool {
    left as u8 == right as u8
}

const fn same_precision(left: Precision, right: Precision) -> bool {
    left as u8 == right as u8
}

const fn same_class(left: ComputeClass, right: ComputeClass) -> bool {
    left as u8 == right as u8
}

/// Whether a catalogue entry of `part` has a tensor-core body serving the probe.
const fn has_tensor_body(
    part: Part,
    precision: Precision,
    phase: KernelPhase,
    rows: usize,
) -> bool {
    let mut entry_index = 0;
    while entry_index < CATALOGUE.len() {
        let entry = &CATALOGUE[entry_index];
        if same_part(entry.part, part) {
            let mut symbol_index = 0;
            while symbol_index < entry.symbols.len() {
                let symbol = &entry.symbols[symbol_index];
                if same_class(symbol.class, ComputeClass::TensorCore)
                    && symbol.serves(precision, phase, rows)
                {
                    return true;
                }
                symbol_index += 1;
            }
        }
        entry_index += 1;
    }
    false
}

/// Whether any GEMM body of `part` claims `precision`.
const fn part_accepts(part: Part, precision: Precision) -> bool {
    let mut entry_index = 0;
    while entry_index < CATALOGUE.len() {
        let entry = &CATALOGUE[entry_index];
        if same_part(entry.part, part) {
            let mut symbol_index = 0;
            while symbol_index < entry.symbols.len() {
                let symbol = &entry.symbols[symbol_index];
                if symbol.class.is_gemm() && symbol.accepts(precision) {
                    return true;
                }
                symbol_index += 1;
            }
        }
        entry_index += 1;
    }
    false
}

/// Whether a waiver line covers a probe.
const fn waived(part: Part, precision: Precision, phase: KernelPhase, rows: usize) -> bool {
    let mut index = 0;
    while index < WAIVERS.len() {
        let waiver = &WAIVERS[index];
        let part_matches = match waiver.part {
            Some(waiver_part) => same_part(waiver_part, part),
            None => false,
        };
        let precision_matches = match waiver.precision {
            Some(waiver_precision) => same_precision(waiver_precision, precision),
            None => true,
        };
        if part_matches
            && precision_matches
            && waiver.phase.serves(phase)
            && waiver.rows.contains(rows)
        {
            return true;
        }
        index += 1;
    }
    false
}

/// Whether one GEMM-class part has a tensor-core body in both passes, at or below its accepted
/// precisions, or a waiver for each gap.
const fn gemm_part_has_bodies(part: Part) -> bool {
    let mut index = 0;
    while index < PRECISIONS.len() {
        let precision = PRECISIONS[index];
        if part_accepts(part, precision) {
            if !has_tensor_body(part, precision, KernelPhase::Prefill, PROMPT_PROBE_ROWS)
                && !waived(part, precision, KernelPhase::Prefill, PROMPT_PROBE_ROWS)
            {
                return false;
            }
            if !has_tensor_body(part, precision, KernelPhase::Decode, TOKEN_PROBE_ROWS)
                && !waived(part, precision, KernelPhase::Decode, TOKEN_PROBE_ROWS)
            {
                return false;
            }
        }
        index += 1;
    }
    true
}

/// Whether one part is allowed by the catalogue assertion.
///
/// The match is exhaustive on purpose: adding a [Part] variant makes this module fail to compile
/// until the new part is called GEMM-class or elementwise here. A GEMM-class part is then held to
/// [gemm_part_has_bodies], while a non-GEMM part must also disagree with [is_gemm_class], so the
/// two statements of the classification cannot drift apart.
const fn part_bodies_ok(part: Part) -> bool {
    match part {
        Part::DenseGemm | Part::DenseMlp | Part::Moe | Part::GroupedMoeGemm | Part::LmHead => {
            gemm_part_has_bodies(part)
        }
        Part::Embedding
        | Part::RmsNorm
        | Part::MlaAttention
        | Part::Csa2Compression
        | Part::SparseIndexer
        | Part::GqaAttention
        | Part::GdnScan
        | Part::KdaScan
        | Part::Activation
        | Part::Rope
        | Part::RelativeAttention
        | Part::ShortConv
        | Part::BlockAttentionResidual
        | Part::SoftcappedActivation
        | Part::Quantize
        | Part::LayerNorm
        | Part::MoeRouter
        | Part::MoeCombine
        | Part::MoePlan
        | Part::Collective
        | Part::HyperConnection
        | Part::ManifoldHyperConnection
        | Part::Engram
        | Part::Ple
        | Part::Mtp
        | Part::Vision
        | Part::VisionPatchEmbed
        | Part::VisionPosition
        | Part::VisionNorm
        | Part::VisionAttention
        | Part::VisionMlp
        | Part::VisionDownsample
        | Part::VisionMerger => !is_gemm_class(part),
    }
}

/// Whether every part in [Part::ALL] satisfies the catalogue assertion.
const fn every_part_has_bodies() -> bool {
    let mut index = 0;
    while index < Part::ALL.len() {
        if !part_bodies_ok(Part::ALL[index]) {
            return false;
        }
        index += 1;
    }
    true
}

const _: () = assert!(
    every_part_has_bodies(),
    "a GEMM-class part needs a tensor-core body for every precision it accepts in both passes, \
     or a WAIVERS line naming the part, the row band and the measurement; a new Part variant \
     must also be classified in part_bodies_ok"
);

/// The catalogue entries that implement part on target, in catalogue order.
pub fn kernels(part: Part, target: Target) -> impl Iterator<Item = &'static PartKernel> {
    CATALOGUE
        .iter()
        .filter(move |entry| entry.part == part && entry.target == target)
}

/// The row count a tensor-core prompt probe asks the catalogue at: the grouped GEMM's floor.
pub const TENSOR_CORE_PROMPT_ROWS: usize = crate::projection::QWEN_GEMM_MIN_BATCH;

/// The tensor-core coverage of an architecture's matrix-product parts at TP 1, as
/// `sparkflash onboard` reports it: for each part, precision and pass (a
/// [TENSOR_CORE_PROMPT_ROWS]-row prompt and a one-row token), whether the catalogue has a
/// tensor-core body, a [WAIVERS] line covers the gap, or neither does.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorCoreGaps {
    /// Gaps a [WAIVERS] line covers, each with the waiver's measurement.
    pub waived: std::collections::BTreeSet<String>,
    /// Gaps nothing covers: the family writes a body of its own (a `PartExec`).
    pub needing: std::collections::BTreeSet<String>,
}

/// The operator-facing tensor-core section of `sparkflash onboard`.
#[must_use]
pub fn tensor_core_gap_report(architecture: &Architecture, target: Target) -> String {
    use std::fmt::Write as _;
    let gaps = tensor_core_gaps(architecture, target);
    let mut out = format!("tensor-core coverage at {}:\n", target.name());
    if gaps.waived.is_empty() && gaps.needing.is_empty() {
        let _ = writeln!(
            out,
            "  every matrix-product part has a tensor-core body in both passes"
        );
        return out;
    }
    for gap in &gaps.waived {
        let _ = writeln!(out, "  waived: {gap}");
    }
    for gap in &gaps.needing {
        let _ = writeln!(
            out,
            "  needs a tensor-core body (write a PartExec or add a WAIVERS line): {gap}"
        );
    }
    out
}

/// [TensorCoreGaps] for `architecture` at `target`.
#[must_use]
pub fn tensor_core_gaps(architecture: &Architecture, target: Target) -> TensorCoreGaps {
    let needed = architecture.parts();
    let mut per_part: std::collections::BTreeMap<Part, Vec<Precision>> =
        std::collections::BTreeMap::new();
    match crate::EnginePlan::for_architecture(architecture.clone(), target, 1, 0) {
        Ok(plan) => {
            for requirement in plan.requirements() {
                let precisions = per_part.entry(requirement.part).or_default();
                if !precisions.contains(&requirement.precision) {
                    precisions.push(requirement.precision);
                }
            }
        }
        Err(_) => {
            let policy = crate::PrecisionPolicy::from_architecture(architecture);
            let all = [policy.dense, policy.experts, policy.attention, policy.mtp];
            for part in needed.iter().filter(|part| expects_tensor_core(*part)) {
                per_part.entry(part).or_default().extend(all);
            }
        }
    }

    let mut needing = std::collections::BTreeSet::new();
    let mut waived = std::collections::BTreeSet::new();
    for part in needed.iter().filter(|part| expects_tensor_core(*part)) {
        let Some(precisions) = per_part.get(&part) else {
            needing.insert(format!("{}: no TP1 requirement", part.name()));
            continue;
        };
        for precision in precisions {
            for (phase, rows) in [
                (KernelPhase::Prefill, TENSOR_CORE_PROMPT_ROWS),
                (KernelPhase::Decode, 1),
            ] {
                if has_tensor_core_body(target, part, *precision, phase, rows) {
                    continue;
                }
                match tensor_core_waiver(part, *precision, phase, rows) {
                    Some(waiver) => {
                        waived.insert(format!(
                            "{} {} {}: {}",
                            part.name(),
                            precision.name(),
                            phase.name(),
                            waiver.measurement
                        ));
                    }
                    None => {
                        needing.insert(format!(
                            "{} {} {} ({} rows): no tensor-core body, no WAIVERS line",
                            part.name(),
                            precision.name(),
                            phase.name(),
                            rows
                        ));
                    }
                }
            }
        }
    }
    TensorCoreGaps { waived, needing }
}

/// Whether the catalogue exports a tensor-core body for a part, precision, pass and row count.
fn has_tensor_core_body(
    target: Target,
    part: Part,
    precision: Precision,
    phase: KernelPhase,
    rows: usize,
) -> bool {
    CATALOGUE
        .iter()
        .filter(|entry| entry.part == part && entry.target == target)
        .flat_map(|entry| entry.symbols)
        .any(|symbol| {
            symbol.class == ComputeClass::TensorCore && symbol.serves(precision, phase, rows)
        })
}

/// The waiver that covers a gap, if one does.
fn tensor_core_waiver(
    part: Part,
    precision: Precision,
    phase: KernelPhase,
    rows: usize,
) -> Option<&'static Waiver> {
    WAIVERS.iter().find(|waiver| {
        waiver.part == Some(part)
            && waiver.precision.is_none_or(|waived| waived == precision)
            && waiver.phase.serves(phase)
            && waiver.rows.contains(rows)
    })
}

/// Whether a part's arithmetic is a matrix product, and so expects a tensor-core body.
fn expects_tensor_core(part: Part) -> bool {
    matches!(
        part,
        Part::DenseGemm | Part::DenseMlp | Part::Moe | Part::GroupedMoeGemm | Part::LmHead
    )
}

/// The parts the catalogue carries for target.
///
/// A part is carried when at least one catalogue entry implements it, whether or not the image
/// is embedded in this build. [image_available] answers the narrower question.
#[must_use]
pub fn carried(target: Target) -> PartSet {
    CATALOGUE
        .iter()
        .filter(|entry| entry.target == target)
        .map(|entry| entry.part)
        .collect()
}

/// Whether any catalogued image for part is embedded in this build.
///
/// This is stricter than [carried]: a part can be carried by an image that is loaded at run
/// time rather than embedded, which is the case for [Image::Deepseek].
#[must_use]
pub fn image_available(part: Part, target: Target) -> bool {
    kernels(part, target).any(|entry| entry.image.available())
}

/// The parts architecture needs that the catalogue does not carry for target.
///
/// This is the catalogue-level answer: a part counts when any entry implements it, whether or
/// not this build embeds the entry's image. [missing_loadable] is the serving answer.
#[must_use]
pub fn missing(architecture: &Architecture, target: Target) -> Vec<Part> {
    architecture.missing(&carried(target))
}

/// The environment variable naming the run-time cubin directory the DeepSeek arm loads.
pub const CUBIN_DIR_ENV: &str = "SPARKFLASH_CUBINS";

/// Whether image can be loaded by this process.
///
/// This is the one predicate the resolver and the catalogue's serving answer share, so
/// "the resolver accepts it" and "parts::loadable names it" cannot disagree:
///
/// * an embedded image answers for itself;
/// * [Image::Deepseek] is the shared PTX/cubin set loaded at run time, so it is loadable only
///   when the cubin directory is configured;
/// * [Image::Rocenante] compiles its own image at run time, so it is loadable whenever the
///   collective is built.
#[must_use]
pub fn image_loadable(image: Image, cubin_dir: bool) -> bool {
    image.available() || (image == Image::Deepseek && cubin_dir) || image == Image::Rocenante
}

/// The parts the catalogue can load for target, using lookup for the environment.
///
/// A part is loadable when at least one of its catalogue entries names an image this process can
/// load: an embedded image ([Image::available]), the DeepSeek arm's run-time image when
/// [CUBIN_DIR_ENV] names a non-empty directory, or the run-time-compiled RoCEnante image.
#[must_use]
pub fn loadable(target: Target, lookup: impl Fn(&str) -> Option<String>) -> PartSet {
    let cubin_dir = lookup(CUBIN_DIR_ENV).is_some_and(|value| !value.trim().is_empty());
    CATALOGUE
        .iter()
        .filter(|entry| entry.target == target && image_loadable(entry.image, cubin_dir))
        .map(|entry| entry.part)
        .collect()
}

/// The parts the dynamic host reference backend executes.
///
/// This is the one capability set for a checkpoint served on the CPU by
/// `sparkflash-cli/src/serve/dynamic_host.rs`. A novel family can run exactly these parts; the
/// device catalogue is a different question and must not decide whether the host path accepts a
/// checkpoint. Both the CLI dynamic path and the web control plane read this list, so a model
/// the console calls supported is one the host path can actually execute.
///
/// [Part::Mtp] is deliberately absent: the host forward does not run the MTP stage, so a
/// checkpoint that declares one is judged on its backbone (see [host_reference_missing]).
pub const HOST_REFERENCE_PARTS: &[Part] = &[
    Part::Embedding,
    Part::RmsNorm,
    Part::GqaAttention,
    Part::GdnScan,
    Part::DenseMlp,
    Part::Moe,
    Part::LmHead,
];

/// Whether a part belongs to the vision stack.
#[must_use]
pub const fn is_vision_part(part: Part) -> bool {
    matches!(
        part,
        Part::Vision
            | Part::VisionPatchEmbed
            | Part::VisionPosition
            | Part::VisionNorm
            | Part::VisionAttention
            | Part::VisionMlp
            | Part::VisionDownsample
            | Part::VisionMerger
    )
}

/// The parts `architecture` names that the host reference backend does not execute.
///
/// MTP and vision are optional for this backend: [Architecture::parts] inserts [Part::Mtp]
/// whenever the document declares a stage and [Part::Vision] whenever it declares a tower, but
/// the host forward executes neither. Refusing the backbone over a stage or tower the host
/// deliberately ignores would keep the dynamic path from serving an MTP- or vision-bearing
/// document it runs correctly, so both are dropped from the missing set. The storage walk
/// applies the same rule through [host_reference_serves_tensor].
#[must_use]
pub fn host_reference_missing(architecture: &Architecture) -> Vec<Part> {
    let mut available = PartSet::new();
    for part in HOST_REFERENCE_PARTS {
        available.insert(*part);
    }
    architecture
        .missing(&available)
        .into_iter()
        .filter(|part| !matches!(part, Part::Mtp) && !is_vision_part(*part))
        .collect()
}

/// The Dory symbols the arm loads. The catalogue and the boot gate check that the image names
/// every one before a Dory checkpoint is advertised as a device arm.
const DORY_SYMBOLS: &[&str] = &[
    "dory_attn_combine",
    "dory_attn_mma128",
    "dory_attn_mma16",
    "dory_attn_mma256",
    "dory_attn_mma64",
    "dory_attn_split",
    "dory_justnorm",
    "dory_logit_scale",
    "dory_loop_combine",
    "dory_mlp_epilogue",
    "dory_mma_wide",
    "dory_mma_wide4",
    "dory_rope_norm",
    "dory_rope_norm_lanes",
    "dory_slerp",
];

/// The shared Qwen token kernels Dory's projections and embedding load.
const DORY_QWEN_SYMBOLS: &[&str] = &["q4e_embed", "q4e_gemv4_bf16", "q4e_gemv_bf16"];

/// The device arm that serves a checkpoint of a family registered on the dynamic engine.
///
/// Some families' device arms read only some of their releases, and the rest stay on the host
/// reference: the Qwen3.5/3.6-MoE (`qwen3_5_moe`) arm reads the FP8 release, whose routed experts
/// and wide dense projections are E4M3 on 128x128 tiles, but not the BF16 release's fused
/// `mlp.experts.gate_up_proj`, which no device kernel reads yet. This is the one rule the CLI's
/// dynamic route, the arm's own load and the web browser agree on. A family no pack transcribes
/// is the shared dense arm's ([DENSE_DEVICE_ARM]) when [dense_device_refusal] takes it and this
/// build carries its kernels ([dense_device_images]).
///
/// # Errors
/// Why no device arm serves the checkpoint, so the host reference is its executor.
pub fn dynamic_device_arm(architecture: &Architecture) -> Result<&'static str, String> {
    dynamic_device_arm_hosting(architecture, &[])
}

/// [dynamic_device_arm] for a family that registers executors of its own for the parts in
/// `own`, which the shared dense arm then takes as [dense_device_refusal_hosting] says.
///
/// # Errors
/// As [dynamic_device_arm].
pub fn dynamic_device_arm_hosting(
    architecture: &Architecture,
    own: &[Part],
) -> Result<&'static str, String> {
    let family = sparkflash_model::canonical_model_type(&architecture.family);
    if family == "dory" {
        // The released BF16 checkpoint has a Dory device arm: its custom kernels live in the
        // Dory image and its token projections reuse the Qwen token image. Without both images
        // this build's executor is the CPU host reference, so the arm is not advertised.
        if !Image::Dory.available() || !Image::Qwen4Exp.available() {
            return Err(
                "this build has no Dory/Qwen cubin, so the Dory checkpoint stays on the CPU \
                 host reference"
                    .to_owned(),
            );
        }
        let missing: Vec<&str> = DORY_SYMBOLS
            .iter()
            .copied()
            .filter(|symbol| !Image::Dory.exported_symbols().contains(symbol))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "the Dory image does not list the kernels the arm launches: {missing:?}"
            ));
        }
        let missing: Vec<&str> = DORY_QWEN_SYMBOLS
            .iter()
            .copied()
            .filter(|symbol| !Image::Qwen4Exp.exported_symbols().contains(symbol))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "the Qwen token image does not list the kernels the Dory arm launches: {missing:?}"
            ));
        }
        return Ok("the Dory device arm");
    }
    if family != "qwen3_5_moe" {
        // A family no pack transcribes is the shared dense arm's when it is the dense
        // grouped-query decoder that arm executes and this build carries its kernels.
        if let Some(refusal) = dense_device_refusal_hosting(architecture, own) {
            return Err(format!(
                "this build has no device arm for the {family} family: {refusal}"
            ));
        }
        dense_device_images()?;
        return Ok(DENSE_DEVICE_ARM);
    }
    let tile = |format: WeightFormat| {
        matches!(
            format,
            WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                ..
            }
        )
    };
    let quantization = &architecture.quantization;
    if quantization.expert_format == WeightFormat::Bf16 {
        return Err(
            "BF16 routed experts (mlp.experts.gate_up_proj) have no device kernel yet; \
                    serve the FP8 release, or this checkpoint on the host reference"
                .to_owned(),
        );
    }
    if !tile(quantization.expert_format) || !tile(quantization.dense_format) {
        return Err(format!(
            "the Qwen3.5-MoE device arm reads 128x128-tile E4M3 experts and dense projections, \
             not {} experts with {} dense projections",
            quantization.expert_format.name(),
            quantization.dense_format.name()
        ));
    }
    Ok("the Qwen3.5-MoE device arm")
}

/// The columns the shared dense arm's BF16 expert passes read at a time: an expert's width, and
/// each rank's share of it, is whole runs of this many.
pub const DENSE_BF16_EXPERT_RUN: usize = 8;

/// The name [dynamic_device_arm] gives the shared dense arm.
pub const DENSE_DEVICE_ARM: &str = "the shared dense device arm";

/// The token-image kernels the shared dense arm launches.
pub const DENSE_TOKEN_SYMBOLS: &[&str] = &[
    "q4e_add",
    "q4e_deinterleave",
    "q4e_embed",
    "q4e_gemv_bf16",
    "q4e_gemv_tiled_e4m3_tile",
    "q4e_rmsnorm",
    "q4e_rope",
    "q4e_sum_slots",
    "q4e_swiglu",
    "q4e_swiglu_clamped",
    "q4e_zero",
];

/// The scan-image kernels the shared dense arm's Gated DeltaNet layers launch
/// ([crate::gdn]): the conv and its history, the prep, the gated norm, and the scan at each head
/// pair [crate::scan::ScanPlan] may resolve.
pub const DENSE_GDN_SYMBOLS: &[&str] = &[
    "q4p_gdn_conv",
    "q4p_gdn_conv_state",
    "q4p_gdn_gate",
    "q4p_gdn_prep",
    "q4p_gdn_scan",
    "q4p_gdn_scan_128_256",
    "q4p_gdn_scan_64_64",
];

/// The prefill-image kernel the shared dense arm's block-scaled FP8 prompt projections launch
/// ([crate::projection::Fp8Tile]).
pub const DENSE_FP8_TILE_SYMBOL: &str = "q4p_dense_fp8";

/// The attention-image kernels the shared dense arm launches.
pub const DENSE_ATTENTION_SYMBOLS: &[&str] =
    &["q4e_attn_full", "q4e_attn_full_combine", "q4e_cache_bf16"];

/// The routed-expert kernels the shared dense arm launches for an MoE layer: the router, the
/// token path's BF16 and block-scaled FP8 expert passes (and their clamped forms) and the combine.
pub const DENSE_MOE_SYMBOLS: &[&str] = &[
    "q4e_expert_down_bf16",
    "q4e_expert_down_fp8",
    "q4e_expert_gate_up_bf16",
    "q4e_expert_gate_up_bf16_clamped",
    "q4e_expert_gate_up_fp8",
    "q4e_expert_gate_up_fp8_clamped",
    "q4e_moe_combine",
    "q4e_route",
];

/// The most experts the shared arm's router and plan hold (`Q4E_MAX_EXPERTS`,
/// `MOE_PLAN_MAX_EXPERTS`).
pub const DENSE_MAX_EXPERTS: usize = crate::moe::MAX_EXPERTS;

/// The most experts a token of the shared arm activates (`Q4E_MAX_TOPK`).
pub const DENSE_MAX_TOP_K: usize = 64;

/// The image entry of the shared wide tile the dense arm's prompt projections launch
/// ([crate::projection::WideTile]); the Dory image carries it.
pub const DENSE_WIDE_SYMBOL: &str = "dory_mma_wide";

/// The largest head the split full attention takes (`Q4E_BLOCK`).
pub const DENSE_MAX_HEAD_DIM: usize = 256;

/// Whether this build's images carry every kernel the shared dense arm launches.
///
/// # Errors
/// The image or the symbols this build is missing.
pub fn dense_device_images() -> Result<(), String> {
    for (image, symbols) in [
        (Image::Qwen4Exp, DENSE_TOKEN_SYMBOLS),
        (Image::Qwen4Attn, DENSE_ATTENTION_SYMBOLS),
        (Image::Dory, &[DENSE_WIDE_SYMBOL][..]),
        (Image::Qwen4Prefill, &[DENSE_FP8_TILE_SYMBOL][..]),
        (Image::Qwen4Moe, DENSE_MOE_SYMBOLS),
        (Image::Qwen4Scan, DENSE_GDN_SYMBOLS),
        (Image::MoePlan, &["moe_plan"][..]),
    ] {
        if !image.available() {
            return Err(format!(
                "this build has no {image:?} cubin, which the shared dense arm launches from"
            ));
        }
        let missing: Vec<&str> = symbols
            .iter()
            .copied()
            .filter(|symbol| !image.exported_symbols().contains(symbol))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "the {image:?} image does not list the kernels the shared dense arm launches: \
                 {missing:?}"
            ));
        }
    }
    Ok(())
}

/// Why the shared dense device arm cannot execute `architecture`, if it cannot.
///
/// The arm runs the host reference's arithmetic on the token image: a BF16 embedding, RMSNorm,
/// grouped-query attention (with or without the per-head output gate and q/k norms, plain
/// rotate-half RoPE over the leading rotated dimensions) or a Gated DeltaNet (the scan image's
/// chunk kernels, every linear layer of one geometry), a dense SwiGLU or routed experts (a
/// softmax, sigmoid or sqrt-softplus score, the top-k with or without renormalising, a routed
/// scale, a shared expert with or without its sigmoid gate), a plain residual and the head, on
/// one rank or as a tensor-parallel group's rank (each sharded axis split as [sparkflash_model::TpPlan]
/// splits it, every branch's partial rows summed in rank order). Its projections and experts are BF16 or E4M3 under one inverse scale a 128 x 128
/// tile, over widths of whole tiles; the embedding, the head and the norms are BF16. It is gated
/// by a per-layer parity report against that reference, so it takes nothing the reference does
/// not execute, and nothing the reference does that its kernels do not (a linear rotary rule,
/// another activation, a clamp). Everything else is refused by name, and the host reference stays
/// that checkpoint's executor.
#[must_use]
pub fn dense_device_refusal(architecture: &Architecture) -> Option<String> {
    dense_device_refusal_hosting(architecture, &[])
}

/// [dense_device_refusal] for a family that registers executors of its own for the parts in
/// `own` (a token mixer no catalogue part runs, such as an MLA). A layer whose mixer is one of
/// them is the family's to execute: the arm's attention rules (its kinds, head groups, rotary,
/// window and output tiles) do not apply to it, and the host reference's part list does not
/// name it. Every other rule holds: a compression, an indexer, a side lookup or a widened
/// residual is still refused, since the shared walk runs a layer's mixer and feed-forward only.
#[must_use]
pub fn dense_device_refusal_hosting(architecture: &Architecture, own: &[Part]) -> Option<String> {
    use sparkflash_model::architecture::{Activation, AttentionKind, Ffn, Residual};
    let owned: Vec<bool> = architecture
        .steps()
        .iter()
        .map(|step| own.contains(&step.attention))
        .collect();
    let owned = |index: usize| owned.get(index).copied().unwrap_or(false);
    let missing: Vec<String> = host_reference_missing(architecture)
        .into_iter()
        .filter(|part| !own.contains(part))
        .map(|part| part.to_string())
        .collect();
    if !missing.is_empty() {
        return Some(host_reference_missing_refusal(&missing));
    }
    if let Some(refusal) = rope_refusal_where(architecture, &|index| !owned(index))
        .or_else(|| host_reference_ffn_refusal(architecture))
    {
        return Some(refusal);
    }
    let dense_format = architecture.quantization.dense_format;
    if !dense_arm_reads(dense_format) {
        return Some(format!(
            "its dense projections are {}, and the arm reads BF16 or 128 x 128 block-scaled FP8",
            dense_format.name()
        ));
    }
    let tiled = dense_format != WeightFormat::Bf16;
    if tiled && !architecture.hidden_size.is_multiple_of(E4M3_TILE) {
        return Some(format!(
            "its block-scaled FP8 projections read whole {E4M3_TILE}-column scale tiles, and the \
             hidden width {} is not one",
            architecture.hidden_size
        ));
    }
    if architecture.hidden_size == 0
        || architecture.vocab_size == 0
        || architecture.layers.is_empty()
    {
        return Some("the checkpoint declares an empty model".to_owned());
    }
    let whole = [
        (architecture.looped.is_some(), "a looped recurrence"),
        (architecture.inkling.is_some(), "Inkling-2's attention"),
        (architecture.attention_bias, "attention biases"),
    ];
    if let Some((_, what)) = whole.iter().find(|(declared, _)| *declared) {
        return Some(format!(
            "it declares {what}, which the arm does not execute"
        ));
    }
    // Every linear-attention layer takes the one scan kernel the arm loads for the first.
    let first_gdn = architecture
        .layers
        .iter()
        .find_map(|layer| match &layer.attention.kind {
            AttentionKind::Gdn(gdn) => Some((layer.index, gdn)),
            _ => None,
        });
    for layer in &architecture.layers {
        let index = layer.index;
        let attention = &layer.attention;
        let mixer_owned = owned(index);
        let kv_heads = match &attention.kind {
            _ if mixer_owned => None,
            AttentionKind::Gqa { kv_heads, .. } => Some(*kv_heads),
            AttentionKind::Gdn(gdn) => {
                if let Some((first, geometry)) = first_gdn
                    && geometry != gdn
                {
                    return Some(format!(
                        "layer {index}'s Gated DeltaNet is not layer {first}'s, and the arm runs \
                         every linear-attention layer on one scan"
                    ));
                }
                if let Some(refusal) = dense_gdn_refusal(gdn) {
                    return Some(format!("layer {index}'s Gated DeltaNet: {refusal}"));
                }
                None
            }
            // Every kind is named, no wildcard: a new kind leads the compiler here.
            AttentionKind::Mla { .. } | AttentionKind::Inkling { .. } | AttentionKind::Kda(_) => {
                return Some(format!(
                    "layer {index} is neither grouped-query attention nor a Gated DeltaNet, the \
                     attentions the arm runs"
                ));
            }
        };
        let declared = [
            (
                !mixer_owned && attention.window.is_some(),
                "a sliding window",
            ),
            (attention.compression.is_some(), "a compressed cache"),
            (attention.indexer.is_some(), "a sparse indexer"),
            (
                !mixer_owned && attention.rope.mrope_section.is_some(),
                "an mRoPE section",
            ),
            (!layer.sides.is_empty(), "a side table"),
            (
                !matches!(layer.residual, Residual::Plain),
                "a widened residual",
            ),
            (layer.post_norm, "a norm after each sublayer"),
        ];
        if let Some((_, what)) = declared.iter().find(|(declared, _)| *declared) {
            return Some(format!(
                "layer {index} declares {what}, which the arm does not execute"
            ));
        }
        if let Some(kv_heads) = kv_heads
            && let Some(refusal) = dense_gqa_refusal(attention, kv_heads)
        {
            return Some(format!("layer {index}{refusal}"));
        }
        // The output projection's columns: the attended heads, or the linear attention's values.
        let query_width = match &attention.kind {
            // The family's part reads its own output projection.
            _ if mixer_owned => 0,
            AttentionKind::Gdn(gdn) => gdn.value_heads * gdn.value_dim,
            AttentionKind::Gqa { .. }
            | AttentionKind::Mla { .. }
            | AttentionKind::Inkling { .. }
            | AttentionKind::Kda(_) => attention.heads * attention.head_dim,
        };
        let ffn_width = match &layer.ffn {
            Ffn::Dense(dense) => dense.intermediate,
            Ffn::Moe(moe) if moe.shared_experts > 0 => moe.shared_intermediate,
            Ffn::Moe(_) => 0,
        };
        if tiled && !(query_width.is_multiple_of(E4M3_TILE) && ffn_width.is_multiple_of(E4M3_TILE))
        {
            return Some(format!(
                "layer {index}'s block-scaled FP8 output and down projections read whole \
                 {E4M3_TILE}-column scale tiles, and the attention width {query_width} or the MLP \
                 width {ffn_width} is not one"
            ));
        }
        match &layer.ffn {
            Ffn::Dense(dense) if dense.activation == Activation::Silu => {}
            Ffn::Dense(dense) => {
                return Some(format!(
                    "layer {index}'s MLP is {:?}, and the arm runs a SwiGLU",
                    dense.activation
                ));
            }
            Ffn::Moe(moe) => {
                if let Some(refusal) = dense_moe_refusal(architecture, moe) {
                    return Some(format!("layer {index}'s routed experts: {refusal}"));
                }
            }
        }
    }
    None
}

/// Why the shared dense arm's split attention and rotary cannot execute a grouped-query layer's
/// `attention` over `kv_heads`, if they cannot: the reason as it follows "layer N".
fn dense_gqa_refusal(
    attention: &sparkflash_model::architecture::Attention,
    kv_heads: usize,
) -> Option<String> {
    if attention.heads == 0
        || kv_heads == 0
        || !attention.heads.is_multiple_of(kv_heads)
        || attention.head_dim == 0
        || attention.head_dim > DENSE_MAX_HEAD_DIM
    {
        return Some(format!(
            "'s attention ({} heads over {kv_heads} key/value heads of {}) is not one the split \
             attention takes (whole groups, heads of 1..={DENSE_MAX_HEAD_DIM})",
            attention.heads, attention.head_dim
        ));
    }
    let rope = &attention.rope;
    if !rope.rotated_dims.min(attention.head_dim).is_multiple_of(2) {
        return Some(format!(
            " rotates {} dimensions a head, which is not even",
            rope.rotated_dims
        ));
    }
    if rope.rotated_dims >= 2
        && let Some(scaling) = &rope.scaling
        && !matches!(scaling.kind.to_ascii_lowercase().as_str(), "" | "default")
    {
        return Some(format!(
            " declares rope.scaling {:?} (factor {}); the arm's rotary is the plain rule",
            scaling.kind, scaling.factor
        ));
    }
    None
}

/// Why the scan image's chunk kernels cannot execute `gdn`, if they cannot: a head pair the scan
/// is not compiled for ([SCAN_HEAD_PAIRS]), a conv window past its history, key heads that do not
/// divide the value heads, or an output gate other than SiLU or sigmoid.
fn dense_gdn_refusal(gdn: &sparkflash_model::architecture::Gdn) -> Option<String> {
    use sparkflash_model::architecture::Activation;
    let heads = (gdn.key_dim, gdn.value_dim);
    if !SCAN_HEAD_PAIRS.contains(&heads) {
        return Some(format!(
            "its (key, value) heads {heads:?} are not a pair the scan is compiled for \
             {SCAN_HEAD_PAIRS:?}"
        ));
    }
    let most = crate::gdn::CONV_HISTORY + 1;
    if !(2..=most).contains(&gdn.conv_kernel) {
        return Some(format!(
            "its conv takes {} taps, and the scan carries 2..={most}",
            gdn.conv_kernel
        ));
    }
    if gdn.key_heads == 0
        || gdn.value_heads == 0
        || !gdn.value_heads.is_multiple_of(gdn.key_heads)
        || gdn.value_heads / gdn.key_heads > gdn.key_dim
    {
        return Some(format!(
            "its {} key heads do not divide its {} value heads into groups of at most {}",
            gdn.key_heads, gdn.value_heads, gdn.key_dim
        ));
    }
    if !matches!(gdn.output_gate, Activation::Silu | Activation::Sigmoid) {
        return Some(format!(
            "its output gate is {:?}, and the gated norm takes SiLU or sigmoid",
            gdn.output_gate
        ));
    }
    None
}

/// Whether the shared dense arm reads weights stored as `format`: BF16, or E4M3 codes under one
/// inverse scale a 128 x 128 tile (the host reference's one FP8 arithmetic, whatever the release
/// declares of its activations: both keep them f32).
fn dense_arm_reads(format: WeightFormat) -> bool {
    matches!(
        format,
        WeightFormat::Bf16
            | WeightFormat::Fp8Tile {
                block_m: E4M3_TILE,
                block_n: E4M3_TILE,
                ..
            }
    )
}

/// Whether the shared dense arm reads routed experts stored as `format`: [dense_arm_reads], or
/// NVFP4 (E2M1 codes, one E4M3 scale a 16-value group and a per-tensor f32).
fn dense_arm_reads_experts(format: WeightFormat) -> bool {
    dense_arm_reads(format) || matches!(format, WeightFormat::NvFp4 { .. })
}

/// Why the shared dense arm's routed experts cannot execute `moe`, if they cannot.
fn dense_moe_refusal(
    architecture: &Architecture,
    moe: &sparkflash_model::architecture::Moe,
) -> Option<String> {
    use sparkflash_model::architecture::{Activation, Score};
    let format = architecture.quantization.expert_format;
    if !dense_arm_reads_experts(format) {
        return Some(format!(
            "they are {}, and the arm reads BF16, 128 x 128 block-scaled FP8 or NVFP4 experts",
            format.name()
        ));
    }
    if moe.activation != Activation::Silu {
        return Some(format!(
            "their MLP is {:?}, and the arm runs a SwiGLU",
            moe.activation
        ));
    }
    if moe.routed_experts == 0 || moe.routed_experts > DENSE_MAX_EXPERTS {
        return Some(format!(
            "{} experts is outside the router's 1..={DENSE_MAX_EXPERTS}",
            moe.routed_experts
        ));
    }
    if moe.active_experts == 0
        || moe.active_experts > DENSE_MAX_TOP_K
        || moe.active_experts > moe.routed_experts
    {
        return Some(format!(
            "a token activates {} of {} experts, outside the router's 1..={DENSE_MAX_TOP_K}",
            moe.active_experts, moe.routed_experts
        ));
    }
    let hidden = architecture.hidden_size;
    let (multiple, rows) = match format {
        WeightFormat::Bf16 => (DENSE_BF16_EXPERT_RUN, "BF16 rows eight columns at a time"),
        WeightFormat::NvFp4 { .. } => (16, "NVFP4 rows a 16-value scale group at a time"),
        _ => (
            E4M3_TILE,
            "block-scaled FP8 rows a 128-column scale tile at a time",
        ),
    };
    if !hidden.is_multiple_of(multiple)
        || moe.intermediate == 0
        || !moe.intermediate.is_multiple_of(multiple)
    {
        return Some(format!(
            "the expert passes read {rows}, and the hidden width {hidden} or the expert width {} \
             is not a positive multiple of {multiple}",
            moe.intermediate
        ));
    }
    if moe.norm_topk
        && moe.active_experts == 1
        && moe.score != Score::Softmax
        && !moe.selection_bias
    {
        return Some(
            "one pick renormalised under an independent score: the router keeps its score (the \
             top-k-above-one rule) where the host reference divides it to one"
                .to_owned(),
        );
    }
    if moe.shared_experts > 1 {
        return Some(format!(
            "{} shared experts; the arm adds one",
            moe.shared_experts
        ));
    }
    if moe.shared_experts == 0 && moe.shared_gated {
        return Some("a shared-expert gate with no shared expert".to_owned());
    }
    None
}

/// Why the host reference backend cannot execute `architecture`, if it cannot.
///
/// This is the one boot-support predicate the CLI host backend and the web control plane share.
/// It covers everything that is cheap to decide from the checkpoint document alone: a part the
/// host does not execute, a rotary scaling rule the reference forward does not apply, and a
/// declared quantized weight format the reference's BF16/F32 decode would serve dishonestly.
/// Both callers additionally inspect tensor headers for quantization the document did not
/// declare, through [host_reference_storage_quantization_refusal]. A Hub-only web inspection has
/// no shards yet, so that preview remains declaration-only.
#[must_use]
pub fn host_reference_refusal(architecture: &Architecture) -> Option<String> {
    host_reference_structure_refusal(architecture)
        .or_else(|| host_reference_quantization_refusal(architecture))
}

/// The document-level host refusal that needs no tensor inspection.
///
/// This is the half both backends run before they resolve a weight; their storage check then
/// inspects the actual tensor headers, so a `U8` checkpoint is named as such rather than only by
/// the declaration. The web gate uses [host_reference_refusal], which also folds in the declared
/// formats.
#[must_use]
pub fn host_reference_structure_refusal(architecture: &Architecture) -> Option<String> {
    let missing = host_reference_missing(architecture);
    if !missing.is_empty() {
        let names: Vec<String> = missing.iter().map(ToString::to_string).collect();
        return Some(host_reference_missing_refusal(&names));
    }
    host_reference_rope_refusal(architecture).or_else(|| host_reference_ffn_refusal(architecture))
}

/// The refusal that names the parts the host reference does not execute.
fn host_reference_missing_refusal(names: &[String]) -> String {
    format!(
        "the checkpoint names parts the dynamic host path does not execute: {}",
        names.join(", ")
    )
}

/// The first layer whose MLP or routing rule the host reference does not apply.
///
/// The reference selects experts by score, or by score plus the `noaux_tc` correction bias
/// ([sparkflash_weights::topk::noaux_top_k]), over every expert, and clamps a SwiGLU the release
/// limits (`silu(min(gate, limit)) * clamp(up, -limit, limit)`, GLM's form). Group-limited
/// selection (`n_group`, `topk_group`), a bias over a softmax score and a clamp on another
/// activation would be computed as if undeclared, so they are refused at boot.
#[must_use]
pub fn host_reference_ffn_refusal(architecture: &Architecture) -> Option<String> {
    use sparkflash_model::architecture::{Activation, Ffn, Score};
    for layer in &architecture.layers {
        let index = layer.index;
        let (activation, clamp) = match &layer.ffn {
            Ffn::Dense(dense) => (dense.activation, dense.clamp),
            Ffn::Moe(moe) => {
                if let Some(limit) = moe.group_limit {
                    return Some(format!(
                        "layer {index} picks its experts from the best {} of {} expert groups, \
                         which the dynamic host path does not limit",
                        limit.kept, limit.groups
                    ));
                }
                if moe.selection_bias && moe.score == Score::Softmax {
                    return Some(format!(
                        "layer {index} adds a selection bias to a softmax score; the dynamic host \
                         path adds it to independent (sigmoid or sqrt-softplus) scores"
                    ));
                }
                (moe.activation, moe.clamp)
            }
        };
        if clamp.is_some() && activation != Activation::Silu {
            return Some(format!(
                "layer {index} clamps a {activation:?} MLP; the dynamic host path clamps a SwiGLU"
            ));
        }
    }
    None
}

/// The first layer whose rotary scaling rule the host reference does not apply.
///
/// The host forward applies plain RoPE and a positive `linear` factor only. A `yarn`, `dynamic`
/// or other rule is accepted by the shared architecture and the plan, but the reference
/// arithmetic would silently compute unscaled positions; refuse it at boot instead of failing
/// the first request inside `HostSession::forward`.
#[must_use]
pub fn host_reference_rope_refusal(architecture: &Architecture) -> Option<String> {
    rope_refusal_where(architecture, &|_| true)
}

/// [host_reference_rope_refusal] over the layers `checked` takes.
fn rope_refusal_where(
    architecture: &Architecture,
    checked: &dyn Fn(usize) -> bool,
) -> Option<String> {
    for layer in architecture
        .layers
        .iter()
        .filter(|layer| checked(layer.index))
    {
        let rope = &layer.attention.rope;
        // The forward skips a layer whose rotation is narrower than one pair (GLM's NoPE MLA).
        // Its scaling rule is inert there, so a yarn declaration on such a layer is not a
        // mismatch: the host and the forward both ignore it.
        if rope.rotated_dims < 2 {
            continue;
        }
        let Some(scaling) = &rope.scaling else {
            continue;
        };
        let kind = scaling.kind.to_ascii_lowercase();
        match kind.as_str() {
            // The released documents spell plain RoPE both ways; treat them the same as the
            // forward does.
            "" => {}
            "default" => {
                if scaling.factor != 1.0 {
                    return Some(format!(
                        "layer {} declares rope.scaling default with factor {}, which the dynamic host path ignores; use factor 1 or a positive linear rope, or serve with a device pack",
                        layer.index, scaling.factor
                    ));
                }
            }
            "linear" if scaling.factor > 0.0 => {}
            "linear" => {
                return Some(format!(
                    "layer {} declares rope.scaling linear with factor {}, which is not positive",
                    layer.index, scaling.factor
                ));
            }
            other => {
                return Some(format!(
                    "layer {} declares rope.scaling {other:?}, which the dynamic host path does not apply; use a plain or linear rope, or serve with a device pack",
                    layer.index
                ));
            }
        }
    }
    None
}

/// Whether the dynamic host reference decodes a tensor with this name.
///
/// The host resolves the backbone's own tensors. The MTP stage and the vision tower/aligner are
/// stored beside them but the reference forward never executes them, so a quantized tensor under
/// either belongs to a stage the host ignores rather than to the arithmetic it serves. An MTP
/// stage can also be stored as an ordinary extra decoder layer, which has no `mtp.` prefix and
/// needs the architecture; [is_extra_layer] covers that spelling. The storage walk uses this
/// predicate to judge only the tensors the host actually decodes; the structure gate separately
/// ignores [Part::Mtp] and [Part::Vision] in [host_reference_missing].
#[must_use]
pub fn host_reference_serves_tensor(name: &str) -> bool {
    let mtp = name.starts_with("mtp.") || name.contains(".mtp.");
    let vision = name.starts_with("vision.")
        || name.starts_with("model.visual")
        || name.starts_with("aligner.")
        || name.starts_with("model.aligner");
    !(mtp || vision)
}

/// Whether `name` addresses a decoder layer beyond the architecture's backbone.
///
/// A GLM-style MTP stage is stored as an ordinary extra layer at
/// `model.language_model.layers.<num_hidden_layers>.*` (or `model.layers.N`), not under an
/// `mtp.` prefix, so [host_reference_serves_tensor] cannot tell it apart by name alone. The
/// storage walk consults the document instead: when it declares an MTP stage, layers at or
/// beyond `architecture.layers.len()` are the stage's, which the host forward never reads.
fn is_extra_layer(name: &str, architecture: &Architecture) -> bool {
    if architecture.mtp.is_none() {
        return false;
    }
    const LAYER_PREFIXES: [&str; 3] = ["model.language_model.layers.", "model.layers.", "layers."];
    LAYER_PREFIXES.iter().any(|prefix| {
        name.strip_prefix(prefix).is_some_and(|rest| {
            rest.split('.')
                .next()
                .and_then(|index| index.parse::<usize>().ok())
                .is_some_and(|index| index >= architecture.layers.len())
        })
    })
}

/// The inverse-scale tensor beside an E4M3 tile-quantized weight, if the checkpoint stores one.
///
/// Qwen and GLM append `_scale_inv` to the weight name. It addresses a grid of
/// `ceil(rows / 128) x ceil(cols / 128)` values, which the host applies as a tile scale.
#[must_use]
fn fp8_tile_scale(source: &dyn sparkflash_weights::WeightSource, name: &str) -> Option<String> {
    if !name.ends_with(".weight") {
        return None;
    }
    let candidate = format!("{name}_scale_inv");
    source
        .spec(&candidate)
        .ok()
        .filter(|spec| {
            matches!(
                spec.dtype,
                sparkflash_weights::DType::F32 | sparkflash_weights::DType::BF16
            )
        })
        .map(|_| candidate)
}

/// Whether a tensor name is a quantization scale rather than a weight.
#[must_use]
fn is_scale_tensor(name: &str) -> bool {
    name.ends_with("_scale_inv")
        || name.ends_with("_scale")
        || name.ends_with("weight_scale")
        || name.ends_with("weight_scale_2")
        || name.ends_with("input_scale")
}

/// The stored weight format the host reference cannot decode honestly, if any.
///
/// A document can omit `quantization_config` while its shards store `U8`, packed FP4 or FP8
/// codes beside scale tensors. The host decodes BF16/F32 per tensor and applies one 128x128
/// inverse-scale tile to an E4M3 weight, so the first tensor it cannot decode is named from its
/// safetensors header. The CLI and the web control plane both call this over the checkpoint
/// source they already opened, so a storage-only quantized checkpoint cannot be supported by one
/// and refused by the other.
///
/// Only tensors [host_reference_serves_tensor] accepts are inspected: a quantized `mtp.*` or
/// vision tensor is not part of the forward the host runs, so it must not refuse the backbone.
/// `architecture` supplies the layer count [is_extra_layer] needs to skip a GLM-style MTP stage
/// stored as an extra decoder layer.
#[must_use]
pub fn host_reference_storage_quantization_refusal(
    architecture: &Architecture,
    source: &dyn sparkflash_weights::WeightSource,
) -> Option<String> {
    for name in source.names() {
        if !host_reference_serves_tensor(&name) || is_extra_layer(&name, architecture) {
            continue;
        }
        let Ok(spec) = source.spec(&name) else {
            continue;
        };
        if spec.dtype == sparkflash_weights::DType::F8E4M3 {
            // E4M3 with a tile inverse scale is decodable: the host multiplies each element by
            // its 128x128 block's scale on the way in. Only an unscaled code tensor is opaque.
            if fp8_tile_scale(source, &name).is_some() {
                continue;
            }
        } else if is_scale_tensor(&name) {
            // A scale beside a supported weight is metadata, not a weight of its own.
            continue;
        }
        if matches!(
            spec.dtype,
            sparkflash_weights::DType::U8
                | sparkflash_weights::DType::F4E2M1
                | sparkflash_weights::DType::F8E4M3
                | sparkflash_weights::DType::F8E8M0
        ) {
            return Some(format!(
                "the checkpoint stores quantized tensor {name} ({}); the dynamic host reference decodes BF16/F32 tensors and applies one tile scale to an E4M3 weight but no other quantized storage, so it cannot serve this tensor honestly - convert or serve with its device pack",
                spec.dtype.name()
            ));
        }
    }
    None
}

/// The declared weight format the host reference cannot decode honestly, if any.
///
/// The host path decodes BF16/F32 per tensor and applies one 128x128 inverse-scale tile to an
/// E4M3 weight. A document that declares NVFP4, MXFP8 or another block size would be served
/// with the wrong arithmetic, so it is refused at boot; a 128x128 FP8 tile is accepted.
#[must_use]
pub fn host_reference_quantization_refusal(architecture: &Architecture) -> Option<String> {
    let quantization = &architecture.quantization;
    // The host does not execute the MTP stage, so an MTP-only quantized format is not a reason
    // to refuse the backbone the host does run. Only the dense and expert formats the host
    // decodes belong in this predicate.
    let host_decodes = |format: &WeightFormat| {
        matches!(format, WeightFormat::Bf16)
            || matches!(
                format,
                WeightFormat::Fp8Tile {
                    block_m: 128,
                    block_n: 128,
                    ..
                }
            )
    };
    if host_decodes(&quantization.dense_format) && host_decodes(&quantization.expert_format) {
        return None;
    }
    Some(format!(
        "the checkpoint declares quantized weights (method={:?}, dense={:?}, experts={:?}); the dynamic host reference decodes BF16/F32 tensors and applies one 128x128 tile scale to an E4M3 weight, so it cannot serve other quantized formats honestly - convert or serve it with its device pack, or use an unquantized checkpoint",
        quantization.method,
        quantization.dense_format.name(),
        quantization.expert_format.name(),
    ))
}

/// The parts a serving process that stages the checkpoint's vision tower launches for target.
///
/// The DeepSeek arm loads and runs the 14-patch tower, so [Vision](Part::Vision) stays in its
/// serving set and a 16-patch document is refused at boot. [serving_language] is the narrower
/// set for the Qwen and GLM arms, whose tower is a later extraction (D84).
#[must_use]
pub fn serving(target: Target) -> PartSet {
    carried(target)
}

/// The parts a language-only serving process launches for target.
///
/// Every released document plans a vision tower (D84), but the Qwen and GLM packs do not stage
/// one, so their serve gate must demand a body only for the language parts. A serving boot check
/// uses this set; [CATALOGUE] and [crate::resolver::KernelResolver::boot_check] still judge the
/// whole plan, so the catalogue's honest 14-patch vision constraint is untouched.
#[must_use]
pub fn serving_language(target: Target) -> PartSet {
    serving(target)
        .into_iter()
        .filter(|part| !is_vision_part(*part))
        .collect()
}

/// The parts architecture needs that this build can actually load for target.
///
/// This is [missing] narrowed by [loadable]: a part whose only catalogued image is the DeepSeek
/// run-time image is missing until [CUBIN_DIR_ENV] names that directory.
#[must_use]
pub fn missing_loadable_with(
    architecture: &Architecture,
    target: Target,
    lookup: impl Fn(&str) -> Option<String>,
) -> Vec<Part> {
    architecture.missing(&loadable(target, lookup))
}

/// [missing_loadable_with] over the process environment.
#[must_use]
pub fn missing_loadable(architecture: &Architecture, target: Target) -> Vec<Part> {
    missing_loadable_with(architecture, target, |name| std::env::var(name).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::PrecisionMask;
    use std::collections::HashSet;

    /// The block-scaled FP8 declaration the Qwen3 FP8 releases write.
    fn fp8() -> serde_json::Value {
        serde_json::json!({
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128]
        })
    }

    const DEEPSEEK: &str =
        include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
    const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
    const QWEN: &str =
        include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");

    fn architectures() -> [(&'static str, Architecture); 3] {
        [
            (
                "DeepSeek V4.1",
                Architecture::from_json(DEEPSEEK).expect("DeepSeek config"),
            ),
            ("GLM-5.3", Architecture::from_json(GLM).expect("GLM config")),
            (
                "Qwen3.8",
                Architecture::from_json(QWEN).expect("Qwen config"),
            ),
        ]
    }

    #[test]
    fn every_part_has_sm121_entries_with_symbols() {
        for part in Part::ALL {
            let entries: Vec<&PartKernel> = kernels(*part, Target::Sm121).collect();
            assert!(!entries.is_empty(), "{part} has no sm121 entry");
            for entry in entries {
                assert_eq!(entry.part, *part);
                assert_eq!(entry.target, Target::Sm121);
                assert!(!entry.symbols.is_empty(), "{part} has an empty symbol list");
                for symbol in entry.symbols {
                    assert!(!symbol.name.is_empty(), "{part} names an empty symbol");
                }
            }
            assert!(
                kernels(*part, Target::Sm100).next().is_none(),
                "{part} claims an sm100 entry"
            );
            assert!(
                kernels(*part, Target::Sm120).next().is_none(),
                "{part} claims an sm120 entry"
            );
        }
        assert!(CATALOGUE.iter().all(|entry| entry.target == Target::Sm121));
    }

    #[test]
    fn every_symbol_declares_its_own_precision_and_pass() {
        for entry in CATALOGUE {
            for symbol in entry.symbols {
                assert!(
                    symbol.precisions.count() > 0,
                    "{}::{} on {} declares no precision",
                    entry.part,
                    symbol.name,
                    entry.image.name()
                );
                assert_ne!(
                    symbol.precisions,
                    PrecisionMask::ALL,
                    "{}::{} on {} still accepts every precision",
                    entry.part,
                    symbol.name,
                    entry.image.name()
                );
            }
        }
    }

    #[test]
    fn there_is_exactly_one_entry_per_part_and_image() {
        for part in Part::ALL {
            let mut images = HashSet::new();
            for entry in kernels(*part, Target::Sm121) {
                assert!(
                    images.insert(entry.image),
                    "{part} has two catalogue entries for {}",
                    entry.image.name()
                );
            }
        }
    }

    #[test]
    fn every_requested_symbol_is_exported_by_its_image() {
        for entry in CATALOGUE {
            for symbol in entry.symbols {
                assert!(
                    entry.image.exported_symbols().contains(&symbol.name),
                    "{}::{} is not exported by {}",
                    entry.part,
                    symbol.name,
                    entry.image.name()
                );
            }
        }
    }

    #[test]
    fn every_catalogued_image_is_a_known_image_variant() {
        let known = [
            Image::Qwen4Exp,
            Image::Qwen4Attn,
            Image::Qwen4Moe,
            Image::Qwen4Prefill,
            Image::Qwen4Scan,
            Image::Dflash2,
            Image::GlmPrefill,
            Image::GlmAttention,
            Image::GlmMoe,
            Image::GlmDecode,
            Image::MoePlan,
            Image::Deepseek,
            Image::Dory,
            Image::Inkling,
            Image::Rocenante,
        ]
        .map(Image::name);
        for entry in CATALOGUE {
            assert!(
                known.contains(&entry.image.name()),
                "{} names the unknown image {}",
                entry.part,
                entry.image.name()
            );
        }
    }

    #[test]
    fn deepseek_remains_canonical_for_every_part_it_implements() {
        for part in Part::ALL {
            let entries: Vec<&PartKernel> = kernels(*part, Target::Sm121).collect();
            let canonical: Vec<&&PartKernel> =
                entries.iter().filter(|entry| entry.canonical).collect();
            assert_eq!(
                canonical.len(),
                1,
                "{part} has {} canonical entries, expected exactly one",
                canonical.len()
            );
            assert!(
                entries[0].canonical,
                "{part} does not list its canonical entry first"
            );
            if entries.iter().any(|entry| entry.image == Image::Deepseek) {
                assert_eq!(
                    entries[0].image,
                    Image::Deepseek,
                    "{part} does not put its DeepSeek entry first"
                );
                assert_eq!(
                    canonical[0].image,
                    Image::Deepseek,
                    "{part} makes a non-DeepSeek image canonical"
                );
            }
        }
    }

    #[test]
    fn the_three_released_configs_are_complete_on_sm121() {
        for (name, architecture) in architectures() {
            let missing = missing(&architecture, Target::Sm121);
            assert!(missing.is_empty(), "{name} is missing {missing:?} on sm121");
        }
    }

    /// Inkling-2-Small's four new arithmetic families are catalogued at the geometry the
    /// released document names, with the bounds the kernel source states.
    #[test]
    fn inkling_entries_match_the_released_geometry() {
        use crate::plan::PartGeometry;

        const INKLING: &str =
            include_str!("../../sparkflash-model/tests/data/inkling2_config.json");
        let architecture = Architecture::from_json(INKLING).expect("the Inkling-2 config parses");
        let required = [
            Part::RelativeAttention,
            Part::ShortConv,
            Part::BlockAttentionResidual,
            Part::SoftcappedActivation,
        ];
        for part in required {
            assert!(
                architecture.parts().contains(part),
                "the released document requires {part}"
            );
            let entry = kernels(part, Target::Sm121)
                .find(|entry| entry.image == Image::Inkling)
                .unwrap_or_else(|| panic!("{part} has no Inkling image entry"));
            assert!(entry.canonical, "{part}'s Inkling entry must be canonical");
            assert_eq!(entry.shape.hidden_equals, Some(INKLING_HIDDEN));
        }

        // The attention body takes the GQA group as a runtime argument, so the entry must not
        // pin the released group 6: the plan's RelativeAttention geometry carries heads and
        // head width but no kv_heads, and a gqa_groups rule would refuse it.
        let attention = kernels(Part::RelativeAttention, Target::Sm121)
            .next()
            .expect("the relative-attention entry");
        let released = PartGeometry {
            hidden: INKLING_HIDDEN,
            heads: 48,
            kv_heads: 8,
            head_dim: INKLING_HEAD_DIM,
            groups: 6,
            positions: 1024,
            ..PartGeometry::new()
        };
        assert!(attention.shape.accepts(&released).is_ok());
        assert!(
            attention
                .shape
                .accepts(&PartGeometry {
                    kv_heads: 0,
                    ..released
                })
                .is_ok(),
            "the relative-attention entry must accept the plan's geometry, which has no kv_heads"
        );
        // The body maps a query head to its kv head at run time, so another group is a shape
        // the same source runs, not a refusal.
        assert!(
            attention
                .shape
                .accepts(&PartGeometry {
                    heads: 32,
                    kv_heads: 4,
                    groups: 8,
                    ..released
                })
                .is_ok()
        );
        for wrong in [
            PartGeometry {
                hidden: 4096,
                ..released
            },
            PartGeometry {
                head_dim: 64,
                ..released
            },
        ] {
            assert!(
                attention.shape.accepts(&wrong).is_err(),
                "the relative-attention entry accepted {wrong:?}"
            );
        }

        // The short-convolution body rejects only kernels outside 1..=8
        // (`kernels/inkling_conv_sm121.cu:134`); the released width is 4.
        let convolution = kernels(Part::ShortConv, Target::Sm121)
            .next()
            .expect("the short-convolution entry");
        for window in [1, 4, 8] {
            assert!(
                convolution
                    .shape
                    .accepts(&PartGeometry {
                        hidden: INKLING_HIDDEN,
                        window,
                        ..PartGeometry::new()
                    })
                    .is_ok(),
                "the short-convolution entry refused kernel {window}"
            );
        }
        for window in [0, 9] {
            assert!(
                convolution
                    .shape
                    .accepts(&PartGeometry {
                        hidden: INKLING_HIDDEN,
                        window,
                        ..PartGeometry::new()
                    })
                    .is_err(),
                "the short-convolution entry accepted kernel {window}"
            );
        }
        let activation = kernels(Part::SoftcappedActivation, Target::Sm121)
            .next()
            .expect("the softcapped activation entry");
        assert!(
            activation
                .shape
                .accepts(&PartGeometry {
                    hidden: INKLING_HIDDEN,
                    intermediate: INKLING_INTERMEDIATE,
                    ..PartGeometry::new()
                })
                .is_ok()
        );
        assert!(
            activation
                .shape
                .accepts(&PartGeometry {
                    hidden: INKLING_HIDDEN,
                    intermediate: 4096,
                    ..PartGeometry::new()
                })
                .is_err()
        );
    }

    /// The token image's embedding, head screen and dense projection are runtime-shaped, so
    /// the catalogue must not pin them to Qwen's 2,560 width or its 128-aligned vocabulary.
    #[test]
    fn qwen_token_bodies_take_the_runtime_widths_their_arms_launch() {
        use crate::plan::PartGeometry;

        let geometry = PartGeometry {
            hidden: 4608,
            vocab: 201024,
            ..PartGeometry::new()
        };
        let embedding = kernels(Part::Embedding, Target::Sm121)
            .find(|entry| entry.image == Image::Qwen4Exp)
            .expect("the token embedding entry");
        assert!(
            embedding.shape.accepts(&geometry).is_ok(),
            "q4e_embed is runtime-shaped: {geometry:?}"
        );
        let head = kernels(Part::LmHead, Target::Sm121)
            .find(|entry| entry.image == Image::Qwen4Exp)
            .expect("the token head entry");
        assert!(
            head.shape.accepts(&geometry).is_ok(),
            "q4e_argmax_slices is runtime-shaped: {geometry:?}"
        );
        let dense = kernels(Part::DenseGemm, Target::Sm121)
            .find(|entry| entry.image == Image::Qwen4Exp)
            .expect("the token dense entry");
        for hidden in [2560, 4608] {
            assert!(
                dense
                    .shape
                    .accepts(&PartGeometry {
                        hidden,
                        ..PartGeometry::new()
                    })
                    .is_ok(),
                "q4e_gemv_bf16 / q4e_gemm_bf16 are runtime-shaped at {hidden}"
            );
        }
    }

    /// TP4 splits Inkling's routed experts over the ranks; the expert bodies take that owned
    /// run as a runtime argument, and only the router has a bound (4,096 experts, 64 picks).
    #[test]
    fn inkling_moe_takes_the_rank_local_expert_run() {
        use crate::plan::PartGeometry;

        let entry = kernels(Part::Moe, Target::Sm121)
            .find(|entry| entry.image == Image::Inkling)
            .expect("the Inkling routed-expert entry");
        for (experts, top_k) in [(96, 8), (384, 8)] {
            assert!(
                entry
                    .shape
                    .accepts(&PartGeometry {
                        hidden: INKLING_HIDDEN,
                        intermediate: INKLING_INTERMEDIATE,
                        experts,
                        top_k,
                        ..PartGeometry::new()
                    })
                    .is_ok(),
                "the Inkling MoE entry refused {experts} experts and top-{top_k}"
            );
        }
        for (experts, top_k) in [(4097, 8), (384, 65)] {
            assert!(
                entry
                    .shape
                    .accepts(&PartGeometry {
                        hidden: INKLING_HIDDEN,
                        intermediate: INKLING_INTERMEDIATE,
                        experts,
                        top_k,
                        ..PartGeometry::new()
                    })
                    .is_err(),
                "the Inkling MoE entry accepted {experts} experts and top-{top_k}"
            );
        }
    }

    /// Dory's prompt projections are dory_mma_wide, which the arm names a dense projection;
    /// the body was reachable only under the MLP part until this entry.
    #[test]
    fn dory_dense_projection_is_catalogued_under_its_own_part() {
        use crate::plan::PartGeometry;

        let entry = kernels(Part::DenseGemm, Target::Sm121)
            .find(|entry| entry.image == Image::Dory)
            .expect("the Dory dense-projection entry");
        assert!(!entry.canonical);
        for symbol in ["dory_mma_wide", "dory_mma_wide4"] {
            assert!(
                entry
                    .symbols
                    .iter()
                    .any(|candidate| candidate.name == symbol),
                "the Dory dense entry does not name {symbol}"
            );
        }
        assert!(
            entry
                .shape
                .accepts(&PartGeometry {
                    hidden: 2560,
                    ..PartGeometry::new()
                })
                .is_ok()
        );
        let wide4 = entry
            .symbols
            .iter()
            .find(|symbol| symbol.name == "dory_mma_wide4")
            .expect("the four-matrix body");
        assert!(
            wide4
                .shape
                .accepts(&PartGeometry {
                    hidden: 2560,
                    output: 2560,
                    ..PartGeometry::new()
                })
                .is_ok()
        );
        let refused = wide4
            .shape
            .accepts(&PartGeometry {
                hidden: 2560,
                output: 2559,
                ..PartGeometry::new()
            })
            .expect_err("a 2559-column output is not a 32-column step");
        assert!(refused.contains("multiple of 32"), "{refused}");
    }

    /// Dory's served shape resolves every part its arm launches through the catalogue.
    #[test]
    fn dory_served_shape_is_catalogued() {
        use crate::plan::PartGeometry;

        let served = PartGeometry {
            hidden: 2560,
            vocab: 131072,
            positions: 262144,
            heads: 16,
            kv_heads: 4,
            head_dim: 256,
            groups: 4,
            intermediate: 7168,
            ..PartGeometry::new()
        };
        for (part, images) in [
            (Part::Embedding, &[Image::Qwen4Exp][..]),
            (Part::RmsNorm, &[Image::Dory][..]),
            (Part::LmHead, &[Image::Dory][..]),
            (Part::GqaAttention, &[Image::Dory][..]),
            (Part::DenseMlp, &[Image::Dory][..]),
            (
                Part::DenseGemm,
                &[Image::Qwen4Exp, Image::Qwen4Prefill, Image::Dory][..],
            ),
        ] {
            for image in images {
                let entry = kernels(part, Target::Sm121)
                    .find(|entry| entry.image == *image)
                    .unwrap_or_else(|| panic!("{part} has no {} entry", image.name()));
                assert!(
                    entry.shape.accepts(&served).is_ok(),
                    "{} refuses Dory's served shape: {served:?}",
                    image.name()
                );
            }
        }
    }

    #[test]
    fn sm120_names_every_required_part() {
        for (name, architecture) in architectures() {
            let required: Vec<Part> = architecture.parts().iter().collect();
            assert_eq!(
                missing(&architecture, Target::Sm120),
                required,
                "{name} does not name every part missing on sm120"
            );
            assert!(!required.is_empty());
        }
    }

    #[test]
    fn the_model_side_claim_and_the_kernel_catalogue_agree() {
        for target in Target::ALL {
            assert_eq!(
                sparkflash_model::parts::built(*target),
                carried(*target),
                "the model-side claim and the kernel catalogue disagree for {target}"
            );
        }
    }

    #[test]
    fn the_loadable_answer_is_never_larger_than_the_catalogue_answer() {
        for (name, architecture) in architectures() {
            let catalogue = missing(&architecture, Target::Sm121);
            let loadable = missing_loadable_with(&architecture, Target::Sm121, |_| None);
            for part in catalogue {
                assert!(
                    loadable.contains(&part),
                    "{name}: {part} is catalogued but loadable without an image"
                );
            }
        }
    }

    #[test]
    fn the_resolver_and_the_catalogue_share_one_loadability_predicate() {
        for image in [
            Image::Qwen4Exp,
            Image::Qwen4Attn,
            Image::Qwen4Moe,
            Image::Qwen4Prefill,
            Image::Qwen4Scan,
            Image::Dflash2,
            Image::GlmPrefill,
            Image::GlmAttention,
            Image::GlmMoe,
            Image::GlmDecode,
            Image::MoePlan,
            Image::Deepseek,
            Image::Dory,
            Image::Inkling,
            Image::Rocenante,
        ] {
            for cubin_dir in [false, true] {
                assert_eq!(
                    image_loadable(image, cubin_dir),
                    crate::resolver::Availability::runtime(cubin_dir).loadable(image),
                    "{} disagrees at cubin_dir={cubin_dir}",
                    image.name()
                );
            }
        }
    }

    #[test]
    fn the_deepseek_arm_loads_only_with_its_runtime_cubin_directory() {
        let (_, deepseek) = architectures()
            .into_iter()
            .find(|(name, _)| *name == "DeepSeek V4.1")
            .expect("the DeepSeek config");
        // CSA2 has exactly one image, the DeepSeek run-time image: with no directory configured
        // the loadable answer names it, while the catalogue answer does not.
        let without = missing_loadable_with(&deepseek, Target::Sm121, |_| None);
        assert!(
            without.contains(&Part::Csa2Compression),
            "a run-time-only image must be missing without its directory: {without:?}"
        );
        assert!(missing(&deepseek, Target::Sm121).is_empty());
        let with = missing_loadable_with(&deepseek, Target::Sm121, |name| {
            (name == CUBIN_DIR_ENV).then(|| "/opt/sparkflash/cubins".to_owned())
        });
        assert!(
            with.is_empty(),
            "the DeepSeek arm is complete with cubins: {with:?}"
        );
    }

    /// The Qwen3.6 FP8 release is the Qwen3.5-MoE device arm's; the same document with no
    /// quantization block (BF16 fused experts) and every other family are not.
    #[test]
    fn dory_has_a_dynamic_device_arm_only_with_both_cubins() {
        const DORY: &str = include_str!(
            "../../sparkflash-model-dory/config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json"
        );
        let architecture = Architecture::from_json(DORY).expect("Dory config");
        let arm = dynamic_device_arm(&architecture);
        if Image::Dory.available() && Image::Qwen4Exp.available() {
            assert_eq!(arm, Ok("the Dory device arm"));
        } else {
            let error = arm.expect_err("a build without both images stays on the host");
            assert!(error.contains("no Dory/Qwen cubin"), "{error}");
        }
    }

    #[test]
    fn only_the_tile_quantized_qwen35_release_has_a_dynamic_device_arm() {
        const QWEN36: &str =
            include_str!("../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json");
        let fp8 = Architecture::from_json(QWEN36).expect("the Qwen3.6 document parses");
        assert!(
            dynamic_device_arm(&fp8).is_ok(),
            "{:?}",
            dynamic_device_arm(&fp8)
        );
        let mut document: serde_json::Value =
            serde_json::from_str(QWEN36).expect("the Qwen3.6 document");
        document
            .as_object_mut()
            .expect("an object")
            .remove("quantization_config");
        let bf16 = Architecture::from_json(&document.to_string()).expect("the BF16 twin parses");
        let refusal = dynamic_device_arm(&bf16).expect_err("BF16 experts stay on the host");
        assert!(refusal.contains("BF16 routed experts"), "{refusal}");
        let mut other = fp8.clone();
        other.family = "novel_family".to_owned();
        assert!(dynamic_device_arm(&other).is_err());
    }

    /// A dense grouped-query family no pack names is the shared dense arm's, and what that arm
    /// does not execute is refused by name.
    #[test]
    fn a_dense_gqa_family_is_the_shared_dense_arms_and_the_rest_is_named() {
        let document = serde_json::json!({
            "model_type": "acme_dense",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        });
        let parse = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut document = document.clone();
            edit(&mut document);
            Architecture::from_json(&document.to_string()).expect("the document parses")
        };
        let dense = parse(&|_| {});
        assert_eq!(dense_device_refusal(&dense), None);
        let tiled = parse(&|document| document["quantization_config"] = fp8());
        assert!(matches!(
            tiled.quantization.dense_format,
            WeightFormat::Fp8Tile { .. }
        ));
        assert_eq!(dense_device_refusal(&tiled), None, "FP8 projections");
        let arm = dynamic_device_arm(&dense);
        if dense_device_images().is_ok() {
            assert_eq!(arm, Ok(DENSE_DEVICE_ARM));
        } else {
            assert!(arm.expect_err("no images, no arm").contains("cubin"));
        }
        for (edit, named) in [
            (
                &(|document: &mut serde_json::Value| {
                    document["sliding_window"] = serde_json::json!(128);
                    document["use_sliding_window"] = serde_json::json!(true);
                }) as &dyn Fn(&mut serde_json::Value),
                "sliding window",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2.0});
                },
                "rope.scaling",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["hidden_act"] = serde_json::json!("gelu");
                },
                "SwiGLU",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["attention_bias"] = serde_json::json!(true);
                },
                "attention biases",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["head_dim"] = serde_json::json!(512);
                },
                "heads of 1..=256",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["quantization_config"] = fp8();
                    document["intermediate_size"] = serde_json::json!(544);
                },
                "the MLP width 544",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["quantization_config"] =
                        serde_json::json!({"quant_method": "fp8", "weight_block_size": [64, 64]});
                },
                "BF16 or 128 x 128 block-scaled FP8",
            ),
        ] {
            let architecture = parse(edit);
            let refusal = dense_device_refusal(&architecture)
                .or_else(|| host_reference_structure_refusal(&architecture))
                .expect("the shape is refused");
            assert!(refusal.contains(named), "{named}: {refusal}");
        }
    }

    /// A hybrid family's Gated DeltaNet layers are the shared dense arm's at every head pair the
    /// scan is compiled for, and what its chunk kernels do not execute is refused by name.
    #[test]
    fn gated_deltanet_layers_are_the_shared_dense_arms_and_the_rest_is_named() {
        let document = serde_json::json!({
            "model_type": "acme_hybrid",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention", "full_attention", "linear_attention", "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        });
        let parse = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut document = document.clone();
            edit(&mut document);
            Architecture::from_json(&document.to_string()).expect("the document parses")
        };
        let hybrid = parse(&|_| {});
        assert!(matches!(
            hybrid.layers[0].attention.kind,
            sparkflash_model::architecture::AttentionKind::Gdn(_)
        ));
        assert_eq!(dense_device_refusal(&hybrid), None);
        for (key, dim) in [(64, 64), (128, 256)] {
            let pair = parse(&|document| {
                document["linear_key_head_dim"] = serde_json::json!(key);
                document["linear_value_head_dim"] = serde_json::json!(dim);
                document["output_gate_type"] = serde_json::json!("sigmoid");
            });
            assert_eq!(dense_device_refusal(&pair), None, "({key}, {dim})");
        }
        let tiled = parse(&|document| document["quantization_config"] = fp8());
        assert_eq!(dense_device_refusal(&tiled), None, "FP8 projections");
        for (edit, named) in [
            (
                &(|document: &mut serde_json::Value| {
                    document["linear_value_head_dim"] = serde_json::json!(96);
                }) as &dyn Fn(&mut serde_json::Value),
                "a pair the scan is compiled for",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["linear_conv_kernel_dim"] = serde_json::json!(10);
                },
                "2..=9",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["linear_num_value_heads"] = serde_json::json!(3);
                },
                "do not divide",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["output_gate_type"] = serde_json::json!("gelu");
                },
                "SiLU or sigmoid",
            ),
        ] {
            let architecture = parse(edit);
            let refusal = dense_device_refusal(&architecture)
                .or_else(|| host_reference_structure_refusal(&architecture))
                .expect("the shape is refused");
            assert!(refusal.contains(named), "{named}: {refusal}");
        }
    }

    /// A routed family's BF16 experts are the shared dense arm's under any score, and what its
    /// router and expert passes do not execute is refused by name.
    #[test]
    fn routed_bf16_experts_are_the_shared_dense_arms_and_the_rest_is_named() {
        let document = serde_json::json!({
            "model_type": "acme_moe",
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 256,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        });
        let parse = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut document = document.clone();
            edit(&mut document);
            Architecture::from_json(&document.to_string()).expect("the document parses")
        };
        assert_eq!(dense_device_refusal(&parse(&|_| {})), None);
        let tiled = parse(&|document| document["quantization_config"] = fp8());
        assert!(matches!(
            tiled.quantization.expert_format,
            WeightFormat::Fp8Tile { .. }
        ));
        assert_eq!(dense_device_refusal(&tiled), None, "FP8 experts");
        for score in ["sigmoid", "sqrtsoftplus"] {
            let independent = parse(&|document| {
                document["scoring_func"] = serde_json::json!(score);
                document["norm_topk_prob"] = serde_json::json!(false);
                document["routed_scaling_factor"] = serde_json::json!(2.5);
            });
            assert_eq!(dense_device_refusal(&independent), None, "{score}");
        }
        // GLM's router and SwiGLU: the noaux_tc correction bias over a sigmoid score (one pick
        // included: it keeps its score, the top-k-above-one rule), one group of every expert, and
        // a clamp on every MLP.
        for picks in [1, 2] {
            let noaux = parse(&|document| {
                document["scoring_func"] = serde_json::json!("sigmoid");
                document["topk_method"] = serde_json::json!("noaux_tc");
                document["n_group"] = serde_json::json!(1);
                document["topk_group"] = serde_json::json!(1);
                document["num_experts_per_tok"] = serde_json::json!(picks);
                document["swiglu_limit"] = serde_json::json!(7.0);
            });
            let sparkflash_model::architecture::Ffn::Moe(moe) = &noaux.layers[0].ffn else {
                panic!("a routed layer");
            };
            assert!(moe.selection_bias && moe.group_limit.is_none() && moe.clamp == Some(7.0));
            assert_eq!(dense_device_refusal(&noaux), None, "noaux_tc at {picks}");
        }
        for (edit, named) in [
            (
                &(|document: &mut serde_json::Value| {
                    document["scoring_func"] = serde_json::json!("sigmoid");
                    document["topk_method"] = serde_json::json!("noaux_tc");
                    document["n_group"] = serde_json::json!(4);
                    document["topk_group"] = serde_json::json!(2);
                }) as &dyn Fn(&mut serde_json::Value),
                "best 2 of 4 expert groups",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["topk_method"] = serde_json::json!("noaux_tc");
                },
                "selection bias to a softmax score",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["hidden_act"] = serde_json::json!("gelu");
                    document["swiglu_limit"] = serde_json::json!(7.0);
                },
                "clamps a",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["num_experts"] = serde_json::json!(1024);
                },
                "outside the router's 1..=512",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["num_experts_per_tok"] = serde_json::json!(9);
                },
                "a token activates 9 of 8",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["moe_intermediate_size"] = serde_json::json!(132);
                },
                "multiple of 8",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["quantization_config"] = fp8();
                    document["moe_intermediate_size"] = serde_json::json!(192);
                },
                "positive multiple of 128",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["quantization_config"] = serde_json::json!({"quant_method": "nvfp4"});
                },
                "BF16 or 128 x 128 block-scaled FP8",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["scoring_func"] = serde_json::json!("sigmoid");
                    document["num_experts_per_tok"] = serde_json::json!(1);
                },
                "one pick renormalised",
            ),
            (
                &|document: &mut serde_json::Value| {
                    document["n_shared_experts"] = serde_json::json!(2);
                },
                "2 shared experts",
            ),
        ] {
            let refusal = dense_device_refusal(&parse(edit)).expect("the shape is refused");
            assert!(refusal.contains(named), "{named}: {refusal}");
        }
    }

    /// Qwen3.6-35B-A3B is a covered host family: its vision tower and MTP stage are stored but
    /// not executed, and its 128x128 E4M3 weights are tile-scaled rather than opaque codes.
    #[test]
    fn qwen36_is_a_host_reference_family() {
        const QWEN36: &str =
            include_str!("../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json");
        let architecture = Architecture::from_json(QWEN36).expect("the Qwen3.6 document parses");
        assert_eq!(architecture.family, "qwen3_5_moe");
        assert!(
            architecture.parts().contains(Part::VisionPatchEmbed),
            "the premise: the document carries a tower"
        );
        assert!(
            host_reference_missing(&architecture).is_empty(),
            "the host ignores the tower: {:?}",
            host_reference_missing(&architecture)
        );
        assert_eq!(
            host_reference_quantization_refusal(&architecture),
            None,
            "128x128 E4M3 tiles are decoded by the host"
        );
        assert_eq!(
            host_reference_refusal(&architecture),
            None,
            "the family is host-runnable: {:?}",
            host_reference_refusal(&architecture)
        );
    }

    /// The host does not execute the MTP stage, so an MTP-only quantized format must not refuse
    /// the BF16 backbone the host actually decodes. The web predicate reads this same function.
    #[test]
    fn an_mtp_only_quantized_format_does_not_refuse_the_backbone() {
        // A host-runnable document (dense MLP, GQA, no vision/PLE/hyper-connection/indexer)
        // that still declares an MTP stage. Making the premise explicit matters: the released
        // Qwen document also carries stages the host deliberately refuses, so it cannot show
        // that the MTP declaration alone is ignored.
        let document = serde_json::json!({
            "model_type": "acme_mtp",
            "architectures": ["AcmeMtpForCausalLM"],
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 512,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "eos_token_id": 2,
            "layer_types": ["full_attention"],
            "mtp_num_hidden_layers": 1,
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "mtp.layers.0.mlp.experts": {
                        "quant_algo": "FP8_PB_WO",
                        "group_size": 128
                    }
                }
            }
        });
        let architecture =
            Architecture::from_json(&document.to_string()).expect("the MTP-only document parses");
        assert!(
            architecture.mtp.is_some(),
            "the premise: the document declares an MTP stage"
        );
        assert!(
            architecture.quantization.mtp_expert_format.is_some(),
            "the MTP format is still recorded"
        );
        assert_eq!(
            host_reference_quantization_refusal(&architecture),
            None,
            "the host does not run MTP, so its format is not a backbone refusal"
        );
        assert!(
            host_reference_refusal(&architecture).is_none(),
            "the host must serve the backbone of an MTP-bearing document: {:?}",
            host_reference_refusal(&architecture)
        );

        // The storage half must apply the same rule: a quantized MTP tensor, and a quantized
        // vision tensor the reference never executes, are skipped. A quantized tensor the host
        // does decode still names its dtype, so the filter is not a blanket amnesty.
        let directory = tempfile::tempdir().expect("tempdir");
        sparkflash_weights::safetensors::write_checkpoint(
            directory.path(),
            &[
                (
                    sparkflash_weights::TensorSpec {
                        name: "model.embed_tokens.weight".to_owned(),
                        dtype: sparkflash_weights::DType::BF16,
                        shape: vec![1],
                    },
                    vec![0; 2],
                ),
                (
                    sparkflash_weights::TensorSpec {
                        name: "mtp.layers.0.mlp.experts.weight".to_owned(),
                        dtype: sparkflash_weights::DType::U8,
                        shape: vec![1],
                    },
                    vec![0x21],
                ),
                (
                    sparkflash_weights::TensorSpec {
                        name: "model.visual.blocks.0.attn.wqkv.weight".to_owned(),
                        dtype: sparkflash_weights::DType::U8,
                        shape: vec![1],
                    },
                    vec![0x21],
                ),
            ],
            1,
        )
        .expect("the MTP/vision quantized shard writes");
        let source = sparkflash_weights::safetensors::open_checkpoint(directory.path())
            .expect("the MTP/vision shard opens");
        assert_eq!(
            host_reference_storage_quantization_refusal(&architecture, &source),
            None,
            "quantized MTP and vision tensors are not part of the host forward"
        );

        let served = tempfile::tempdir().expect("tempdir");
        sparkflash_weights::safetensors::write_checkpoint(
            served.path(),
            &[(
                sparkflash_weights::TensorSpec {
                    name: "model.layers.0.self_attn.q_proj.weight".to_owned(),
                    dtype: sparkflash_weights::DType::U8,
                    shape: vec![1],
                },
                vec![0x21],
            )],
            1,
        )
        .expect("the served quantized shard writes");
        let served = sparkflash_weights::safetensors::open_checkpoint(served.path())
            .expect("the served shard opens");
        let refusal = host_reference_storage_quantization_refusal(&architecture, &served)
            .expect("a served U8 tensor is still refused");
        assert!(
            refusal.contains("q_proj") && refusal.contains("U8"),
            "{refusal}"
        );
    }

    /// A GLM-style MTP stage is an ordinary extra layer at
    /// `model.language_model.layers.<num_hidden_layers>.*`, so the prefix-only storage filter
    /// misses it. The walk must use the document's layer count instead, and still refuse a
    /// quantized tensor on a backbone layer.
    #[test]
    fn an_extra_layer_mtp_stage_is_not_a_served_tensor() {
        let document = serde_json::json!({
            "model_type": "acme_extra_layer_mtp",
            "architectures": ["AcmeExtraLayerMtpForCausalLM"],
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 512,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "eos_token_id": 2,
            "layer_types": ["full_attention"],
            "num_nextn_predict_layers": 1
        });
        let architecture = Architecture::from_json(&document.to_string())
            .expect("the extra-layer MTP document parses");
        assert_eq!(architecture.layers.len(), 1);
        assert!(
            architecture.mtp.is_some(),
            "the premise: the document declares an MTP stage"
        );
        assert!(
            host_reference_refusal(&architecture).is_none(),
            "the host must serve the backbone: {:?}",
            host_reference_refusal(&architecture)
        );

        // The stage is spelled both ways: the canonical `model.layers.N` and the nested
        // `model.language_model.layers.N`. Neither is part of the forward the host runs.
        let directory = tempfile::tempdir().expect("tempdir");
        sparkflash_weights::safetensors::write_checkpoint(
            directory.path(),
            &[
                (
                    sparkflash_weights::TensorSpec {
                        name: "model.layers.0.mlp.down_proj.weight".to_owned(),
                        dtype: sparkflash_weights::DType::BF16,
                        shape: vec![1],
                    },
                    vec![0; 2],
                ),
                (
                    sparkflash_weights::TensorSpec {
                        name: "model.layers.1.mlp.down_proj.weight".to_owned(),
                        dtype: sparkflash_weights::DType::U8,
                        shape: vec![1],
                    },
                    vec![0x21],
                ),
                (
                    sparkflash_weights::TensorSpec {
                        name: "model.language_model.layers.1.mlp.down_proj.weight".to_owned(),
                        dtype: sparkflash_weights::DType::U8,
                        shape: vec![1],
                    },
                    vec![0x21],
                ),
            ],
            1,
        )
        .expect("the extra-layer MTP shard writes");
        let source = sparkflash_weights::safetensors::open_checkpoint(directory.path())
            .expect("the extra-layer MTP shard opens");
        assert_eq!(
            host_reference_storage_quantization_refusal(&architecture, &source),
            None,
            "the MTP layer is stored as an extra decoder layer the host never reads"
        );

        // The filter is by layer index, not a blanket amnesty for a layer-shaped name.
        for name in [
            "model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
        ] {
            let served = tempfile::tempdir().expect("tempdir");
            sparkflash_weights::safetensors::write_checkpoint(
                served.path(),
                &[(
                    sparkflash_weights::TensorSpec {
                        name: name.to_owned(),
                        dtype: sparkflash_weights::DType::U8,
                        shape: vec![1],
                    },
                    vec![0x21],
                )],
                1,
            )
            .expect("the served quantized shard writes");
            let served = sparkflash_weights::safetensors::open_checkpoint(served.path())
                .expect("the served shard opens");
            let refusal = host_reference_storage_quantization_refusal(&architecture, &served)
                .unwrap_or_else(|| panic!("a served U8 tensor {name} is still refused"));
            assert!(
                refusal.contains(name) && refusal.contains("U8"),
                "{refusal}"
            );
        }
    }

    /// A historical document spells the rule as `type`, not `rope_type`. The shared parser must
    /// still see yarn, and the host gate must refuse it instead of running the host unscaled.
    #[test]
    fn a_legacy_type_yarn_is_refused_by_the_host_gate() {
        let document = r#"{
            "model_type": "acme_yarn",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 512,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "yarn", "factor": 4.0},
            "tie_word_embeddings": true,
            "eos_token_id": 2
        }"#;
        let architecture = Architecture::from_json(document).expect("the legacy document parses");
        let refusal = host_reference_rope_refusal(&architecture).expect("yarn is refused");
        assert!(refusal.contains("rope.scaling"), "{refusal}");
        assert!(refusal.contains("yarn"), "{refusal}");
    }

    /// The forward skips a layer with fewer than two rotated dimensions, so a yarn declaration
    /// on a NoPE MLA layer is inert rather than a mismatch. The boot predicate must mirror that
    /// and not refuse a document the host executes correctly.
    #[test]
    fn a_nope_layer_accepts_an_inert_yarn_declaration() {
        let document = r#"{
            "model_type": "acme_nope",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "qk_rope_head_dim": 0,
            "vocab_size": 512,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "yarn", "factor": 4.0},
            "tie_word_embeddings": true,
            "eos_token_id": 2
        }"#;
        let architecture = Architecture::from_json(document).expect("the NoPE document parses");
        assert_eq!(architecture.layers[0].attention.rope.rotated_dims, 0);
        assert_eq!(
            host_reference_rope_refusal(&architecture),
            None,
            "an inert scaling on a NoPE layer must not refuse the host"
        );
    }

    /// A default rule with a non-unit factor is not the forward's plain RoPE: the forward
    /// ignores the factor, so the host would silently run a different position scale.
    #[test]
    fn a_default_rope_with_a_non_unit_factor_is_refused() {
        let document = r#"{
            "model_type": "acme_default_factor",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 512,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": {"rope_type": "default", "factor": 4.0},
            "tie_word_embeddings": true,
            "eos_token_id": 2
        }"#;
        let architecture = Architecture::from_json(document).expect("the document parses");
        let refusal =
            host_reference_rope_refusal(&architecture).expect("a non-unit default is refused");
        assert!(refusal.contains("default with factor 4"), "{refusal}");
    }

    #[test]
    fn the_gemm_class_list_is_the_matrix_product_parts() {
        for part in [
            Part::DenseGemm,
            Part::DenseMlp,
            Part::Moe,
            Part::GroupedMoeGemm,
            Part::LmHead,
        ] {
            assert!(is_gemm_class(part), "{part} is GEMM-class");
        }
        for part in Part::ALL {
            if !is_gemm_class(*part) {
                assert!(
                    !matches!(
                        part,
                        Part::DenseGemm
                            | Part::DenseMlp
                            | Part::Moe
                            | Part::GroupedMoeGemm
                            | Part::LmHead
                    ),
                    "{part} needs an assertion"
                );
            }
        }
    }

    #[test]
    fn the_first_waiver_table_names_the_released_gaps() {
        for waiver in WAIVERS {
            assert!(
                !waiver.measurement.trim().is_empty(),
                "a waiver line must name its evidence"
            );
            assert!(
                waiver.rows.first <= waiver.rows.last,
                "a waiver's row band must be ordered"
            );
        }
        let has = |part: Part, phase: KernelPhase| {
            WAIVERS
                .iter()
                .any(|waiver| waiver.part == Some(part) && waiver.phase.serves(phase))
        };
        // GLM's and Inkling's row-GEMV decode.
        assert!(has(Part::DenseGemm, KernelPhase::Decode));
        assert!(has(Part::DenseMlp, KernelPhase::Decode));
        assert!(has(Part::Moe, KernelPhase::Decode));
        // Dory's decode rows take q4p_dense_rows, the wide tile's few-row body.
        assert!(WAIVERS.iter().any(|waiver| {
            waiver.part == Some(Part::DenseGemm) && waiver.measurement.contains("q4p_dense_rows")
        }));
        // The heads on row GEMVs.
        assert!(has(Part::LmHead, KernelPhase::Both));
        // DeepSeek's three bit dependences cite the documents that hold the numbers.
        for document in [
            "docs/design.md:113",
            "docs/optimization-strategy.md:497",
            "docs/NUMERICS.md:281",
        ] {
            assert!(
                WAIVERS
                    .iter()
                    .any(|waiver| waiver.part.is_none() && waiver.measurement.contains(document)),
                "{document} is cited"
            );
        }
    }

    #[test]
    fn every_catalogue_gemm_class_body_has_a_pass_band() {
        for entry in CATALOGUE {
            for symbol in entry.symbols {
                if !symbol.class.is_gemm() {
                    continue;
                }
                assert!(
                    symbol.rows.first >= 1 && symbol.rows.first <= symbol.rows.last,
                    "{} has an empty row band",
                    symbol.name
                );
                assert!(
                    symbol.phase.serves(KernelPhase::Decode)
                        || symbol.phase.serves(KernelPhase::Prefill),
                    "{} serves no pass",
                    symbol.name
                );
            }
        }
    }
}
