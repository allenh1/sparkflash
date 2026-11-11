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

use crate::plan::{KernelSymbol, Precision, ShapeConstraint};
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
        (Part::Embedding, Image::Qwen4Exp) => ShapeConstraint::ANY.vocab_multiple(128),
        (Part::RmsNorm, Image::Deepseek) => {
            ShapeConstraint::ANY.hidden_equals(5120).hidden_multiple(2)
        }
        (Part::RmsNorm, Image::Qwen4Exp) => ShapeConstraint::ANY.hidden_multiple(2),
        (Part::LmHead, Image::Deepseek) => {
            ShapeConstraint::ANY.hidden_equals(5120).vocab_multiple(128)
        }
        (Part::LmHead, Image::Qwen4Exp) => ShapeConstraint::ANY.vocab_multiple(128),
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
        // The shared dense projection. The b12x and q4p bodies are generic over the hidden and
        // intermediate widths, so this entry is not pinned to one model's 5120.
        (Part::DenseGemm, Image::Deepseek) => ShapeConstraint::ANY,
        (Part::DenseGemm, Image::Qwen4Prefill) => ShapeConstraint::ANY.hidden_equals(2560),
        (Part::DenseGemm, Image::GlmPrefill) => ShapeConstraint::ANY.hidden_equals(4096),
        (Part::DenseGemm, Image::Qwen4Exp) => ShapeConstraint::ANY.hidden_equals(2560),
        (Part::DenseGemm, Image::GlmDecode) => ShapeConstraint::ANY.hidden_equals(4096),
        (Part::Activation, Image::Deepseek) => ShapeConstraint::ANY.hidden_equals(5120),
        (Part::Activation, Image::Qwen4Exp) => ShapeConstraint::ANY.hidden_equals(2560),
        (Part::Activation, Image::Qwen4Prefill) => ShapeConstraint::ANY.hidden_equals(2560),
        (Part::Activation, Image::GlmMoe | Image::GlmDecode) => {
            ShapeConstraint::ANY.hidden_equals(4096)
        }
        (Part::Rope, Image::Deepseek) => ShapeConstraint::ANY.hidden_equals(5120),
        (Part::Rope, Image::Qwen4Exp) => ShapeConstraint::ANY.hidden_equals(2560),
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
        // The shared vision tower's 16-patch kernel exists, but no orchestrated 16-patch tower.
        (Part::Vision, Image::Deepseek) => ShapeConstraint::ANY.patch_sizes(VISION_PATCH_SIZES),
        _ => ShapeConstraint::ANY,
    }
}

/// A symbol accepting exactly these precisions in both passes.
const fn sym(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::both(name, precisions)
}

/// A bulk-prefill symbol accepting exactly these precisions.
const fn pre(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::prefill(name, precisions)
}

/// A token/decode symbol accepting exactly these precisions.
const fn dec(name: &'static str, precisions: &[Precision]) -> KernelSymbol {
    KernelSymbol::decode(name, precisions)
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
            sym("opt_rows_embed_own", &[Precision::Bf16]),
            sym("opt_rows_embed_stream", &[Precision::Bf16]),
            sym("opt_rows_embed_table", &[Precision::Bf16]),
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
            sym("rmsnorm_sm121", &[Precision::Bf16, Precision::F32]),
            sym("opt_rmsnorm_bf16", &[Precision::Bf16, Precision::F32]),
            sym("opt_k_norm_rows", &[Precision::Bf16, Precision::F32]),
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
            sym("cert_lm_head_sm121", &[Precision::Bf16]),
            sym("cert_lm_head_batched_sm121", &[Precision::Bf16]),
            sym("lm_head_reduce_sm121", &[Precision::Bf16]),
            sym("opt_lm_head", &[Precision::Bf16]),
            sym("opt_rows_lm_head", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::LmHead,
        Image::Qwen4Exp,
        &[sym(
            "q4e_argmax_slices",
            &[Precision::Bf16, Precision::Fp8E4M3],
        )],
        false,
    ),
    sm121(
        Part::MlaAttention,
        Image::Deepseek,
        &[
            sym("flash_mla_csa2_sm121", &[Precision::MxFp8, Precision::Bf16]),
            sym("opt_bulk_attn", &[Precision::MxFp8, Precision::Bf16]),
            sym(
                "opt_flash_mla_csa2_seq",
                &[Precision::MxFp8, Precision::Bf16],
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
            sym(
                "flash_mla_csa2_ring_sm121",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "compressed_append_sm121",
                &[Precision::MxFp8, Precision::Bf16],
            ),
            sym(
                "compressed_selected_fill_sm121",
                &[Precision::MxFp8, Precision::Bf16],
            ),
            sym("pool_group_f32", &[Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::SparseIndexer,
        Image::Deepseek,
        &[
            sym(
                "ref_select_sm121",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_ref_select_fast",
                &[Precision::Bf16, Precision::F32, Precision::Fp8E4M3],
            ),
            sym(
                "opt_mma_index_scores",
                &[Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "flash_mla_csa2_seq_sm121",
                &[Precision::MxFp8, Precision::Bf16],
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
            sym("b12x_mxfp8_dense_sm121", &[Precision::MxFp8]),
            sym("b12x_mxfp8_dense_split_k_sm121", &[Precision::MxFp8]),
            sym("swiglu_mxfp8_quant_sm121", &[Precision::MxFp8]),
            sym("opt_mma_bf16a_bf16o", &[Precision::Bf16]),
            sym("opt_mma_f32a_f32o", &[Precision::F32]),
            sym("opt_mma_fp8_bf16o", &[Precision::Fp8E4M3]),
            sym("shared_swiglu_f32", &[Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::DenseMlp,
        Image::BLOCK_PREFILL,
        &[
            pre("glm_prefill_bf16_mma", &[Precision::Bf16]),
            pre("glm_prefill_bf16_mma_bf16a", &[Precision::Bf16]),
            pre("glm_prefill_bf16_wide", &[Precision::Bf16]),
            pre("glm_prefill_bf16_wide_bf16a", &[Precision::Bf16]),
            pre("glm_prefill_fp8b128_mma", &[Precision::Fp8E4M3]),
            pre("glm_prefill_fp8b128_mma_bf16a", &[Precision::Fp8E4M3]),
            pre("glm_prefill_q8_mma", &[Precision::Fp8E4M3]),
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
            pre("q4p_dense_mma", &[Precision::Bf16, Precision::Fp8E4M3]),
            pre("q4p_dense_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_epilogue", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::DenseMlp,
        Image::TOKEN,
        &[
            sym("q4e_gemm_bf16", &[Precision::Bf16, Precision::F32]),
            sym("q4e_shared_gate_up", &[Precision::Bf16, Precision::F32]),
            // The same body is the Activation part's; it runs over the projected chunk in
            // prefill and over a token's row in decode, so both entries describe one phase.
            sym("q4e_swiglu", &[Precision::Bf16, Precision::F32]),
            dec("q4e_quantize_e4m3_rows", &[Precision::Bf16]),
            dec("q4e_gemv_tiled_bf16", &[Precision::Bf16]),
            dec("q4e_gemv_tiled_e4m3", &[Precision::Fp8E4M3]),
            // The Qwen3.5/3.6 FP8 release's block-scaled dense projections (BF16 tile scales).
            dec("q4e_gemv_tiled_e4m3_tile", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::DenseMlp,
        Image::DECODE,
        &[
            dec(
                "q4e_gemv_tiled_bf16_parts",
                &[Precision::Bf16, Precision::F32],
            ),
            dec("q4e_gemv_tiled_e4m3_block", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    // The shared dense projection, distinct from the MLP part so its b12x body is not pinned to
    // one model's hidden width.
    sm121(
        Part::DenseGemm,
        Image::Deepseek,
        &[
            sym("b12x_mxfp8_dense_sm121", &[Precision::MxFp8]),
            sym("b12x_mxfp8_dense_split_k_sm121", &[Precision::MxFp8]),
            sym("b12x_mxfp8_dense_par_k_sm121", &[Precision::MxFp8]),
            sym("b12x_mxfp8_dense_tile_sm121", &[Precision::MxFp8]),
            sym("b12x_mxfp8_dense_warp_sm121", &[Precision::MxFp8]),
            dec("b12x_mxfp8_dense_m1_sm121", &[Precision::MxFp8]),
            sym("b12x_bf16a_mxfp8w_dense_sm121", &[Precision::MxFp8]),
            sym("b12x_bf16a_mxfp8w_wo_a_sm121", &[Precision::MxFp8]),
            sym("b12x_mxfp8_wo_a_sm121", &[Precision::MxFp8]),
            dec("b12x_mxfp8_gemv_m1_sm121", &[Precision::MxFp8]),
            dec("b12x_hostw_gemv_m1_bf16w_sm121", &[Precision::Bf16]),
            dec("b12x_hostw_gemv_m1_f32w_f32o_sm121", &[Precision::F32]),
            sym("opt_mma_bf16a_bf16o", &[Precision::Bf16]),
            sym("opt_mma_f32a_f32o", &[Precision::F32]),
            sym("opt_mma_fp8_bf16o", &[Precision::Fp8E4M3]),
            sym("opt_mma_fp8_f32o", &[Precision::Fp8E4M3]),
        ],
        true,
    ),
    sm121(
        Part::DenseGemm,
        Image::PREFILL,
        &[
            pre("q4p_dense_mma", &[Precision::Bf16, Precision::Fp8E4M3]),
            pre("q4p_dense_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_dense_rows", &[Precision::Bf16]),
            pre("q4p_gemm_rows", &[Precision::Bf16]),
            // The hyper connection's up projection and stream mix for a few rows.
            pre("q4p_hyper_up_rows", &[Precision::Bf16]),
            pre("q4p_split_bf16", &[Precision::Bf16]),
        ],
        false,
    ),
    sm121(
        Part::DenseGemm,
        Image::BLOCK_PREFILL,
        &[
            pre("glm_prefill_bf16_mma", &[Precision::Bf16]),
            pre("glm_prefill_bf16_mma_bf16a", &[Precision::Bf16]),
            pre("glm_prefill_bf16_wide", &[Precision::Bf16]),
            pre("glm_prefill_bf16_wide_bf16a", &[Precision::Bf16]),
            pre("glm_prefill_fp8b128_mma", &[Precision::Fp8E4M3]),
            pre("glm_prefill_fp8b128_mma_bf16a", &[Precision::Fp8E4M3]),
            pre("glm_prefill_q8_mma", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::DenseGemm,
        Image::TOKEN,
        &[
            sym("q4e_gemm_bf16", &[Precision::Bf16, Precision::F32]),
            dec("q4e_gemv_bf16", &[Precision::Bf16]),
            dec("q4e_gemv4_bf16", &[Precision::Bf16]),
            dec("q4e_gemv_tiled_bf16", &[Precision::Bf16]),
            dec("q4e_gemv_tiled_e4m3", &[Precision::Fp8E4M3]),
            // The Qwen3.5/3.6 FP8 release's block-scaled dense projections (BF16 tile scales).
            dec("q4e_gemv_tiled_e4m3_tile", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::DenseGemm,
        Image::DECODE,
        &[
            dec(
                "q4e_gemv_tiled_bf16_parts",
                &[Precision::Bf16, Precision::F32],
            ),
            dec("q4e_gemv_tiled_e4m3_block", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::Activation,
        Image::Deepseek,
        &[
            sym("swiglu_mxfp8_quant_sm121", &[Precision::MxFp8]),
            sym("shared_swiglu_f32", &[Precision::F32]),
            sym("opt_rows_shared_swiglu", &[Precision::F32]),
        ],
        true,
    ),
    sm121(
        Part::Activation,
        Image::TOKEN,
        &[sym("q4e_swiglu", &[Precision::Bf16, Precision::F32])],
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
            sym("opt_rope_heads_bf16", &[Precision::Bf16]),
            sym("rope_rotate_rows_bf16", &[Precision::Bf16]),
            sym("rope_rotate_bf16", &[Precision::Bf16]),
            sym("opt_rows_inv_rope", &[Precision::Bf16]),
            sym("kv_norm_rope_bf16", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::Rope,
        Image::TOKEN,
        &[sym("q4e_rope", &[Precision::Bf16])],
        false,
    ),
    sm121(
        Part::Quantize,
        Image::Deepseek,
        &[
            sym("opt_bulk_fp4_pack", &[Precision::F32]),
            sym("mxfp8_quantize_activations_sm121", &[Precision::Bf16]),
        ],
        true,
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
            sym("b12x_mxfp4_moe_sm121", &[Precision::NvFp4]),
            sym("b12x_mxfp4_moe_stored_sm121", &[Precision::NvFp4]),
            sym("opt_moe_gate_up_sm121", &[Precision::NvFp4]),
            sym("opt_moe_gate_up_k3_sm121", &[Precision::NvFp4]),
            sym("opt_moe_down_combine_sm121", &[Precision::NvFp4]),
            sym("opt_moe_down_combine_k3_sm121", &[Precision::NvFp4]),
            sym("opt_mma_moe_gate_up", &[Precision::NvFp4]),
            sym("opt_mma_moe_down", &[Precision::NvFp4]),
            sym("moe_expert_stage_sm121", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::Moe,
        Image::ROUTED_EXPERTS,
        &[
            // These bodies decode packed E2M1 codes, so their operand class is NVFP4 alone.
            // A BF16 or F32 document has no routed-expert body here.
            dec("q4e_expert_gate_up", &[Precision::NvFp4]),
            dec("q4e_expert_down", &[Precision::NvFp4]),
            // The token FP8 experts are the short-batch path (a decode step and a chunk below
            // GEMM_MIN_BATCH); the grouped FP8 tile in the PREFILL entry below is the chunk path.
            sym("q4e_expert_gate_up_fp8", &[Precision::Fp8E4M3]),
            sym("q4e_expert_down_fp8", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    // The genuine FP8 grouped prefill bodies. `q4p_moe_gate_up` and `q4p_moe_down` are the NVFP4
    // grouped bodies, and `q4p_moe_rows_*` are their panel form: `q4p_moe_body` reads packed
    // E2M1 codes and has no FP8 or BF16 weight mode, so they claim `NvFp4` only. A Qwen FP8
    // chunk resolves the `_fp8` symbols in this image.
    sm121(
        Part::Moe,
        Image::PREFILL,
        &[
            pre("q4p_moe_gate_up", &[Precision::NvFp4]),
            pre("q4p_moe_down", &[Precision::NvFp4]),
            pre("q4p_moe_rows_gate_up", &[Precision::NvFp4]),
            pre("q4p_moe_rows_down", &[Precision::NvFp4]),
            pre("q4p_moe_gate_up_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_moe_swiglu_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_moe_down_fp8", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::Moe,
        Image::BLOCK_ROUTED_EXPERTS,
        &[
            pre("glm_moe_fp8b128_mma", &[Precision::Fp8E4M3]),
            // `_bf16a` names the BF16 activations and accumulator, not the stored weights: the
            // body reads E4M3 codes with a 128x128 BF16 tile scale, so its operand class is FP8.
            // A BF16 document therefore has no routed-expert body in either pass.
            pre("glm_moe_fp8b128_mma_bf16a", &[Precision::Fp8E4M3]),
            pre("glm_moe_q8_mma", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::Moe,
        Image::DECODE,
        &[dec("q4e_expert_gemv_e4m3_block", &[Precision::Fp8E4M3])],
        false,
    ),
    // Router, combine and grouped GEMM are their own catalogue identities. Sharing Part::Moe
    // would either duplicate a (part, image) entry or apply the MoE block's shape rule to the
    // combine, which is what made GLM's combine unreachable.
    sm121(
        Part::MoeRouter,
        Image::Deepseek,
        &[
            sym("moe_router_sm121", &[Precision::Bf16, Precision::F32]),
            sym("moe_router_topk_sm121", &[Precision::Bf16, Precision::F32]),
            sym("opt_router_sm121", &[Precision::Bf16]),
            sym("opt_router_e128k3_sm121", &[Precision::Bf16]),
            sym("opt_mma_router_dots", &[Precision::Bf16]),
            sym("opt_mma_router_topk", &[Precision::Bf16]),
        ],
        true,
    ),
    sm121(
        Part::MoeRouter,
        Image::ROUTED_EXPERTS,
        &[sym("q4e_router", &[Precision::Bf16, Precision::F32])],
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
            sym(
                "moe_routed_combine_sm121",
                &[Precision::Bf16, Precision::F32],
            ),
            sym("opt_moe_combine_sm121", &[Precision::Bf16, Precision::F32]),
            sym(
                "opt_moe_combine_k3_sm121",
                &[Precision::Bf16, Precision::F32],
            ),
            sym("opt_mma_moe_combine", &[Precision::NvFp4]),
        ],
        true,
    ),
    sm121(
        Part::MoeCombine,
        Image::ROUTED_EXPERTS,
        &[sym("q4e_moe_combine", &[Precision::Bf16, Precision::F32])],
        false,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::Deepseek,
        &[
            sym("b12x_mxfp4_moe_sm121", &[Precision::NvFp4]),
            sym("b12x_mxfp4_moe_stored_sm121", &[Precision::NvFp4]),
            sym("opt_moe_gate_up_sm121", &[Precision::NvFp4]),
            sym("opt_moe_down_combine_sm121", &[Precision::NvFp4]),
            sym("opt_mma_moe_gate_up", &[Precision::NvFp4]),
            sym("opt_mma_moe_down", &[Precision::NvFp4]),
        ],
        true,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::ROUTED_EXPERTS,
        &[
            // These bodies decode packed E2M1 codes, so their operand class is NVFP4 alone.
            // A BF16 or F32 document has no routed-expert body here.
            dec("q4e_expert_gate_up", &[Precision::NvFp4]),
            dec("q4e_expert_down", &[Precision::NvFp4]),
            // The token FP8 bodies serve a decode step and a short prefill chunk; a real chunk
            // resolves the grouped `q4p_moe_*_fp8` symbols below.
            sym("q4e_expert_gate_up_fp8", &[Precision::Fp8E4M3]),
            sym("q4e_expert_down_fp8", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::PREFILL,
        &[
            pre("q4p_moe_gate_up", &[Precision::NvFp4]),
            pre("q4p_moe_down", &[Precision::NvFp4]),
            pre("q4p_moe_rows_gate_up", &[Precision::NvFp4]),
            pre("q4p_moe_rows_down", &[Precision::NvFp4]),
            pre("q4p_moe_gate_up_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_moe_swiglu_fp8", &[Precision::Fp8E4M3]),
            pre("q4p_moe_down_fp8", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::BLOCK_ROUTED_EXPERTS,
        &[
            pre("glm_moe_fp8b128_mma", &[Precision::Fp8E4M3]),
            // `_bf16a` names the BF16 activations and accumulator, not the stored weights: the
            // body reads E4M3 codes with a 128x128 BF16 tile scale, so its operand class is FP8.
            // A BF16 document therefore has no routed-expert body in either pass.
            pre("glm_moe_fp8b128_mma_bf16a", &[Precision::Fp8E4M3]),
            pre("glm_moe_q8_mma", &[Precision::Fp8E4M3]),
        ],
        false,
    ),
    sm121(
        Part::GroupedMoeGemm,
        Image::DECODE,
        &[dec("q4e_expert_gemv_e4m3_block", &[Precision::Fp8E4M3])],
        false,
    ),
    sm121(
        Part::MoePlan,
        Image::MoePlan,
        &[sym("moe_plan", &[Precision::Bf16])],
        true,
    ),
    sm121(
        Part::Collective,
        Image::Rocenante,
        &[sym(
            "rocenante_oneshot_allreduce_bf16_tp4",
            &[Precision::Bf16],
        )],
        true,
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
            sym(
                "mega_mhc_fused_bf16_sm121",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "mega_mhc_fused_bf16_batched_sm121",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "mega_mhc_init_sm121",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_mhc_seam",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_mhc_post",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_bulk_mhc_tail",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_bulk_mhc_partials",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_bulk_mhc_post_partials",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym("mhc_entry_bf16", &[Precision::Bf16]),
            sym("mhc_ffn_seams", &[Precision::Bf16]),
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
            sym(
                "engram_embed_sm121",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "engram_gate_write_sm121",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "engram_wkv_sm121",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym("engram_write_bf16", &[Precision::Bf16]),
            sym("engram_narrow_bf16", &[Precision::Bf16]),
            sym("engram_widen_f32", &[Precision::F32]),
            sym(
                "opt_bulk_engram_embed",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_bulk_engram_post",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym("opt_engram_gate_write", &[Precision::Bf16]),
            sym("opt_engram_wkv", &[Precision::Bf16]),
            sym("opt_rows_engram_post", &[Precision::Bf16]),
            sym("opt_rows_engram_wkv", &[Precision::Bf16]),
            sym("opt_rows_engram_wkv_s", &[Precision::Bf16]),
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
            sym("opt_draft_fold_prob", &[Precision::NvFp4]),
            sym("opt_draft_pick_prob", &[Precision::NvFp4]),
            sym("opt_draft_embed_blocks", &[Precision::NvFp4]),
            sym("opt_draft_fold_seqs", &[Precision::NvFp4]),
            sym("opt_draft_pick_seqs", &[Precision::NvFp4]),
            sym("opt_draft_attn", &[Precision::NvFp4]),
            sym("opt_draft_logits", &[Precision::NvFp4]),
            sym("opt_draft_markov", &[Precision::NvFp4]),
            sym("opt_draft_markov_seqs", &[Precision::NvFp4]),
            sym("opt_draft_markov_seqs_mma", &[Precision::NvFp4]),
            sym("opt_draft_head_fp8", &[Precision::Fp8E4M3]),
            sym("opt_draft_head_fp8t", &[Precision::Fp8E4M3]),
            sym("opt_draft_logits_fp8", &[Precision::Fp8E4M3]),
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
        ],
        false,
    ),
    sm121(
        Part::Vision,
        Image::Deepseek,
        &[
            sym(
                "opt_vision_patchify",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_patchify_16",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_gemm",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_rmsnorm",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_rope",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_attn",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_swiglu",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_unfold",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_span_rows",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
            sym(
                "opt_vision_route",
                &[Precision::MxFp8, Precision::Bf16, Precision::Fp8E4M3],
            ),
        ],
        true,
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
            sym("dory_mma_wide", &[Precision::Bf16]),
            sym("dory_mma_wide4", &[Precision::Bf16]),
        ],
        false,
    ),
];

/// The catalogue entries that implement part on target, in catalogue order.
pub fn kernels(part: Part, target: Target) -> impl Iterator<Item = &'static PartKernel> {
    CATALOGUE
        .iter()
        .filter(move |entry| entry.part == part && entry.target == target)
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
        .filter(|part| !matches!(part, Part::Mtp | Part::Vision))
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
/// dynamic route, the arm's own load and the web browser agree on.
///
/// # Errors
/// Why no device arm serves the checkpoint, so the host reference is its executor.
pub fn dynamic_device_arm(architecture: &Architecture) -> Result<&'static str, String> {
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
        return Err(format!(
            "this build has no device arm for the {family} family"
        ));
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
        return Some(format!(
            "the checkpoint names parts the dynamic host path does not execute: {}",
            names.join(", ")
        ));
    }
    host_reference_rope_refusal(architecture)
}

/// The first layer whose rotary scaling rule the host reference does not apply.
///
/// The host forward applies plain RoPE and a positive `linear` factor only. A `yarn`, `dynamic`
/// or other rule is accepted by the shared architecture and the plan, but the reference
/// arithmetic would silently compute unscaled positions; refuse it at boot instead of failing
/// the first request inside `HostSession::forward`.
#[must_use]
pub fn host_reference_rope_refusal(architecture: &Architecture) -> Option<String> {
    for layer in &architecture.layers {
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
        .filter(|part| *part != Part::Vision)
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

    /// Qwen3.6-35B-A3B is a covered host family: its vision tower and MTP stage are stored but
    /// not executed, and its 128x128 E4M3 weights are tile-scaled rather than opaque codes.
    #[test]
    fn qwen36_is_a_host_reference_family() {
        const QWEN36: &str =
            include_str!("../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json");
        let architecture = Architecture::from_json(QWEN36).expect("the Qwen3.6 document parses");
        assert_eq!(architecture.family, "qwen3_5_moe");
        assert!(
            architecture.parts().contains(Part::Vision),
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
}
