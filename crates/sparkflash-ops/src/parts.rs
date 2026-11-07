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
            .hidden_equals(2560)
            .intermediate_multiple(16)
            .experts_equals(512)
            .top_k_equals(10)
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
            .hidden_equals(2560)
            .intermediate_multiple(16)
            .experts_equals(512)
            .top_k_equals(10)
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
            pre("q4p_dense_rows", &[Precision::Bf16]),
            pre("q4p_gemm_rows", &[Precision::Bf16]),
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
}
