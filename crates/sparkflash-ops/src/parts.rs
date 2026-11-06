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
//! that claim means. [CATALOGUE] names one entry per `(part, target, image)` triple, so a part
//! served by several model images lists one entry for each. A part is carried for a target when
//! the catalogue has at least one entry for it ([carried]), and [image_available] narrows that
//! to the entries whose image this build actually embeds.
//!
//! # Canonical entries
//!
//! [PartKernel::canonical] marks the implementation a target brings up first. Exactly one entry
//! per `(part, target)` is canonical. When DeepSeek implements a part, its entry is canonical
//! and comes first: DeepSeek's kernels are the proven implementation and a Qwen or GLM entry
//! must never silently become the default for a shared part.
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
//! Only [Target::Sm121] has entries today: the kernels under `kernels/` are named
//! `*_sm121`, so no part is claimed for sm100 or sm120 until its kernels are compiled for
//! them.

use sparkflash_kernels::images::Image;
use sparkflash_model::Architecture;

use crate::plan::{KernelPhase, PrecisionMask, ShapeConstraint};
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
    /// The image's kernel symbols that implement the part.
    pub symbols: &'static [&'static str],
    /// Whether this is the implementation a new target brings up first.
    ///
    /// Exactly one entry per `(part, target)` is canonical. A DeepSeek entry is always the
    /// canonical one for a part DeepSeek implements.
    pub canonical: bool,
    /// The shapes this implementation accepts. [ShapeConstraint::ANY] is an unproven rule.
    pub shape: ShapeConstraint,
    /// The operand precisions this implementation accepts.
    pub precisions: PrecisionMask,
    /// The pass this implementation belongs to.
    pub phase: KernelPhase,
}

/// The pass an image implements.
const fn phase_for(image: Image) -> KernelPhase {
    match image {
        Image::GlmPrefill | Image::GlmAttention | Image::GlmMoe | Image::Qwen4Prefill => {
            KernelPhase::Prefill
        }
        Image::GlmDecode
        | Image::Qwen4Exp
        | Image::Qwen4Attn
        | Image::Qwen4Moe
        | Image::Qwen4Scan
        | Image::Dflash2
        // The planner has no pass of its own: both a chunk's prefill and a token step's
        // decode call it before their expert launches.
        | Image::MoePlan => KernelPhase::Decode,
        Image::Deepseek | Image::Rocenante => KernelPhase::Both,
    }
}

/// The shape rule of one catalogue entry.
///
/// This is the model-independent part of kernel selection: a kernel that only works on whole
/// 16-wide heads, or only on group sizes 1/2/4/6/8, says so here and the resolver refuses the
/// checkpoint before a weight is read. A rule this tree has not proven stays `ANY` rather than
/// pretending to a constraint.
const fn shape_for(part: Part, _image: Image) -> ShapeConstraint {
    match part {
        Part::RmsNorm => ShapeConstraint::ANY.hidden_multiple(2),
        Part::LmHead | Part::Embedding => ShapeConstraint::ANY.vocab_multiple(128),
        Part::MlaAttention => ShapeConstraint::ANY
            .head_dim_multiple(16)
            .positions(0, 1 << 21),
        Part::GqaAttention => ShapeConstraint::ANY.head_dim_multiple(16),
        Part::GdnScan | Part::KdaScan => ShapeConstraint::ANY.head_dim_multiple(16),
        Part::Csa2Compression => ShapeConstraint::ANY.min_ratio(1).positions(0, 1 << 21),
        Part::SparseIndexer => ShapeConstraint::ANY.head_dim_multiple(16),
        Part::DenseMlp | Part::Moe => ShapeConstraint::ANY.intermediate_multiple(16),
        Part::HyperConnection | Part::ManifoldHyperConnection => {
            ShapeConstraint::ANY.streams_multiple(1)
        }
        Part::Engram | Part::Ple | Part::Mtp | Part::Vision => ShapeConstraint::ANY,
    }
}

/// The precisions one catalogue entry accepts.
///
/// The entries are deliberately permissive until a kernel's precision contract is pinned next
/// to its launch; the resolver still checks the field, so tightening one entry is local.
const fn precisions_for(_part: Part, _image: Image) -> PrecisionMask {
    PrecisionMask::ALL
}

/// A catalogue entry on the one target the kernels are built for today.
const fn sm121(
    part: Part,
    image: Image,
    symbols: &'static [&'static str],
    canonical: bool,
) -> PartKernel {
    PartKernel {
        part,
        target: Target::Sm121,
        image,
        symbols,
        canonical,
        shape: shape_for(part, image),
        precisions: precisions_for(part, image),
        phase: phase_for(image),
    }
}

/// Every `(part, target, image)` implementation this tree catalogues.
///
/// Entries are grouped by [Part] in declaration order. Within a part the canonical entry comes
/// first, and a DeepSeek entry is canonical for every part DeepSeek implements.
pub const CATALOGUE: &[PartKernel] = &[
    sm121(
        Part::Embedding,
        Image::Deepseek,
        &[
            "opt_rows_embed_own",
            "opt_rows_embed_stream",
            "opt_rows_embed_table",
        ],
        true,
    ),
    sm121(Part::Embedding, Image::Qwen4Exp, &["q4e_embed"], false),
    sm121(
        Part::RmsNorm,
        Image::Deepseek,
        &["rmsnorm_sm121", "opt_rmsnorm_bf16"],
        true,
    ),
    sm121(Part::RmsNorm, Image::Qwen4Exp, &["q4e_rmsnorm"], false),
    sm121(
        Part::LmHead,
        Image::Deepseek,
        &[
            "cert_lm_head_sm121",
            "lm_head_reduce_sm121",
            "opt_lm_head",
            "opt_rows_lm_head",
        ],
        true,
    ),
    sm121(Part::LmHead, Image::Qwen4Exp, &["q4e_argmax_slices"], false),
    sm121(
        Part::MlaAttention,
        Image::Deepseek,
        &["flash_mla_csa2_sm121", "opt_bulk_attn"],
        true,
    ),
    sm121(
        Part::MlaAttention,
        Image::GlmAttention,
        &[
            "glm_prefill_mla_attend",
            "glm_prefill_mla_attend_bf16a",
            "glm_prefill_heads_bf16_mma",
            "glm_prefill_heads_fp8b128_mma",
        ],
        false,
    ),
    sm121(
        Part::MlaAttention,
        Image::GlmDecode,
        &[
            "q4e_glm_mla_cache",
            "q4e_glm_mla_cache_fp8",
            "q4e_glm_mla_absorb",
            "q4e_glm_mla_attend",
            "q4e_glm_mla_scores",
            "q4e_glm_mla_softmax",
            "q4e_glm_mla_mix",
            "q4e_glm_mla_project",
        ],
        false,
    ),
    sm121(
        Part::Csa2Compression,
        Image::Deepseek,
        &[
            "flash_mla_csa2_ring_sm121",
            "compressed_append_sm121",
            "compressed_selected_fill_sm121",
            "pool_group_f32",
        ],
        true,
    ),
    sm121(
        Part::SparseIndexer,
        Image::Deepseek,
        &[
            "ref_select_sm121",
            "opt_ref_select_fast",
            "opt_mma_index_scores",
        ],
        true,
    ),
    sm121(
        Part::SparseIndexer,
        Image::Qwen4Attn,
        &["q4e_qsa_pool", "q4e_qsa_scores", "q4e_qsa_pick"],
        false,
    ),
    sm121(
        Part::SparseIndexer,
        Image::GlmDecode,
        &[
            "q4e_glm_pool",
            "q4e_glm_index_scores",
            "q4e_glm_pick",
            "q4e_layer_norm",
        ],
        false,
    ),
    sm121(
        Part::GqaAttention,
        Image::Qwen4Attn,
        &["q4e_attn_group", "q4e_attn_group6", "q4e_attn"],
        true,
    ),
    sm121(
        Part::GdnScan,
        Image::Qwen4Scan,
        &[
            "q4p_gdn_conv",
            "q4p_gdn_conv_state",
            "q4p_gdn_prep",
            "q4p_gdn_scan",
            "q4p_gdn_gate",
        ],
        true,
    ),
    sm121(
        Part::GdnScan,
        Image::Qwen4Exp,
        &["q4e_gdn_conv", "q4e_gdn_recur"],
        false,
    ),
    sm121(
        Part::KdaScan,
        Image::Qwen4Scan,
        &["q4p_kda_prep", "q4p_kda_scan", "q4p_gdn_gate"],
        true,
    ),
    sm121(Part::KdaScan, Image::Qwen4Exp, &["q4e_kda_decay"], false),
    sm121(
        Part::DenseMlp,
        Image::GlmPrefill,
        &["glm_prefill_bf16_mma"],
        true,
    ),
    sm121(
        Part::DenseMlp,
        Image::Qwen4Exp,
        &["q4e_shared_gate_up", "q4e_swiglu"],
        false,
    ),
    sm121(
        Part::DenseMlp,
        Image::GlmDecode,
        &[
            "q4e_gemv_tiled_bf16_parts",
            "q4e_gemv_tiled_e4m3_block",
            "q4e_swiglu_limited",
        ],
        false,
    ),
    sm121(
        Part::Moe,
        Image::Deepseek,
        &[
            "moe_router_sm121",
            "moe_router_topk_sm121",
            "b12x_mxfp4_moe_sm121",
            "b12x_mxfp4_moe_stored_sm121",
            "moe_routed_combine_sm121",
            "opt_mma_moe_gate_up",
            "opt_mma_moe_down",
            "opt_mma_moe_combine",
        ],
        true,
    ),
    sm121(
        Part::Moe,
        Image::Qwen4Moe,
        &[
            "q4e_router",
            "q4e_expert_gate_up",
            "q4e_expert_down",
            "q4e_expert_gate_up_fp8",
            "q4e_expert_down_fp8",
            "q4e_moe_combine",
        ],
        false,
    ),
    sm121(
        Part::Moe,
        Image::Qwen4Prefill,
        &["q4p_moe_gate_up", "q4p_moe_down"],
        false,
    ),
    sm121(
        Part::Moe,
        Image::GlmMoe,
        &[
            "glm_moe_fp8b128_mma",
            "glm_moe_fp8b128_mma_bf16a",
            "glm_moe_swiglu",
            "glm_moe_swiglu_q8",
            "glm_moe_q8_mma",
        ],
        false,
    ),
    sm121(
        Part::Moe,
        Image::GlmDecode,
        &["q4e_noaux_router", "q4e_expert_gemv_e4m3_block"],
        false,
    ),
    sm121(
        Part::HyperConnection,
        Image::Deepseek,
        &[
            "mega_mhc_fused_bf16_sm121",
            "mega_mhc_init_sm121",
            "opt_mhc_seam",
            "opt_mhc_post",
        ],
        true,
    ),
    sm121(
        Part::HyperConnection,
        Image::Qwen4Exp,
        &[
            "q4e_hyper_down",
            "q4e_hyper_up",
            "q4e_hyper_mix",
            "q4e_hyper_inject",
        ],
        false,
    ),
    sm121(
        Part::ManifoldHyperConnection,
        Image::GlmDecode,
        &["q4e_mhc_in", "q4e_mhc_post"],
        true,
    ),
    sm121(
        Part::Engram,
        Image::Deepseek,
        &[
            "engram_embed_sm121",
            "engram_gate_write_sm121",
            "engram_wkv_sm121",
            "engram_write_bf16",
            "engram_narrow_bf16",
            "engram_widen_f32",
            "opt_bulk_engram_embed",
            "opt_bulk_engram_post",
        ],
        true,
    ),
    sm121(
        Part::Ple,
        Image::Qwen4Exp,
        &["q4e_ple_gate", "q4e_ple_conv"],
        true,
    ),
    sm121(
        Part::Mtp,
        Image::Deepseek,
        &[
            "opt_draft_fold_prob",
            "opt_draft_pick_prob",
            "opt_draft_embed_blocks",
            "opt_draft_fold_seqs",
            "opt_draft_pick_seqs",
        ],
        true,
    ),
    sm121(
        Part::Mtp,
        Image::Dflash2,
        &[
            "q4d_capture",
            "q4d_block_embed",
            "q4d_conv",
            "q4d_kv_ring",
            "q4d_attn",
            "q4d_topk_partial",
            "q4d_topk_final",
            "q4d_select",
        ],
        false,
    ),
    sm121(
        Part::Vision,
        Image::Deepseek,
        &[
            "opt_vision_patchify",
            "opt_vision_gemm",
            "opt_vision_rmsnorm",
            "opt_vision_rope",
            "opt_vision_attn",
            "opt_vision_swiglu",
            "opt_vision_unfold",
            "opt_vision_span_rows",
            "opt_vision_route",
        ],
        true,
    ),
];

/// The catalogue entries that implement `part` on `target`, in catalogue order.
pub fn kernels(part: Part, target: Target) -> impl Iterator<Item = &'static PartKernel> {
    CATALOGUE
        .iter()
        .filter(move |entry| entry.part == part && entry.target == target)
}

/// The parts the catalogue carries for `target`.
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

/// Whether any catalogued image for `part` is embedded in this build.
///
/// This is stricter than [carried]: a part can be carried by an image that is loaded at run
/// time rather than embedded, which is the case for [Image::Deepseek].
#[must_use]
pub fn image_available(part: Part, target: Target) -> bool {
    kernels(part, target).any(|entry| entry.image.available())
}

/// The parts `architecture` needs that the catalogue does not carry for `target`.
///
/// This is the catalogue-level answer: a part counts when any entry implements it, whether or
/// not this build embeds the entry's image. [missing_loadable] is the serving answer.
#[must_use]
pub fn missing(architecture: &Architecture, target: Target) -> Vec<Part> {
    architecture.missing(&carried(target))
}

/// The environment variable naming the run-time cubin directory the DeepSeek arm loads.
pub const CUBIN_DIR_ENV: &str = "SPARKFLASH_CUBINS";

/// Whether `image` can be loaded by this process.
///
/// An embedded image answers for itself. [Image::Deepseek] is the shared PTX/cubin set loaded at
/// run time, so it is loadable only when the process was pointed at that directory.
fn image_loadable(image: Image, cubin_dir: bool) -> bool {
    image.available() || (image == Image::Deepseek && cubin_dir)
}

/// The parts the catalogue can load for `target`, using `lookup` for the environment.
///
/// A part is loadable when at least one of its catalogue entries names an image this process can
/// load: an embedded image ([Image::available]) or the DeepSeek arm's run-time image when
/// [`CUBIN_DIR_ENV`] names a non-empty directory.
#[must_use]
pub fn loadable(target: Target, lookup: impl Fn(&str) -> Option<String>) -> PartSet {
    let cubin_dir = lookup(CUBIN_DIR_ENV).is_some_and(|value| !value.trim().is_empty());
    CATALOGUE
        .iter()
        .filter(|entry| entry.target == target && image_loadable(entry.image, cubin_dir))
        .map(|entry| entry.part)
        .collect()
}

/// The parts `architecture` needs that this build can actually load for `target`.
///
/// This is [missing] narrowed by [loadable]: a part whose only catalogued image is the DeepSeek
/// run-time image is missing until [`CUBIN_DIR_ENV`] names that directory.
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
                assert!(
                    entry.symbols.iter().all(|symbol| !symbol.is_empty()),
                    "{part} names an empty symbol"
                );
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
