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

//! The released configs must plan and resolve without a device.

use sparkflash_kernels::images::Image;
use sparkflash_model::{Architecture, Part, Target};
use sparkflash_ops::OpPlan;
use sparkflash_ops::engine_plan::{EnginePlan, PartRequirement, PrecisionPolicy};
use sparkflash_ops::gemm::GroupedMoeGemmPlan;
use sparkflash_ops::plan::{
    GroupedFp8Body, KernelId, KernelPhase, ModelFamily, PartGeometry, Precision, StreamId,
};
use sparkflash_ops::resolver::{Availability, KernelResolver, PlanContext, ResolvedKernel};
use sparkflash_ops::tensor_plan::{ArchitectureTensorPlan, Role};

const DEEPSEEK: &str =
    include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
const GLM: &str = include_str!("../../../models/configs/glm53_flash.hf.json");
const QWEN_NVFP4: &str =
    include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");
const QWEN_FP8: &str =
    include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_fp8.hf.json");
const QWEN36_FP8: &str =
    include_str!("../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json");

/// The GLM release with its quantization block removed: the same architecture at BF16.
fn glm_bf16() -> String {
    let mut value: serde_json::Value = serde_json::from_str(GLM).expect("GLM json");
    value
        .as_object_mut()
        .expect("GLM object")
        .remove("quantization_config");
    value.to_string()
}

/// The Qwen NVFP4 release with its quantization block removed: the same architecture at BF16.
fn qwen_bf16() -> String {
    let mut value: serde_json::Value = serde_json::from_str(QWEN_NVFP4).expect("Qwen json");
    value
        .as_object_mut()
        .expect("Qwen object")
        .remove("quantization_config");
    value.to_string()
}

/// One released document, the worlds its arm's kernels serve, and the producer correction its
/// architecture bridge applies before the plan reads the typed formats.
struct Released {
    name: &'static str,
    text: String,
    /// Worlds whose kernels this document resolves. Empty means the document is an explicit
    /// boot-check refusal: GLM BF16 has no BF16 routed-expert body in either pass.
    worlds: &'static [usize],
    /// The arm's architecture correction. A no-op for every document but the Qwen FP8 ones:
    /// Qwen3.8's global `fp8` flag quantises only the routed experts, and the Qwen arm computes
    /// Qwen3.6's E4M3 tiles weight-only.
    correct: fn(&mut Architecture),
}

fn unchanged(_: &mut Architecture) {}

fn released() -> Vec<Released> {
    vec![
        Released {
            name: "DeepSeek",
            text: DEEPSEEK.to_owned(),
            worlds: &[4],
            correct: unchanged,
        },
        Released {
            name: "GLM FP8",
            text: GLM.to_owned(),
            worlds: &[1, 2, 4],
            correct: unchanged,
        },
        Released {
            name: "GLM BF16",
            text: glm_bf16(),
            worlds: &[],
            correct: unchanged,
        },
        Released {
            name: "Qwen BF16",
            text: qwen_bf16(),
            worlds: &[],
            correct: unchanged,
        },
        Released {
            name: "Qwen NVFP4",
            text: QWEN_NVFP4.to_owned(),
            worlds: &[1, 2, 4],
            correct: unchanged,
        },
        Released {
            name: "Qwen FP8",
            text: QWEN_FP8.to_owned(),
            worlds: &[1, 2, 4],
            // The Qwen FP8 arm's bridge calls this; ops cannot depend on a model crate, so the
            // shared operation is applied here too and the plan sees the release's real formats.
            correct: Architecture::experts_only_fp8,
        },
        Released {
            name: "Qwen3.6 FP8",
            text: QWEN36_FP8.to_owned(),
            // The device arm serves one rank; the family's TP group is not wired yet.
            worlds: &[1],
            correct: Architecture::weight_only_fp8,
        },
    ]
}

fn document(name: &str) -> Released {
    released()
        .into_iter()
        .find(|document| document.name == name)
        .unwrap_or_else(|| panic!("no released document named {name}"))
}

/// Plan a released document at `world`, applying the arm's producer correction first.
fn plan_for(document: &Released, world: usize) -> EnginePlan {
    plan_for_rank(document, world, 0)
}

/// Plan one rank of a released document, applying the arm's producer correction first.
fn plan_for_rank(document: &Released, world: usize, rank: usize) -> EnginePlan {
    let mut architecture = Architecture::from_json(&document.text)
        .unwrap_or_else(|error| panic!("{}: {error}", document.name));
    (document.correct)(&mut architecture);
    EnginePlan::for_architecture(architecture, Target::Sm121, world, rank)
        .unwrap_or_else(|error| panic!("{} TP{world}/{rank}: {error}", document.name))
}

fn skip_vision(name: &str) -> bool {
    name.starts_with("Qwen")
}

/// Rewrite every model_type in the document, including the nested text and vision configs.
fn unregister_model_type(text: &str) -> String {
    fn rewrite(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(fields) => {
                if fields.contains_key("model_type") {
                    fields.insert(
                        "model_type".to_owned(),
                        serde_json::Value::String("unregistered_model_type".to_owned()),
                    );
                }
                for child in fields.values_mut() {
                    rewrite(child);
                }
            }
            serde_json::Value::Array(items) => {
                for child in items {
                    rewrite(child);
                }
            }
            _ => {}
        }
    }
    let mut value: serde_json::Value = serde_json::from_str(text).expect("json");
    rewrite(&mut value);
    value.to_string()
}

fn is_vision(part: Part) -> bool {
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

fn resolve_for(document: &Released, world: usize, phase: KernelPhase) -> Vec<ResolvedKernel> {
    let plan = plan_for(document, world);
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    plan.requirements()
        .filter(|requirement| !is_vision(requirement.part))
        .map(|requirement| {
            resolver
                .resolve_for(requirement, phase)
                .unwrap_or_else(|error| panic!("{} TP{world} {phase:?}: {error}", document.name))
        })
        .collect()
}

#[test]
fn every_released_config_plans_at_tp1_tp2_and_tp4() {
    for document in released() {
        for world in [1, 2, 4] {
            let plan = plan_for(&document, world);
            assert_eq!(
                plan.layers().len(),
                plan.architecture().layers.len(),
                "{} TP{world} layer count",
                document.name
            );
            assert!(plan.requirements().count() > plan.layers().len());
            assert!(plan.boot_summary().contains(&format!("TP0/{world}")));
        }
    }
}

#[test]
fn the_tp_split_is_rank_local() {
    let plan = EnginePlan::from_json(GLM, Target::Sm121, 4, 3).expect("GLM TP4 rank 3");
    let tp = plan.tp();
    assert_eq!(tp.world(), 4);
    assert_eq!(tp.rank(), 3);
    let mlp = plan
        .layers()
        .iter()
        .flat_map(|layer| &layer.requirements)
        .find(|requirement| requirement.part == Part::DenseMlp || requirement.part == Part::Moe)
        .expect("a feed-forward part");
    assert_eq!(mlp.geometry.world, 4);
    assert_eq!(mlp.geometry.rank, 3);
    assert!(mlp.geometry.intermediate > 0);

    let qwen = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 4, 3).expect("Qwen TP4 rank 3");
    let gqa = qwen
        .requirements()
        .find(|requirement| requirement.part == Part::GqaAttention)
        .expect("a Qwen GQA layer");
    assert_eq!(gqa.geometry.kv_heads, 1);
    assert_eq!(gqa.geometry.kv_head_offset, 1);
    assert_eq!(gqa.geometry.head_offset, 3 * gqa.geometry.heads);
}

#[test]
fn every_released_config_resolves_at_every_supported_world() {
    for document in released() {
        for &world in document.worlds {
            let plan = plan_for(&document, world);
            let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
            let mut resolved = Vec::new();
            for requirement in plan.requirements() {
                if is_vision(requirement.part) {
                    continue;
                }
                resolved.push(
                    resolver
                        .resolve_for(requirement, KernelPhase::Both)
                        .unwrap_or_else(|error| panic!("{} TP{world}: {error}", document.name)),
                );
            }
            assert!(!resolved.is_empty());
            assert!(resolved.iter().all(|kernel| kernel.target == Target::Sm121));
            assert!(resolved.iter().any(|kernel| kernel.part == Part::Moe));
        }
    }
}

#[test]
fn deepseek_moe_refuses_tp1_and_tp2_with_the_intermediate_width() {
    for world in [1, 2] {
        let plan = EnginePlan::from_json(DEEPSEEK, Target::Sm121, world, 0)
            .expect("the plan itself is well formed");
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let failures = resolver.failures(&plan);
        assert!(!failures.is_empty(), "TP{world} must refuse");
        for failure in failures.iter().filter(|failure| !is_vision(failure.part)) {
            assert_eq!(
                failure.part,
                Part::Moe,
                "TP{world} refused an unexpected part: {failure}"
            );
            let message = failure.to_string();
            assert!(
                message.contains(&format!("intermediate {} ", 2304 / world)),
                "{message}"
            );
            assert!(message.contains("576"), "{message}");
            assert!(message.contains("deepseek_ptx"), "{message}");
        }
        let moe = plan
            .requirements()
            .find(|requirement| requirement.part == Part::Moe)
            .expect("a MoE requirement");
        assert_eq!(moe.geometry.intermediate, 2304 / world);
    }
}

#[test]
fn the_qwen_moe_prefill_and_decode_bodies_differ() {
    let plan = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 4, 0).expect("Qwen TP4");
    let moe = plan
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen MoE requirement");
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    assert_eq!(
        resolver
            .resolve_for(moe, KernelPhase::Prefill)
            .expect("prefill MoE")
            .image,
        Image::Qwen4Prefill
    );
    assert_eq!(
        resolver
            .resolve_for(moe, KernelPhase::Decode)
            .expect("decode MoE")
            .image,
        Image::Qwen4Moe
    );
}

/// The image a released arm's part must resolve to, or None when the arm has no such part.
fn expected_image(name: &str, part: Part, phase: KernelPhase) -> Option<Image> {
    match (name, part) {
        ("DeepSeek", Part::LmHead | Part::DenseMlp) => Some(Image::Deepseek),
        (_, Part::LmHead) => Some(Image::Qwen4Exp),
        ("GLM FP8" | "GLM BF16", Part::DenseMlp) => Some(match phase {
            KernelPhase::Decode => Image::GlmDecode,
            KernelPhase::Prefill | KernelPhase::Both => Image::GlmPrefill,
        }),
        _ => None,
    }
}

#[test]
fn the_shared_head_and_dense_mlp_resolve_their_arm_specific_images() {
    for document in released() {
        let plan = plan_for(&document, 1);
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        for part in [Part::LmHead, Part::DenseMlp] {
            let Some(requirement) = plan
                .requirements()
                .find(|requirement| requirement.part == part)
            else {
                continue;
            };
            for phase in [KernelPhase::Prefill, KernelPhase::Decode] {
                let resolved = resolver
                    .resolve_for(requirement, phase)
                    .unwrap_or_else(|error| panic!("{} {part} {phase:?}: {error}", document.name));
                assert!(
                    resolved.phase().serves(phase),
                    "{} {part} {phase:?} selected {} which is {}",
                    document.name,
                    resolved.symbol_name(),
                    resolved.phase().name()
                );
                let expected = expected_image(document.name, part, phase)
                    .unwrap_or_else(|| panic!("{} {part}: no pinned image", document.name));
                assert_eq!(
                    resolved.image,
                    expected,
                    "{} {part} {phase:?} selected {} on {}",
                    document.name,
                    resolved.symbol_name(),
                    resolved.image.name()
                );
            }
        }
    }
}

/// The Qwen dense-MLP body the arm launches at bulk prefill. Qwen has no dense decoder FFN, but
/// its prefill GEMM path is described by a DenseMlp requirement at the release geometry; the
/// catalogue used to answer GlmPrefill for it because the GLM entry accepted any 128-wide hidden.
#[test]
fn both_qwen_arms_resolve_their_own_dense_prefill_body() {
    for name in ["Qwen NVFP4", "Qwen FP8"] {
        let document = document(name);
        let plan = plan_for(&document, 1);
        let requirement = PartRequirement {
            part: Part::DenseMlp,
            geometry: PartGeometry {
                hidden: 2560,
                rows: 64,
                intermediate: 640,
                positions: plan.architecture().max_position_embeddings,
                world: 1,
                rank: 0,
                // The arm's requirement names its family: a GLM body that fits the widths is
                // not Qwen's.
                model: Some(ModelFamily::Qwen),
                ..PartGeometry::new()
            },
            format: sparkflash_ops::engine_plan::format_of_precision(plan.precision().dense),
        };
        assert_eq!(requirement.precision(), Precision::Bf16, "{name}");
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        let prefill = resolver
            .resolve_for(&requirement, KernelPhase::Prefill)
            .unwrap_or_else(|error| panic!("{name} dense prefill: {error}"));
        assert_eq!(prefill.image, Image::Qwen4Prefill, "{name}");
        assert_eq!(prefill.symbol_name(), "q4p_dense_mma", "{name}");
        assert_ne!(prefill.image, Image::GlmPrefill, "{name}");
        let decode = resolver
            .resolve_for(&requirement, KernelPhase::Decode)
            .unwrap_or_else(|error| panic!("{name} dense decode: {error}"));
        assert_eq!(decode.image, Image::Qwen4Exp, "{name}");
    }
}

#[test]
fn a_boot_check_proves_both_passes_for_the_deepseek_arm() {
    let plan = EnginePlan::from_json(DEEPSEEK, Target::Sm121, 4, 0).expect("DeepSeek TP4");
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    let served = sparkflash_ops::parts::serving_language(Target::Sm121);
    let language = plan
        .requirements()
        .filter(|requirement| !is_vision(requirement.part))
        .count();
    let resolved = resolver
        .boot_check_serving(&plan, &served)
        .unwrap_or_else(|errors| panic!("boot check: {errors:?}"));
    assert!(!resolved.is_empty());
    assert_eq!(resolved.len(), 2 * language);
}

#[test]
fn the_glm_fp8_arm_boot_checks_both_passes() {
    let plan = EnginePlan::from_json(GLM, Target::Sm121, 4, 0).expect("GLM FP8 TP4");
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    let served = sparkflash_ops::parts::serving_language(Target::Sm121);
    let language = plan
        .requirements()
        .filter(|requirement| !is_vision(requirement.part))
        .count();
    let resolved = resolver
        .boot_check_serving(&plan, &served)
        .unwrap_or_else(|errors| panic!("GLM boot check: {errors:?}"));
    assert_eq!(resolved.len(), 2 * language);
}

/// A Qwen FP8 serve boots both passes at every rank width the release serves, not only at TP1.
/// The released vision tower is a separate, explicit refusal (a 16-patch document against the
/// shared 14-patch body), so this checks the language parts: the FP8 grouped decode bodies and
/// the dense prefill body must be present at TP4, where the release actually runs.
#[test]
fn the_qwen_fp8_arm_boot_checks_both_passes_at_tp4() {
    let document = document("Qwen FP8");
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    for world in [1, 4] {
        let plan = plan_for(&document, world);
        for phase in [KernelPhase::Prefill, KernelPhase::Decode] {
            for requirement in plan
                .requirements()
                .filter(|requirement| !is_vision(requirement.part))
            {
                let resolved = resolver
                    .resolve_for(requirement, phase)
                    .unwrap_or_else(|error| {
                        panic!("Qwen FP8 TP{world} {} {phase:?}: {error}", requirement.part)
                    });
                assert!(
                    resolved.phase().serves(phase),
                    "Qwen FP8 TP{world} {} {phase:?} selected {} which is {}",
                    requirement.part,
                    resolved.symbol_name(),
                    resolved.phase().name()
                );
            }
        }
        // The dense prefill body is Qwen's own bulk MMA, not one of the GLM prefill projections.
        let dense = PartRequirement {
            part: Part::DenseMlp,
            geometry: PartGeometry {
                hidden: 2560,
                rows: 64,
                intermediate: 640,
                positions: plan.architecture().max_position_embeddings,
                world,
                rank: 0,
                model: Some(ModelFamily::Qwen),
                ..PartGeometry::new()
            },
            format: sparkflash_ops::engine_plan::format_of_precision(plan.precision().dense),
        };
        assert_eq!(dense.precision(), Precision::Bf16, "Qwen FP8 TP{world}");
        let prefill = resolver
            .resolve_for(&dense, KernelPhase::Prefill)
            .unwrap_or_else(|error| panic!("Qwen FP8 TP{world} dense prefill: {error}"));
        assert_eq!(prefill.image, Image::Qwen4Prefill, "Qwen FP8 TP{world}");
        assert_eq!(prefill.symbol_name(), "q4p_dense_mma", "Qwen FP8 TP{world}");
    }
}

/// D1: a Qwen FP8 chunk resolves the genuine grouped FP8 prefill bodies, not the NVFP4 grouped
/// GEMM that `q4p_moe_gate_up`/`q4p_moe_down` implement, and a token step resolves the FP8 token
/// experts. The token FP8 bodies serve both passes because the device launches them over short
/// prefill batches as well as decode steps.
#[test]
fn the_qwen_fp8_arm_resolves_the_grouped_fp8_symbols_at_tp4() {
    let document = document("Qwen FP8");
    let plan = plan_for(&document, 4);
    let moe = plan
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen FP8 MoE requirement");
    assert_eq!(moe.precision(), Precision::Fp8E4M3, "Qwen FP8 experts");
    // Rank 0 of four holds two of the expert's five 128-wide scale tiles (256 columns), the
    // width `rank_config` uploads; the old 160-column even split is not a run of whole tiles.
    assert_eq!(moe.geometry.intermediate, 256, "Qwen FP8 TP4 rank 0");
    let requirement = PartRequirement {
        part: Part::GroupedMoeGemm,
        ..*moe
    };
    // Qwen's host takes the grouped tile from GEMM_MIN_BATCH (64 tokens); below that it runs
    // its token FP8 experts. The plan fills the threshold and body from the released formats,
    // so a caller deriving a grouped requirement does not hand-set them.
    assert_eq!(requirement.geometry.grouped_min_tokens, 64, "Qwen FP8 TP4");
    assert_eq!(
        requirement.geometry.grouped_fp8_body,
        Some(GroupedFp8Body::Tiled),
        "Qwen FP8 TP4"
    );
    let context = PlanContext::SM121_CATALOGUE;
    // A chunk (>= the device's GEMM_MIN_BATCH) is the grouped tensor-core tile; the NVFP4 body
    // is not eligible because its own mask no longer claims Fp8E4M3.
    let chunk = GroupedMoeGemmPlan::new(context, &requirement, 64, StreamId(0))
        .unwrap_or_else(|error| panic!("Qwen FP8 TP4 grouped prefill: {error}"));
    assert_eq!(
        chunk.kernel(),
        KernelId::new(Target::Sm121, Image::Qwen4Prefill, "q4p_moe_gate_up_fp8"),
        "Qwen FP8 TP4 grouped prefill"
    );
    // A token step falls through the prefill-only grouped bodies to the token FP8 experts.
    let token = GroupedMoeGemmPlan::new(context, &requirement, 1, StreamId(0))
        .unwrap_or_else(|error| panic!("Qwen FP8 TP4 token grouped experts: {error}"));
    assert_eq!(
        token.kernel(),
        KernelId::new(Target::Sm121, Image::Qwen4Moe, "q4e_expert_gate_up_fp8"),
        "Qwen FP8 TP4 token grouped experts"
    );
    // Both passes of the plan's MoE requirement resolve an FP8 body, and the prefill pass is not
    // the NVFP4 grouped GEMM.
    for phase in [KernelPhase::Prefill, KernelPhase::Decode] {
        let resolved = context
            .resolve(
                &requirement,
                phase,
                &["q4p_moe_gate_up_fp8", "q4e_expert_gate_up_fp8"],
            )
            .unwrap_or_else(|error| panic!("Qwen FP8 TP4 {phase:?}: {error}"));
        assert_eq!(resolved.precision(), Precision::Fp8E4M3, "{phase:?}");
        assert_ne!(resolved.symbol_name(), "q4p_moe_gate_up", "{phase:?}");
    }
}

/// A GLM requirement derived from its [EnginePlan] carries the release's grouped policy, so a
/// caller never inherits Qwen's tile: the released dynamic-activation default names the q8 body
/// and the host's threshold of eight token rows becomes the plan's nine-token grouped start.
#[test]
fn the_glm_fp8_arm_inherits_its_grouped_policy_from_the_plan() {
    let document = document("GLM FP8");
    let plan = plan_for(&document, 4);
    let moe = plan
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a GLM FP8 MoE requirement");
    assert_eq!(moe.precision(), Precision::Fp8E4M3, "GLM FP8 experts");
    // The released-defaults case: no profile override is applied here, so the plan must carry
    // the dynamic q8 body and the host's nine-token grouped start (`MAX_ROWS` is 8).
    assert_eq!(
        moe.geometry.grouped_min_tokens, 9,
        "GLM groups from nine token rows"
    );
    assert_eq!(
        moe.geometry.grouped_fp8_body,
        Some(GroupedFp8Body::QuantizedActivation),
        "GLM FP8 TP4"
    );
    let requirement = PartRequirement {
        part: Part::GroupedMoeGemm,
        ..*moe
    };
    let context = PlanContext::SM121_CATALOGUE;
    for tokens in [1, 8] {
        let token = GroupedMoeGemmPlan::new(context, &requirement, tokens, StreamId(0))
            .unwrap_or_else(|error| panic!("GLM FP8 TP4 {tokens}-token step: {error}"));
        assert_eq!(
            token.kernel(),
            KernelId::new(
                Target::Sm121,
                Image::GlmDecode,
                "q4e_expert_gemv_e4m3_block"
            ),
            "{tokens} tokens is GLM's decode GEMV"
        );
    }
    let chunk = GroupedMoeGemmPlan::new(context, &requirement, 9, StreamId(0))
        .unwrap_or_else(|error| panic!("GLM FP8 TP4 grouped prefill: {error}"));
    assert_eq!(
        chunk.kernel(),
        KernelId::new(Target::Sm121, Image::GlmMoe, "glm_moe_q8_mma"),
        "nine tokens is GLM's grouped prefill"
    );
}

/// The released plan names the dynamic-activation default, but a serving profile can override
/// the prefill activation to `bf16` or `split`. [EnginePlan] does not read the profile, so the
/// caller constructing the grouped requirement replaces the body; the plan must then resolve
/// that profile's block MMA instead of the q8 body.
#[test]
fn the_glm_fp8_grouped_requirement_accepts_a_prefill_activation_override() {
    let document = document("GLM FP8");
    let plan = plan_for(&document, 4);
    let moe = plan
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a GLM FP8 MoE requirement");
    let mut requirement = PartRequirement {
        part: Part::GroupedMoeGemm,
        ..*moe
    };
    let context = PlanContext::SM121_CATALOGUE;
    for (body, symbol) in [
        (GroupedFp8Body::Bf16Activation, "glm_moe_fp8b128_mma_bf16a"),
        (GroupedFp8Body::SplitActivation, "glm_moe_fp8b128_mma"),
    ] {
        requirement.geometry.grouped_fp8_body = Some(body);
        let chunk = GroupedMoeGemmPlan::new(context, &requirement, 9, StreamId(0))
            .unwrap_or_else(|error| panic!("{body:?} GLM grouped prefill: {error}"));
        assert_eq!(
            chunk.kernel(),
            KernelId::new(Target::Sm121, Image::GlmMoe, symbol),
            "{body:?} override"
        );
    }
}

/// Every released document is boot-checked here, so a document whose experts have no body in
/// one pass cannot hide behind a "Both"-pass resolution. GLM BF16 and Qwen BF16 are the
/// explicit refusals: their routed experts are only catalogued as quantized weight bodies
/// (the GLM block MMA and q8 prefill bodies, and the Qwen E2M1/E4M3 bodies), and
/// `glm_moe_fp8b128_mma_bf16a` is one of those - its `_bf16a` suffix names BF16 activations,
/// not a BF16 weight class - so both passes refuse the MoE. The Qwen document's 16-patch tower
/// is still refused by the catalogue on the plan itself.
#[test]
fn every_released_document_boot_checks_or_is_an_explicit_refusal() {
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    for document in released() {
        let world = document.worlds.first().copied().unwrap_or(4);
        let plan = plan_for(&document, world);
        let result = resolver.boot_check_serving(
            &plan,
            &sparkflash_ops::parts::serving_language(Target::Sm121),
        );
        if document.worlds.is_empty() {
            let errors = result.expect_err("a BF16 document has no routed-expert body");
            assert!(
                !errors.is_empty(),
                "{}: the refusal names no part",
                document.name
            );
            // The vision tower is a separate refusal: the Qwen document still names a 16-patch
            // tower that the catalogue refuses on the plan itself, and the weight-format
            // refusal below is the language model's.
            let mut phases = Vec::new();
            for error in errors.iter().filter(|error| error.part != Part::Vision) {
                assert_eq!(error.part, Part::Moe, "{}: {error}", document.name);
                assert_eq!(
                    sparkflash_ops::engine_plan::precision_of_stored(error.format),
                    Precision::Bf16,
                    "{}: {error}",
                    document.name
                );
                // The refusal is keyed on the stored format, so it names the format the
                // catalogue cannot decode as well as the precision no body accepts; the wording
                // is either "no symbol accepting bf16 in the <pass> pass" or "<image> exports no
                // symbol decoding BF16 (...)".
                let message = error.to_string();
                assert!(
                    message.contains(&format!(
                        "no symbol accepting bf16 in the {} pass",
                        error.phase.name()
                    )) || (message.contains("exports no symbol decoding BF16")
                        && message.contains(&format!("in the {} pass", error.phase.name()))),
                    "{}: {error}",
                    document.name
                );
                phases.push(error.phase);
            }
            for phase in [KernelPhase::Prefill, KernelPhase::Decode] {
                assert!(
                    phases.contains(&phase),
                    "{}: no {phase:?} refusal; an FP8 weight body must not serve BF16 ({} refusals)",
                    document.name,
                    errors.len()
                );
            }
        } else {
            result.unwrap_or_else(|errors| panic!("{}: {errors:?}", document.name));
        }
    }
}

/// The serving gate and the catalogue answer different questions. The catalogue judges the
/// whole checkpoint plan, so it still refuses Qwen's 16-patch tower; a serve executes the
/// language parts only (D84), so the serving check must resolve them at every released width.
#[test]
fn the_serving_boot_check_skips_the_unloaded_vision_tower() {
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    let served = sparkflash_ops::parts::serving_language(Target::Sm121);
    for document in released() {
        for &world in document.worlds {
            let plan = plan_for(&document, world);
            if skip_vision(document.name) {
                let errors = resolver
                    .boot_check(&plan)
                    .expect_err("the catalogue still refuses the 16-patch tower");
                assert!(
                    errors.iter().all(|error| is_vision(error.part)),
                    "{} TP{world}: the language parts resolve: {errors:?}",
                    document.name
                );
                assert!(
                    errors
                        .iter()
                        .any(|error| error.part == Part::VisionPatchEmbed
                            && error.to_string().contains("1152")),
                    "{} TP{world}: the refusal names the tower width: {errors:?}",
                    document.name
                );
            }
            let language = plan
                .requirements()
                .filter(|requirement| !is_vision(requirement.part))
                .count();
            let resolved = resolver
                .boot_check_serving(&plan, &served)
                .unwrap_or_else(|errors| panic!("{} TP{world}: {errors:?}", document.name));
            assert_eq!(resolved.len(), 2 * language, "{} TP{world}", document.name);
            assert!(resolved.iter().all(|kernel| kernel.target == Target::Sm121));
        }
    }
}

/// D1/P4: a Qwen FP8 expert width is the host's run of whole 128-column scale tiles, not an
/// even split padded to 16 columns. Five 128-wide tiles over four ranks are 2/1/1/1 columns,
/// the same widths `rank_config` uploads.
#[test]
fn the_qwen_fp8_expert_plan_splits_whole_128_tiles() {
    let document = document("Qwen FP8");
    for (rank, width) in [(0, 256), (1, 128), (2, 128), (3, 128)] {
        let plan = plan_for_rank(&document, 4, rank);
        let moe = plan
            .requirements()
            .find(|requirement| requirement.part == Part::Moe)
            .expect("a Qwen FP8 MoE requirement");
        assert_eq!(moe.precision(), Precision::Fp8E4M3, "rank {rank}");
        assert_eq!(moe.geometry.intermediate, width, "rank {rank}");
    }
    let plan = plan_for(&document, 1);
    let moe = plan
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen FP8 MoE requirement");
    assert_eq!(moe.geometry.intermediate, 640, "TP1 holds the whole expert");
}

#[test]
fn the_catalogue_picks_the_image_by_shape_precision_and_pass() {
    let qwen = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 4, 0).expect("Qwen");
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    let moe = qwen
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen MoE requirement");
    assert_eq!(resolver.resolve(moe).expect("MoE").image, Image::Qwen4Moe);
    let head = qwen
        .requirements()
        .find(|requirement| requirement.part == Part::LmHead)
        .expect("a Qwen head requirement");
    assert_eq!(resolver.resolve(head).expect("head").image, Image::Qwen4Exp);
    let scan = qwen
        .requirements()
        .find(|requirement| requirement.part == Part::GdnScan)
        .expect("a Qwen GDN layer");
    assert_eq!(
        resolver.resolve(scan).expect("scan").image,
        Image::Qwen4Scan
    );
    assert!(
        qwen.requirements()
            .filter(|requirement| !is_vision(requirement.part))
            .all(
                |requirement| resolver.resolve(requirement).expect("part").image != Image::Deepseek
            ),
        "a shared checkpoint resolves to the DeepSeek runtime image"
    );

    let glm = EnginePlan::from_json(GLM, Target::Sm121, 4, 0).expect("GLM");
    let moe = glm
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a GLM MoE requirement");
    assert_eq!(resolver.resolve(moe).expect("MoE").image, Image::GlmMoe);
    assert_eq!(
        resolver
            .resolve_for(moe, KernelPhase::Prefill)
            .expect("prefill MoE")
            .image,
        Image::GlmMoe
    );
    assert_eq!(
        resolver
            .resolve_for(moe, KernelPhase::Decode)
            .expect("decode MoE")
            .image,
        Image::GlmDecode
    );
    let head = glm
        .requirements()
        .find(|requirement| requirement.part == Part::LmHead)
        .expect("a GLM head requirement");
    assert_eq!(
        resolver.resolve(head).expect("head").image,
        Image::Qwen4Exp,
        "GLM's token/head kernels are the shared token image"
    );

    // Qwen3.6-35B-A3B is the second published Qwen routed shape (2048, 256, 8): its E4M3 experts
    // resolve to the Qwen token and grouped bodies, while GLM's 4096-wide FP8 experts above stay
    // on the GLM images.
    let qwen36 = plan_for(&document("Qwen3.6 FP8"), 1);
    let moe = qwen36
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen3.6 MoE requirement");
    assert_eq!(
        (
            moe.geometry.hidden,
            moe.geometry.experts,
            moe.geometry.top_k
        ),
        (2048, 256, 8)
    );
    for phase in [KernelPhase::Prefill, KernelPhase::Decode] {
        assert_eq!(
            resolver.resolve_for(moe, phase).expect("Qwen3.6 MoE").image,
            Image::Qwen4Moe,
            "{phase:?}: the token FP8 experts serve both passes, as on Qwen3.8 FP8"
        );
    }
    let grouped = PartRequirement {
        part: Part::GroupedMoeGemm,
        ..*moe
    };
    // The arm computes the E4M3 tiles weight-only, so the grouped policy is Qwen's (the
    // BF16-activation tile from 64 tokens), not the q8 MMA GLM's identical formats select.
    assert_eq!(grouped.geometry.grouped_min_tokens, 64);
    assert_eq!(
        grouped.geometry.grouped_fp8_body,
        Some(GroupedFp8Body::Tiled)
    );
    let chunk = GroupedMoeGemmPlan::new(PlanContext::SM121_CATALOGUE, &grouped, 64, StreamId(0))
        .unwrap_or_else(|error| panic!("Qwen3.6 grouped prefill: {error}"));
    assert_eq!(
        chunk.kernel(),
        KernelId::new(Target::Sm121, Image::Qwen4Prefill, "q4p_moe_gate_up_fp8"),
        "a Qwen3.6 chunk is the grouped FP8 tile"
    );
    // A routed shape neither release has is refused by the Qwen images, not claimed.
    let mut novel = *moe;
    novel.geometry.top_k = 6;
    let error = resolver
        .resolve_for(&novel, KernelPhase::Decode)
        .expect_err("an unpublished routed shape");
    assert!(
        error.to_string().contains("routed shape"),
        "the refusal names the published shapes: {error}"
    );

    let deepseek = EnginePlan::from_json(DEEPSEEK, Target::Sm121, 4, 0).expect("DeepSeek");
    let moe = deepseek
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a DeepSeek MoE requirement");
    assert_eq!(resolver.resolve(moe).expect("MoE").image, Image::Deepseek);
    assert_eq!(
        resolver.resolve(moe).expect("MoE").precision(),
        Precision::MxFp4
    );
}

#[test]
fn a_shared_checkpoint_is_not_replaced_by_a_loadable_cross_image() {
    if Image::Qwen4Moe.available() {
        return;
    }
    let plan = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 4, 0).expect("Qwen plan");
    let moe = plan
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen MoE requirement");
    let resolver = KernelResolver::new(Target::Sm121, Availability::runtime(false));
    let error = resolver
        .resolve(moe)
        .expect_err("the Qwen image is not embedded and DeepSeek must not stand in");
    assert!(
        error
            .reasons
            .iter()
            .any(|reason| reason.contains("qwen4_moe")),
        "{error}"
    );
}

#[test]
fn an_embedded_resolver_refuses_a_build_without_the_images() {
    if Image::Qwen4Exp.available() {
        return;
    }
    let plan = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 1, 0).expect("Qwen plan");
    let resolver = KernelResolver::new(Target::Sm121, Availability::Embedded);
    let failures = resolver.failures(&plan);
    assert!(!failures.is_empty());
    assert!(failures.iter().all(|failure| !failure.reasons.is_empty()));
}

#[test]
fn a_world_that_does_not_divide_a_released_config_is_refused() {
    let error = EnginePlan::from_json(GLM, Target::Sm121, 3, 0).expect_err("GLM TP3");
    assert!(error.to_string().contains("layer"));
}

#[test]
fn the_quantization_block_selects_the_precision_class() {
    let deepseek = EnginePlan::from_json(DEEPSEEK, Target::Sm121, 4, 0).expect("DeepSeek");
    assert_eq!(deepseek.precision().dense, Precision::MxFp8);
    assert_eq!(
        deepseek.precision().experts,
        Precision::MxFp4,
        "\"fp4\" beside a ue8m0 scale is MXFP4: one power-of-two code a 32 values, its own \
         arithmetic rather than NVFP4's per-16 group and tensor-wide factor"
    );
    assert_eq!(
        deepseek.precision().mtp,
        Precision::MxFp4,
        "an unnamed MTP format uses the backbone experts'"
    );

    let glm = EnginePlan::from_json(GLM, Target::Sm121, 4, 0).expect("GLM");
    assert_eq!(glm.precision().dense, Precision::Fp8E4M3);
    assert_eq!(
        glm.precision().experts,
        Precision::Fp8E4M3,
        "GLM names no expert format, so the dense FP8 block governs"
    );
    assert_eq!(glm.precision().mtp, Precision::Fp8E4M3);

    let glm_bf16 = EnginePlan::from_json(&glm_bf16(), Target::Sm121, 4, 0).expect("GLM BF16");
    assert_eq!(glm_bf16.precision().dense, Precision::Bf16);
    assert_eq!(glm_bf16.precision().experts, Precision::Bf16);
    assert_eq!(glm_bf16.precision().mtp, Precision::Bf16);

    let qwen = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 1, 0).expect("Qwen");
    assert_eq!(qwen.precision().dense, Precision::Bf16);
    assert_eq!(qwen.precision().experts, Precision::NvFp4);
    assert_eq!(
        qwen.precision().mtp,
        Precision::Fp8E4M3,
        "the MTP stage keeps its own FP8_PB_WO format"
    );
    let mtp = qwen
        .globals()
        .iter()
        .find(|requirement| requirement.part == Part::Mtp)
        .expect("Qwen's boot plan carries an MTP part");
    assert_eq!(mtp.precision(), Precision::Fp8E4M3);

    // The Qwen FP8 pack stores only the routed experts as E4M3; the document's global
    // `quant_method: fp8` is the producer's ambiguous flag, and the arm's architecture bridge
    // corrects it. Dense, attention and MTP non-expert stay BF16.
    let qwen_fp8 = plan_for(&document("Qwen FP8"), 1);
    assert_eq!(qwen_fp8.precision().dense, Precision::Bf16);
    assert_eq!(qwen_fp8.precision().attention, Precision::Bf16);
    assert_eq!(qwen_fp8.precision().experts, Precision::Fp8E4M3);
    assert_eq!(qwen_fp8.precision().mtp, Precision::Fp8E4M3);
    let moe = qwen_fp8
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen MoE requirement");
    assert_eq!(
        moe.precision(),
        Precision::Fp8E4M3,
        "the routed experts are FP8"
    );
    let mtp = qwen_fp8
        .globals()
        .iter()
        .find(|requirement| requirement.part == Part::Mtp)
        .expect("Qwen's boot plan carries an MTP part");
    assert_eq!(
        mtp.precision(),
        Precision::Fp8E4M3,
        "the MTP experts are FP8"
    );
    let attention = qwen_fp8
        .requirements()
        .find(|requirement| requirement.part == Part::GqaAttention)
        .expect("a Qwen GQA requirement");
    assert_eq!(
        attention.precision(),
        Precision::Bf16,
        "the attention weights are BF16"
    );
}

#[test]
fn a_release_whose_format_the_shared_parser_does_not_know_falls_back_to_bf16() {
    let plan = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 1, 0).expect("Qwen");
    assert_eq!(plan.precision().dense, Precision::Bf16);
    assert_eq!(plan.precision().experts, Precision::NvFp4);
    assert_eq!(
        plan.precision(),
        PrecisionPolicy::from_architecture(plan.architecture())
    );
}

#[test]
fn mutating_model_type_does_not_change_the_resolved_kernels() {
    for document in released() {
        for &world in document.worlds {
            let before = resolve_for(&document, world, KernelPhase::Both);
            let mutated = Released {
                name: document.name,
                text: unregister_model_type(&document.text),
                worlds: document.worlds,
                correct: document.correct,
            };
            let after = resolve_for(&mutated, world, KernelPhase::Both);
            assert_eq!(
                before, after,
                "{} TP{world}: the model_type string changed the kernel list",
                document.name
            );
        }
    }
}

#[test]
fn an_unsupported_scan_head_is_refused_by_part_image_and_constraint() {
    let text = QWEN_NVFP4.replace(
        "\"linear_key_head_dim\": 128",
        "\"linear_key_head_dim\": 96",
    );
    let plan = EnginePlan::from_json(&text, Target::Sm121, 1, 0).expect("Qwen plan");
    let failures = KernelResolver::new(Target::Sm121, Availability::Catalogue).failures(&plan);
    let failure = failures
        .iter()
        .find(|failure| failure.part == Part::GdnScan)
        .unwrap_or_else(|| panic!("no GDN failure: {failures:?}"));
    let message = failure.to_string();
    assert!(message.contains("gdn_scan"), "{message}");
    assert!(message.contains("qwen4_scan_sm121"), "{message}");
    assert!(message.contains("head pair"), "{message}");
}

#[test]
fn the_released_qwen_vision_patch_is_refused() {
    let plan = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 1, 0).expect("Qwen plan");
    let failures = KernelResolver::new(Target::Sm121, Availability::Catalogue).failures(&plan);
    let failure = failures
        .iter()
        .find(|failure| is_vision(failure.part))
        .unwrap_or_else(|| panic!("no vision failure: {failures:?}"));
    let message = failure.to_string();
    assert!(message.contains("vision"), "{message}");
    assert!(
        message.contains("hidden 1152 does not equal 1024"),
        "{message}"
    );
}

#[test]
fn an_mxfp4_block_is_carried_in_the_expert_scale_shape() {
    // MXFP4's released block is 32, not NVFP4's group 16. The plan must use the block it was
    // given: a 2560-wide expert's inverse scale has 80 columns, not the 160 a 16-group would.
    let mut architecture =
        Architecture::from_json(QWEN_NVFP4).expect("the Qwen NVFP4 release parses");
    architecture.quantization.expert_format =
        sparkflash_model::architecture::WeightFormat::MxFp4 { block: 32 };
    let tp = sparkflash_model::TpPlan::new(1, 0).expect("one rank");
    let plan = sparkflash_ops::tensor_plan::ArchitectureTensorPlan::served(&architecture, tp)
        .expect("the MXFP4 plan builds");
    let scale = plan
        .rank_spec("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale")
        .expect("the expert scale is planned");
    assert_eq!(scale.shape, vec![640, 2560usize.div_ceil(32)]);
    assert_ne!(scale.shape[1], 2560usize.div_ceil(16));
    EnginePlan::for_architecture(architecture, Target::Sm121, 1, 0)
        .expect("the execution plan carries the same format");
}

#[test]
fn an_admitted_context_is_capped_by_the_checkpoint_window() {
    let plan = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 1, 0).expect("Qwen plan");
    let window = plan.architecture().max_position_embeddings;
    assert_eq!(
        plan.admitted_context(window / 2).expect("a short context"),
        window / 2
    );
    assert_eq!(
        plan.admitted_context(window + 1)
            .expect("a request past the window"),
        window,
        "a request past the checkpoint window is capped, not allocated"
    );
    assert_eq!(
        plan.admitted_context(window).expect("the whole window"),
        window
    );
    assert!(
        plan.admitted_context(0).is_err(),
        "a zero context is refused"
    );
}

#[test]
fn a_non_replicated_qwen_mtp_is_not_a_tp4_global() {
    // Qwen4-Exp clears mtp.replicated: its TP>1 rank configuration drafts with DFlash and
    // sets the MTP layer count to zero, so served_and_mtp omits the stage at TP4. The boot
    // plan must name the same set; a global the served tensor plan does not carry would claim
    // a part with no weights.
    let qwen_tp4 = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 4, 0).expect("Qwen TP4");
    let mtp = qwen_tp4
        .architecture()
        .mtp
        .as_ref()
        .expect("the Qwen release declares an MTP stage");
    assert!(!mtp.replicated, "Qwen4-Exp is the non-replicated release");
    assert!(
        qwen_tp4
            .globals()
            .iter()
            .all(|requirement| requirement.part != Part::Mtp),
        "the TP4 boot plan must not name a stage the served set omits"
    );
    let tensors = ArchitectureTensorPlan::served_and_mtp(qwen_tp4.architecture(), qwen_tp4.tp())
        .expect("the Qwen TP4 served plan builds");
    assert_eq!(
        tensors.role_bytes(Role::Mtp),
        0,
        "the TP4 served set carries no MTP tensor"
    );
    assert!(tensors.roles().all(|role| role != Role::Mtp));

    // TP1 still carries the stage: the same release serves it when one rank owns the model.
    let qwen_tp1 = EnginePlan::from_json(QWEN_NVFP4, Target::Sm121, 1, 0).expect("Qwen TP1");
    let mtp = qwen_tp1
        .globals()
        .iter()
        .find(|requirement| requirement.part == Part::Mtp)
        .expect("the TP1 boot plan carries the MTP stage");
    assert_eq!(mtp.geometry.world, 1);
    let tensors = ArchitectureTensorPlan::served_and_mtp(qwen_tp1.architecture(), qwen_tp1.tp())
        .expect("the Qwen TP1 served plan builds");
    assert!(
        tensors.roles().any(|role| role == Role::Mtp),
        "the TP1 served set carries the MTP tensor"
    );

    // A replicated stage (GLM's default) stays global at TP4, matching its served set.
    let glm_tp4 = EnginePlan::from_json(GLM, Target::Sm121, 4, 0).expect("GLM TP4");
    assert!(
        glm_tp4
            .architecture()
            .mtp
            .as_ref()
            .is_some_and(|mtp| mtp.replicated)
    );
    glm_tp4
        .globals()
        .iter()
        .find(|requirement| requirement.part == Part::Mtp)
        .expect("the GLM TP4 boot plan carries its replicated MTP stage");
}

/// A Gated DeltaNet block inside the DeepSeek stack runs whole on every rank: the layout states
/// every one of its tensors whole, so its scan is planned over every head at any world. A scan
/// planned over the stack's per-rank share would launch on only that share of the heads.
#[test]
fn a_gated_deltanet_block_in_an_mla_stack_scans_every_head_at_any_world() {
    let mut root: serde_json::Value = serde_json::from_str(DEEPSEEK).expect("DeepSeek config");
    let text = root["text_config"].as_object_mut().expect("text_config");
    let layers = text["num_hidden_layers"].as_u64().expect("layers") as usize;
    let mut types = vec![serde_json::Value::from("compressed_attention"); layers];
    types[0] = serde_json::Value::from(sparkflash_model::architecture::GDN_LAYER_TYPE);
    text.insert("layer_types".into(), serde_json::Value::from(types));
    for (key, value) in [
        ("linear_num_key_heads", 4),
        ("linear_num_value_heads", 8),
        ("linear_key_head_dim", 64),
        ("linear_value_head_dim", 32),
        ("linear_conv_kernel_dim", 4),
    ] {
        text.insert(key.into(), serde_json::Value::from(value));
    }
    text["compress_ratios"][0] = serde_json::Value::from(0);
    let architecture = Architecture::from_json(&root.to_string()).expect("one GDN block");
    for world in [1, 2, 4] {
        for rank in 0..world {
            let plan =
                EnginePlan::for_architecture(architecture.clone(), Target::Sm121, world, rank)
                    .unwrap_or_else(|error| panic!("TP{world}/{rank}: {error}"));
            let scan = plan
                .requirements()
                .find(|requirement| requirement.part == Part::GdnScan)
                .expect("the GDN block's scan");
            let geometry = scan.geometry;
            assert_eq!(
                (
                    geometry.heads,
                    geometry.head_offset,
                    geometry.kv_heads,
                    geometry.kv_head_offset
                ),
                (4, 0, 8, 0),
                "TP{world}/{rank}"
            );
        }
    }
}

/// A Kimi Delta Attention block inside the DeepSeek stack (GLM's `linear_attn_config`
/// spelling) splits by heads as the DeepSeek layout states it: each rank scans its share, and
/// the plan carries the block's GLM-named tensors and none of MLA's.
#[test]
fn a_kda_block_in_the_deepseek_stack_scans_its_ranks_share_of_the_heads() {
    let mut root: serde_json::Value = serde_json::from_str(DEEPSEEK).expect("DeepSeek config");
    let text = root["text_config"].as_object_mut().expect("text_config");
    let layers = text["num_hidden_layers"].as_u64().expect("layers") as usize;
    let mut types = vec![serde_json::Value::from("compressed_attention"); layers];
    types[0] = serde_json::Value::from(sparkflash_model::architecture::GDN_LAYER_TYPE);
    text.insert("layer_types".into(), serde_json::Value::from(types));
    text.insert(
        "linear_attn_config".into(),
        serde_json::json!({
            "num_heads": 16,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [0],
        }),
    );
    text["compress_ratios"][0] = serde_json::Value::from(0);
    let architecture = Architecture::from_json(&root.to_string()).expect("one KDA block");
    for world in [1, 2, 4] {
        for rank in 0..world {
            let plan =
                EnginePlan::for_architecture(architecture.clone(), Target::Sm121, world, rank)
                    .unwrap_or_else(|error| panic!("TP{world}/{rank}: {error}"));
            let share = 16 / world;
            let scan = plan
                .requirements()
                .find(|requirement| requirement.part == Part::KdaScan)
                .expect("the KDA block's scan");
            assert_eq!(
                (scan.geometry.heads, scan.geometry.head_offset),
                (share, rank * share),
                "TP{world}/{rank}"
            );
            assert!(
                !plan
                    .requirements()
                    .any(|requirement| requirement.part == Part::GdnScan),
                "TP{world}/{rank}: a KDA block is not a Gated DeltaNet"
            );
        }
    }
}

/// GLM's KDA blocks sit in an MLA stack too, but the layout states their projections split by
/// rows, so each rank scans its share of the 64 heads. The split is the layout's statement, not
/// a rule about recurrent blocks in an MLA stack. The stack-wide attention facts are the sparse
/// MLA layers', and the KDA blocks carry their own.
#[test]
fn glm_kda_blocks_scan_their_ranks_share_of_the_heads() {
    for world in [1, 2, 4] {
        for rank in 0..world {
            let plan = EnginePlan::from_json(GLM, Target::Sm121, world, rank)
                .unwrap_or_else(|error| panic!("GLM TP{world}/{rank}: {error}"));
            let share = 64 / world;
            let scans: Vec<_> = plan
                .requirements()
                .filter(|requirement| requirement.part == Part::KdaScan)
                .collect();
            assert_eq!(scans.len(), 34, "TP{world}/{rank}: one scan a KDA block");
            for scan in scans {
                assert_eq!(
                    (scan.geometry.heads, scan.geometry.head_offset),
                    (share, rank * share),
                    "TP{world}/{rank}"
                );
            }
            let facts = plan.model_facts();
            let int = |name: &str| match facts.get(name) {
                Some(sparkflash_model::stack::FactValue::Integer(value)) => *value,
                other => panic!("TP{world}/{rank}: {name} is {other:?}"),
            };
            assert_eq!(int("kda_heads"), 64);
            assert_eq!(int("kda_head_dim"), 128);
            assert_eq!(int("kda_conv"), 4);
            assert_eq!(int("n_heads"), 64, "the sparse MLA layers' heads");
            assert_eq!(int("head_dim"), 256, "the sparse MLA layers' query width");
            match facts.get("kda_layers") {
                Some(sparkflash_model::stack::FactValue::Integers(layers)) => {
                    assert_eq!(layers.len(), 34);
                    assert!(!layers.contains(&3) && layers.contains(&0));
                }
                other => panic!("kda_layers is {other:?}"),
            }
        }
    }
}

/// The base forward reads a checkpoint by its own spelling: GLM's layout gives every tensor a
/// slot in DeepSeek's names, so the forward reads it with no GLM name in its code, and the plan
/// claims each by that slot. DeepSeek's names are the base's, so its layout gives none.
#[test]
fn the_glm_layout_names_every_tensor_by_a_base_slot() {
    let glm = Architecture::from_json(GLM).expect("GLM");
    let names = sparkflash_model::layout::checkpoint_names(&glm).expect("the GLM names");
    let layout = sparkflash_model::layout::matching(&glm)
        .expect("layouts")
        .expect("the GLM layout");
    let tensors = layout.tensors(&glm).expect("the GLM tensors");
    let missing: Vec<&str> = tensors
        .iter()
        .filter(|tensor| tensor.slot.is_none())
        .map(|tensor| tensor.name.as_str())
        .collect();
    assert!(
        missing.is_empty(),
        "GLM tensors without a slot: {missing:?}"
    );
    let dense = tensors
        .iter()
        .filter(|tensor| {
            matches!(
                tensor.dtype,
                sparkflash_model::layout::LayoutDtype::Dense
                    | sparkflash_model::layout::LayoutDtype::Expert
            )
        })
        .count();
    assert_eq!(names.len(), tensors.len() + dense);
    assert_eq!(
        names.get("model.language_model.layers.3.self_attn.q_a_proj.weight_scale_inv"),
        Some(&"layers.3.attn.wq_a.scale".to_owned())
    );
    let mtp = glm.layers.len();
    for (stored, base) in [
        ("lm_head.weight".to_owned(), "head.weight".to_owned()),
        (
            "model.language_model.layers.3.self_attn.kv_b_proj.weight".to_owned(),
            "layers.3.attn.wkv_b.weight".to_owned(),
        ),
        (
            "model.language_model.layers.3.mlp.experts.7.down_proj.weight".to_owned(),
            "layers.3.ffn.experts.7.w2.weight".to_owned(),
        ),
        (
            format!("model.language_model.layers.{mtp}.self_attn.q_a_proj.weight"),
            "mtp.0.attn.wq_a.weight".to_owned(),
        ),
        (
            format!("model.language_model.layers.{mtp}.eh_proj.weight"),
            "mtp.0.eh_proj.weight".to_owned(),
        ),
    ] {
        assert_eq!(names.get(&stored), Some(&base), "{stored}");
    }

    let plan = EnginePlan::from_json(GLM, Target::Sm121, 4, 0).expect("GLM TP4 rank 0");
    {
        for stack in plan.stacks() {
            for block in &stack.blocks {
                for instance in &block.parts {
                    for claim in instance.claims.iter().filter(|claim| claim.slot.is_some()) {
                        assert_eq!(
                            Some(claim.base_name()),
                            names.get(&claim.name).map(String::as_str),
                            "{}",
                            claim.name
                        );
                    }
                }
            }
        }
    }

    let deepseek = Architecture::from_json(DEEPSEEK).expect("DeepSeek");
    assert!(
        sparkflash_model::layout::checkpoint_names(&deepseek)
            .expect("the DeepSeek names")
            .is_empty()
    );
}

/// The host reference and the device forward fill one value table from one description: the
/// host from the config text (sparkflash_model_deepseek_v41::described::from_config), the device
/// from its plan's facts and resolved dense format. Both routes agree, and for the released
/// DeepSeek both are the table the DeepSeek config parser reads.
#[test]
fn the_host_and_the_plan_fill_one_value_table() {
    use sparkflash_model_deepseek_v41::config::ModelConfig;
    use sparkflash_model_deepseek_v41::described;
    let route = |text: &str| {
        let plan = EnginePlan::from_json(text, Target::Sm121, 4, 0).expect("plan");
        described::from_facts(
            plan.model_facts(),
            plan.architecture(),
            plan.stored_format(sparkflash_model::Part::DenseMlp, None),
        )
    };
    let released = ModelConfig::from_json(DEEPSEEK).expect("the DeepSeek parser");
    assert_eq!(described::from_config(DEEPSEEK).expect("host"), released);
    assert_eq!(route(DEEPSEEK).expect("plan"), released);
    let glm = glm_bf16();
    for text in [GLM, glm.as_str()] {
        assert_eq!(
            described::from_config(text).map_err(|error| error.to_string()),
            route(text).map_err(|error| error.to_string()),
        );
    }
}

/// Each arm refuses the wiring it has not built, by name: the host runs GLM's described wiring on
/// its whole-model path, the device's base walk names GLM's absorbed latent and pooled indexer,
/// the released DeepSeek opens on both, and the KDA facts the shared body runs are not named.
#[test]
fn an_arm_refuses_the_wiring_it_does_not_run_by_name() {
    use sparkflash_model_deepseek_v41::described;
    described::host_config(DEEPSEEK).expect("the released DeepSeek opens on the host");
    described::host_config(GLM).expect("GLM's described wiring opens on the host");
    let facts = |text: &str| {
        EnginePlan::from_json(text, Target::Sm121, 1, 0)
            .expect("plan")
            .model_facts()
            .clone()
    };
    described::refuse_unrun_wiring(&facts(DEEPSEEK), "the device", &described::BASE_RUNS)
        .expect("the released DeepSeek's wiring is the base's");
    let refusal =
        described::refuse_unrun_wiring(&facts(GLM), "the device", &described::BASE_RUNS)
            .expect_err("GLM's stated wiring on the base walk");
    for unrun in [
        "language.mla_attention.form = absorbed",
        "language.sparse_indexer.scores = f32",
        "language.indexer.pool_compress = true",
        "language.draft.indexer_shared = true",
        "language.manifold_hyper_connection.mix_input = f32",
    ] {
        assert!(refusal.contains(unrun), "{unrun}: {refusal}");
    }
    assert!(!refusal.contains("kda_scan") && !refusal.contains("router_dtype"), "{refusal}");
}
