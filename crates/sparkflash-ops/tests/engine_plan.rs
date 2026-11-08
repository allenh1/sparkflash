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
    GroupedFp8Body, KernelId, KernelPhase, PartGeometry, Precision, StreamId,
};
use sparkflash_ops::resolver::{Availability, KernelResolver, PlanContext, ResolvedKernel};
use sparkflash_ops::tensor_plan::{ArchitectureTensorPlan, Role};

const DEEPSEEK: &str =
    include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
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

fn resolve_for(document: &Released, world: usize, phase: KernelPhase) -> Vec<ResolvedKernel> {
    let plan = plan_for(document, world);
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    plan.requirements()
        .filter(|requirement| requirement.part != Part::Vision)
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
                if requirement.part == Part::Vision && skip_vision(document.name) {
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
        for failure in &failures {
            assert_eq!(
                failure.part,
                Part::Moe,
                "TP{world} refused an unexpected part: {failure}"
            );
            let message = failure.to_string();
            assert!(message.contains("does not equal 576"), "{message}");
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
                ..PartGeometry::new()
            },
            precision: plan.precision().dense,
        };
        assert_eq!(requirement.precision, Precision::Bf16, "{name}");
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
    let resolved = resolver
        .boot_check(&plan)
        .unwrap_or_else(|errors| panic!("boot check: {errors:?}"));
    assert!(!resolved.is_empty());
    assert_eq!(resolved.len(), 2 * plan.requirements().count());
}

#[test]
fn the_glm_fp8_arm_boot_checks_both_passes() {
    let plan = EnginePlan::from_json(GLM, Target::Sm121, 4, 0).expect("GLM FP8 TP4");
    let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
    let resolved = resolver
        .boot_check(&plan)
        .unwrap_or_else(|errors| panic!("GLM boot check: {errors:?}"));
    assert_eq!(resolved.len(), 2 * plan.requirements().count());
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
                .filter(|requirement| requirement.part != Part::Vision)
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
                ..PartGeometry::new()
            },
            precision: plan.precision().dense,
        };
        assert_eq!(dense.precision, Precision::Bf16, "Qwen FP8 TP{world}");
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
    assert_eq!(moe.precision, Precision::Fp8E4M3, "Qwen FP8 experts");
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
        assert_eq!(resolved.precision, Precision::Fp8E4M3, "{phase:?}");
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
    assert_eq!(moe.precision, Precision::Fp8E4M3, "GLM FP8 experts");
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
        let result = resolver.boot_check(&plan);
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
                    error.precision,
                    Precision::Bf16,
                    "{}: {error}",
                    document.name
                );
                assert!(
                    error.to_string().contains(&format!(
                        "no symbol accepting bf16 in the {} pass",
                        error.phase.name()
                    )),
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
            if skip_vision(document.name) {
                assert!(
                    errors.iter().any(|error| error.part == Part::Vision
                        && error.to_string().contains("vision patch 16")),
                    "{}: the 16-patch tower is still a catalogue refusal, even for a BF16 document: {errors:?}",
                    document.name
                );
            }
        } else if skip_vision(document.name) {
            let errors = result.expect_err("the 16-patch vision tower is refused");
            assert!(
                !errors.is_empty(),
                "{}: the refusal names no part",
                document.name
            );
            for error in &errors {
                assert_eq!(error.part, Part::Vision, "{}: {error}", document.name);
                assert!(
                    error.to_string().contains("vision patch 16"),
                    "{}: {error}",
                    document.name
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
                    errors.iter().all(|error| error.part == Part::Vision),
                    "{} TP{world}: the language parts resolve: {errors:?}",
                    document.name
                );
                assert!(
                    errors
                        .iter()
                        .any(|error| error.to_string().contains("vision patch 16")),
                    "{} TP{world}: the refusal names the patch: {errors:?}",
                    document.name
                );
            }
            let language = plan
                .requirements()
                .filter(|requirement| requirement.part != Part::Vision)
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
        assert_eq!(moe.precision, Precision::Fp8E4M3, "rank {rank}");
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
            .filter(|requirement| requirement.part != Part::Vision)
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
        resolver.resolve(moe).expect("MoE").precision,
        Precision::NvFp4
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
        Precision::NvFp4,
        "\"fp4\" is a 4-bit expert format, not BF16"
    );
    assert_eq!(
        deepseek.precision().mtp,
        Precision::NvFp4,
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
    assert_eq!(mtp.precision, Precision::Fp8E4M3);

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
        moe.precision,
        Precision::Fp8E4M3,
        "the routed experts are FP8"
    );
    let mtp = qwen_fp8
        .globals()
        .iter()
        .find(|requirement| requirement.part == Part::Mtp)
        .expect("Qwen's boot plan carries an MTP part");
    assert_eq!(mtp.precision, Precision::Fp8E4M3, "the MTP experts are FP8");
    let attention = qwen_fp8
        .requirements()
        .find(|requirement| requirement.part == Part::GqaAttention)
        .expect("a Qwen GQA requirement");
    assert_eq!(
        attention.precision,
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
        .find(|failure| failure.part == Part::Vision)
        .unwrap_or_else(|| panic!("no vision failure: {failures:?}"));
    let message = failure.to_string();
    assert!(message.contains("vision"), "{message}");
    assert!(message.contains("deepseek_ptx"), "{message}");
    assert!(message.contains("vision patch"), "{message}");
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
