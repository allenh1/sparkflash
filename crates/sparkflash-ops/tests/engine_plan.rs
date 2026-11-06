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
use sparkflash_model::{Part, Target};
use sparkflash_ops::engine_plan::{EnginePlan, PrecisionPolicy};
use sparkflash_ops::plan::{KernelPhase, Precision};
use sparkflash_ops::resolver::{Availability, KernelResolver};

const DEEPSEEK: &str =
    include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
const QWEN: &str =
    include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");

fn released() -> [(&'static str, &'static str); 3] {
    [("DeepSeek", DEEPSEEK), ("GLM", GLM), ("Qwen", QWEN)]
}

#[test]
fn every_released_config_plans_at_tp1_and_tp4() {
    for (name, text) in released() {
        for world in [1, 4] {
            let plan = EnginePlan::from_json(text, Target::Sm121, world, 0)
                .unwrap_or_else(|error| panic!("{name} TP{world}: {error}"));
            assert_eq!(
                plan.layers().len(),
                plan.architecture().layers.len(),
                "{name} TP{world} layer count"
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

    let qwen = EnginePlan::from_json(QWEN, Target::Sm121, 4, 3).expect("Qwen TP4 rank 3");
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
    for (name, text) in released() {
        for world in [1, 2, 4] {
            let plan = EnginePlan::from_json(text, Target::Sm121, world, 0)
                .unwrap_or_else(|error| panic!("{name} TP{world}: {error}"));
            let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue)
                .with_family(plan.family());
            let resolved = resolver
                .resolve_all(&plan)
                .unwrap_or_else(|errors| panic!("{name} TP{world}: {errors:?}"));
            assert_eq!(resolved.len(), plan.requirements().count(), "{name}");
            assert!(resolved.iter().any(|kernel| kernel.part == Part::Moe));
            assert!(resolved.iter().all(|kernel| kernel.target == Target::Sm121));
        }
    }
}

#[test]
fn the_family_prefers_its_own_image_over_the_global_canonical_entry() {
    let qwen = EnginePlan::from_json(QWEN, Target::Sm121, 4, 0).expect("Qwen");
    let resolver =
        KernelResolver::new(Target::Sm121, Availability::Catalogue).with_family(qwen.family());
    let moe = qwen
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen MoE requirement");
    assert_eq!(resolver.resolve(moe).expect("MoE").image, Image::Qwen4Moe);

    let glm = EnginePlan::from_json(GLM, Target::Sm121, 4, 0).expect("GLM");
    let resolver =
        KernelResolver::new(Target::Sm121, Availability::Catalogue).with_family(glm.family());
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
        "GLM's token/head kernels are the shared Qwen image"
    );

    let deepseek = EnginePlan::from_json(DEEPSEEK, Target::Sm121, 4, 0).expect("DeepSeek");
    let resolver =
        KernelResolver::new(Target::Sm121, Availability::Catalogue).with_family(deepseek.family());
    let moe = deepseek
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a DeepSeek MoE requirement");
    assert_eq!(resolver.resolve(moe).expect("MoE").image, Image::Deepseek);
}

#[test]
fn the_deepseek_plan_resolves_with_runtime_images() {
    let plan = EnginePlan::from_json(DEEPSEEK, Target::Sm121, 4, 0).expect("DeepSeek plan");
    let resolver = KernelResolver::new(Target::Sm121, Availability::Runtime);
    let resolved = resolver.resolve_all(&plan).expect("runtime images");
    assert!(
        resolved
            .iter()
            .any(|kernel| kernel.image == Image::Deepseek)
    );
}

#[test]
fn a_family_entry_is_not_replaced_by_a_loadable_cross_family_image() {
    if Image::Qwen4Moe.available() {
        return;
    }
    let plan = EnginePlan::from_json(QWEN, Target::Sm121, 4, 0).expect("Qwen plan");
    let moe = plan
        .requirements()
        .find(|requirement| requirement.part == Part::Moe)
        .expect("a Qwen MoE requirement");
    let resolver =
        KernelResolver::new(Target::Sm121, Availability::Runtime).with_family(plan.family());
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
    let plan = EnginePlan::from_json(QWEN, Target::Sm121, 1, 0).expect("Qwen plan");
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
    assert_eq!(deepseek.precision().experts, Precision::NvFp4);

    let glm = EnginePlan::from_json(GLM, Target::Sm121, 4, 0).expect("GLM");
    assert_eq!(glm.precision().dense, Precision::Fp8E4M3);
    assert_eq!(glm.precision().experts, Precision::Fp8E4M3);
}

#[test]
fn a_release_whose_format_the_shared_parser_does_not_know_falls_back_to_bf16() {
    let plan = EnginePlan::from_json(QWEN, Target::Sm121, 1, 0).expect("Qwen");
    // The Qwen release's dense format is a mixed-precision umbrella; its experts are NVFP4.
    assert_eq!(plan.precision().dense, Precision::Bf16);
    assert_eq!(plan.precision().experts, Precision::NvFp4);
    assert_eq!(
        plan.precision(),
        PrecisionPolicy::from_architecture(plan.architecture())
    );
}
