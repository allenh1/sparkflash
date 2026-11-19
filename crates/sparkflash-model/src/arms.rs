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

//! One entry per serving arm: the registration the console, the ledger and the tests read.
//!
//! [ARMS] is the model table, not the serving paths. Each entry names the arm the console prints,
//! the registry weight selectors it serves, the profile TOML it carries, its default context,
//! its ledger declaration ([crate::ARM_LEDGERS]), the tokenizer and chat template it reads, and
//! the conversion defaults its checkpoint was released with. [crate::MODELS] remains the
//! registry: an arm's weights are selectors into it, and a test holds every one to a descriptor.
//!
//! A file that would have kept a second arm list reads this table instead. The serving dispatch
//! in the CLI is not derived from it (the audit's entry-path table is the list of sites a new
//! family still edits); this table makes the *model* registration one declaration.

use crate::descriptor::{
    DEFAULT_WEIGHTS_ID, DORY1_0_4B_BF16_WEIGHTS_ID, DORY1_0_4B_CONTEXT_TOKENS,
    GLM53_CONTEXT_TOKENS, GLM53_FLASH_WEIGHTS_ID, INKLING2_CONTEXT_TOKENS,
    INKLING2_SMALL_WEIGHTS_ID, QWEN36_CONTEXT_TOKENS, QWEN36_FLASH_35B_A3B_FP8_WEIGHTS_ID,
    QWEN38_CONTEXT_TOKENS, QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID, QWEN38_FLASH_NEXT_WEIGHTS_ID,
    SENSENOVA_LOOP_9B_CONTEXT_TOKENS, SENSENOVA_LOOP_9B_WEIGHTS_ID, V41_CONTEXT_TOKENS,
};
use crate::ledger::{ArmLedger, ledger_for_arm};

/// The tokenizer and chat template an arm reads.
///
/// The ids are data for the console and for the conversion/onboarding surfaces; the family's
/// codec still resolves the files (the audit records that registration as a remaining
/// non-[ARMS] edit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tokenizer {
    /// The tokenizer's identity: the checkpoint id whose `tokenizer.json` the arm reads.
    pub tokenizer_id: &'static str,
    /// The chat template the arm renders with.
    pub chat_template: &'static str,
}

/// How a checkpoint of this arm converts when the operator names no world size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conversion {
    /// The tensor-parallel world the released checkpoint converts at.
    pub world_size: usize,
    /// Whether the converter has a released pack for this family (rather than the dynamic path).
    pub released: bool,
}

/// One serving arm's registration.
///
/// Every item is required: an arm that forgets one does not compile. The marker types below
/// implement it, and [ARMS] is the one table the console and the tests read.
pub trait Arm: Sync {
    /// The arm name the console prints beside its ledger.
    fn id(&self) -> &'static str;
    /// The served model id (the descriptor's `id`) the console prints.
    fn model_id(&self) -> &'static str;
    /// The registry weight selectors this arm serves.
    fn weights(&self) -> &'static [&'static str];
    /// The `model_type` spellings this arm's carried profile answers for.
    fn families(&self) -> &'static [&'static str];
    /// The profile TOML this arm carries, when this build carries one.
    fn profile_toml(&self) -> Option<&'static str>;
    /// The window served when the operator names none.
    fn default_context(&self) -> usize;
    /// The ledger declaration, from [crate::ARM_LEDGERS].
    fn ledger(&self) -> &'static ArmLedger;
    /// The tokenizer and chat template the arm reads.
    fn tokenizer(&self) -> Tokenizer;
    /// The conversion defaults.
    fn conversion(&self) -> Conversion;
}

/// The numerics registry's status for an arm: the same classification `sparkflash arms` prints.
#[must_use]
pub fn numerics_status(arm: &dyn Arm) -> &'static str {
    let registry = match crate::numerics::Registry::embedded() {
        Ok(registry) => registry,
        Err(_) => return "none (the registry does not parse)",
    };
    let mut registered = false;
    let mut captured = false;
    for entry in &registry.config {
        let Some(model) = crate::find_model(&entry.model) else {
            continue;
        };
        if arm.weights().contains(&model.weights_id.as_str()) {
            registered = true;
            captured |= entry.is_captured();
        }
    }
    match (registered, captured) {
        (_, true) => "captured",
        (true, false) => "none (registered, uncaptured)",
        (false, false) => "none (no entry)",
    }
}

/// The released DeepSeek-V4.1-Flash arm.
pub struct DeepSeek;
impl Arm for DeepSeek {
    fn id(&self) -> &'static str {
        "DeepSeek"
    }
    fn model_id(&self) -> &'static str {
        crate::descriptor::DEFAULT_MODEL_ID
    }
    fn weights(&self) -> &'static [&'static str] {
        &[DEFAULT_WEIGHTS_ID]
    }
    fn families(&self) -> &'static [&'static str] {
        &["deepseek_v41"]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        Some(include_str!(
            "../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.toml"
        ))
    }
    fn default_context(&self) -> usize {
        V41_CONTEXT_TOKENS
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("DeepSeek").expect("the DeepSeek ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer_id: crate::descriptor::DEFAULT_MODEL_ID,
            chat_template: "deepseek",
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 4,
            released: true,
        }
    }
}

/// The nvidia NVFP4 Qwen3.8-Flash-Next arm.
pub struct Qwen;
impl Arm for Qwen {
    fn id(&self) -> &'static str {
        "Qwen"
    }
    fn model_id(&self) -> &'static str {
        crate::descriptor::QWEN38_FLASH_NEXT_ID
    }
    fn weights(&self) -> &'static [&'static str] {
        &[QWEN38_FLASH_NEXT_WEIGHTS_ID]
    }
    fn families(&self) -> &'static [&'static str] {
        &["qwen4_exp", "qwen4_exp_text"]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        Some(include_str!(
            "../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.toml"
        ))
    }
    fn default_context(&self) -> usize {
        QWEN38_CONTEXT_TOKENS
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("Qwen").expect("the Qwen ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer_id: crate::descriptor::QWEN38_FLASH_NEXT_ID,
            chat_template: "qwen",
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            released: true,
        }
    }
}

/// The official FP8 Qwen3.8-Flash-Next arm.
pub struct QwenFp8;
impl Arm for QwenFp8 {
    fn id(&self) -> &'static str {
        "Qwen (FP8)"
    }
    fn model_id(&self) -> &'static str {
        crate::descriptor::QWEN38_FLASH_NEXT_FP8_ID
    }
    fn weights(&self) -> &'static [&'static str] {
        &[QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID]
    }
    fn families(&self) -> &'static [&'static str] {
        &["qwen4_exp", "qwen4_exp_text"]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        Qwen.profile_toml()
    }
    fn default_context(&self) -> usize {
        QWEN38_CONTEXT_TOKENS
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("Qwen").expect("the Qwen ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer_id: crate::descriptor::QWEN38_FLASH_NEXT_FP8_ID,
            chat_template: "qwen",
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            released: true,
        }
    }
}

/// The official FP8 Qwen3.6-35B-A3B arm.
pub struct Qwen36;
impl Arm for Qwen36 {
    fn id(&self) -> &'static str {
        "Qwen3.6 (FP8)"
    }
    fn model_id(&self) -> &'static str {
        crate::descriptor::QWEN36_FLASH_35B_A3B_FP8_ID
    }
    fn weights(&self) -> &'static [&'static str] {
        &[QWEN36_FLASH_35B_A3B_FP8_WEIGHTS_ID]
    }
    fn families(&self) -> &'static [&'static str] {
        &["qwen3_5_moe", "qwen3_5_moe_text"]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        Some(include_str!(
            "../../sparkflash-model-qwen4-exp/config/qwen36_35b_a3b_fp8.toml"
        ))
    }
    fn default_context(&self) -> usize {
        QWEN36_CONTEXT_TOKENS
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("Qwen").expect("the Qwen ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer_id: crate::descriptor::QWEN36_FLASH_35B_A3B_FP8_ID,
            chat_template: "qwen",
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            released: false,
        }
    }
}

/// The zai-org GLM-5.3-Flash arm.
pub struct Glm;
impl Arm for Glm {
    fn id(&self) -> &'static str {
        "GLM"
    }
    fn model_id(&self) -> &'static str {
        crate::descriptor::GLM53_FLASH_ID
    }
    fn weights(&self) -> &'static [&'static str] {
        &[GLM53_FLASH_WEIGHTS_ID]
    }
    fn families(&self) -> &'static [&'static str] {
        &["glm5_next"]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        Some(include_str!(
            "../../sparkflash-model-glm5-next/config/glm53_flash.toml"
        ))
    }
    fn default_context(&self) -> usize {
        GLM53_CONTEXT_TOKENS
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("GLM").expect("the GLM ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer_id: crate::descriptor::GLM53_FLASH_ID,
            chat_template: "glm",
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 4,
            released: true,
        }
    }
}

/// The NVIDIA Nemotron Labs Dory1.0-4B arm.
pub struct Dory;
impl Arm for Dory {
    fn id(&self) -> &'static str {
        "Dory"
    }
    fn model_id(&self) -> &'static str {
        crate::descriptor::DORY1_0_4B_BF16_ID
    }
    fn weights(&self) -> &'static [&'static str] {
        &[DORY1_0_4B_BF16_WEIGHTS_ID]
    }
    fn families(&self) -> &'static [&'static str] {
        &["dory"]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        Some(include_str!(
            "../../sparkflash-model-dory/config/dory1_0_4b_bf16.toml"
        ))
    }
    fn default_context(&self) -> usize {
        DORY1_0_4B_CONTEXT_TOKENS
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("Dory").expect("the Dory ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer_id: crate::descriptor::DORY1_0_4B_BF16_ID,
            chat_template: "dory",
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            released: false,
        }
    }
}

/// The thinkingmachines Inkling-2-Small arm.
pub struct Inkling;
impl Arm for Inkling {
    fn id(&self) -> &'static str {
        "Inkling"
    }
    fn model_id(&self) -> &'static str {
        crate::descriptor::INKLING2_SMALL_ID
    }
    fn weights(&self) -> &'static [&'static str] {
        &[INKLING2_SMALL_WEIGHTS_ID]
    }
    fn families(&self) -> &'static [&'static str] {
        &["inkling2"]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        Some(include_str!(
            "../../sparkflash-model-inkling2/config/inkling2_small.toml"
        ))
    }
    fn default_context(&self) -> usize {
        INKLING2_CONTEXT_TOKENS
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("Inkling").expect("the Inkling ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer_id: crate::descriptor::INKLING2_SMALL_ID,
            chat_template: "inkling2",
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            released: true,
        }
    }
}


/// The SenseNova-Loop-9B arm: a Qwen3.5 hybrid whose stored 32 layers are visited 56 times.
///
/// The first arm on the dynamic route that owns a looped plan and a two-stream hyper-connection
/// router, which is why it is registered here rather than as a typed pack: the pack path has no
/// way to say "layer 4 through 28 twice".
pub struct SenseNova;
impl Arm for SenseNova {
    fn id(&self) -> &'static str {
        "SenseNova"
    }
    fn model_id(&self) -> &'static str {
        crate::descriptor::SENSENOVA_LOOP_9B_ID
    }
    fn weights(&self) -> &'static [&'static str] {
        &[SENSENOVA_LOOP_9B_WEIGHTS_ID]
    }
    fn families(&self) -> &'static [&'static str] {
        &["sensenova_loop"]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        Some(include_str!(
            "../../sparkflash-model-sensenova/config/sensenova_loop_9b_fp8.toml"
        ))
    }
    fn default_context(&self) -> usize {
        SENSENOVA_LOOP_9B_CONTEXT_TOKENS
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("SenseNova").expect("the SenseNova ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            // The release ships a Qwen tokenizer and a ChatML template; the arm renders with
            // the shared Qwen codec, not the checkpoint's own jinja.
            tokenizer_id: crate::descriptor::SENSENOVA_LOOP_9B_ID,
            chat_template: "qwen",
        }
    }
    fn conversion(&self) -> Conversion {
        // The FP8 pack is a conversion of the released checkpoint rather than a release of its
        // own, so the dynamic path is what boots it.
        Conversion {
            world_size: 1,
            released: false,
        }
    }
}

/// The shared dense arm: the walk a dynamic family lands on. It serves no registry selector
/// and carries no profile; the checkpoint's own document supplies both the window and the
/// tokenizer.
pub struct Dense;
impl Arm for Dense {
    fn id(&self) -> &'static str {
        "dense"
    }
    fn model_id(&self) -> &'static str {
        "the checkpoint"
    }
    fn weights(&self) -> &'static [&'static str] {
        &[]
    }
    fn families(&self) -> &'static [&'static str] {
        &[]
    }
    fn profile_toml(&self) -> Option<&'static str> {
        None
    }
    fn default_context(&self) -> usize {
        0
    }
    fn ledger(&self) -> &'static ArmLedger {
        ledger_for_arm("dense").expect("the dense ledger is declared")
    }
    fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer_id: "the checkpoint's tokenizer.json",
            chat_template: "dynamic",
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            released: false,
        }
    }
}

/// Every serving arm this build knows, in the console's order.
///
/// One entry per serving arm: the three Qwen releases are three entries because they are three
/// selectors with their own checkpoints, and they share the Qwen ledger and profile. The dense
/// arm is last because it serves no registry model.
pub static ARMS: &[&dyn Arm] = &[
    &DeepSeek, &Qwen, &QwenFp8, &Qwen36, &Glm, &Dory, &Inkling, &SenseNova, &Dense,
];

/// The arm whose carried profile answers for `family`, if any.
#[must_use]
pub fn arm_for_family(family: &str) -> Option<&'static dyn Arm> {
    ARMS.iter()
        .copied()
        .find(|arm| arm.families().contains(&family))
}

/// The arm whose weight selectors include `weights`, if any.
#[must_use]
pub fn arm_for_weights(weights: &str) -> Option<&'static dyn Arm> {
    ARMS.iter()
        .copied()
        .find(|arm| arm.weights().contains(&weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_arm_is_registered_once_and_names_its_items() {
        let mut ids = Vec::new();
        for arm in ARMS {
            assert!(!arm.id().is_empty(), "an arm has no console id");
            assert!(
                !arm.model_id().is_empty(),
                "{} has no served model id",
                arm.id()
            );
            assert!(!ids.contains(&arm.id()), "{} is listed twice", arm.id());
            ids.push(arm.id());
            assert!(!arm.tokenizer().tokenizer_id.is_empty());
            assert!(!arm.tokenizer().chat_template.is_empty());
            assert!(arm.conversion().world_size >= 1);
            assert_eq!(
                arm.ledger().unstated(),
                Vec::new(),
                "{} leaves a strategy out",
                arm.id()
            );
            if let Some(text) = arm.profile_toml() {
                crate::ModelProfile::from_toml(text)
                    .unwrap_or_else(|error| panic!("{}'s profile: {error}", arm.id()));
            }
        }
    }

    #[test]
    fn every_weights_selector_resolves_in_the_registry() {
        for arm in ARMS {
            for weights in arm.weights() {
                assert!(
                    crate::find_model_by_weights(weights).is_some(),
                    "{} names {weights}, which the registry does not know",
                    arm.id()
                );
            }
        }
    }

    #[test]
    fn every_registered_model_has_an_arm() {
        for model in crate::MODELS.iter() {
            assert!(
                arm_for_weights(&model.weights_id).is_some(),
                "{} ({}) has no ARMS entry",
                model.id,
                model.weights_id
            );
        }
    }

    #[test]
    fn every_arm_has_a_numerics_status() {
        for arm in ARMS {
            let status = numerics_status(*arm);
            assert!(
                status == "captured"
                    || status == "none (registered, uncaptured)"
                    || status == "none (no entry)",
                "{} has no numerics status: {status}",
                arm.id()
            );
        }
    }
}
