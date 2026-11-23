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
//! its ledger declaration ([crate::ARM_LEDGERS]), the tokenizer and prompt dialect it reads, the
//! converter its checkpoint goes through, the defaults its group serves by, the row the web
//! console launches it from and the crate and switches the scripts know it by. [crate::MODELS]
//! remains the registry: an arm's weights are selectors into it, and a test holds every one to a
//! descriptor.
//!
//! A file that would have kept a second arm list reads this table instead: the CLI's codec and
//! response engine, its group defaults and its converter dispatch, the console's deployment rows,
//! and (through the tests in `tests/arms.rs`) the launch scripts. The code a family brings (its
//! codec, its converter, its backend) stays in its own crate; this table names which one answers.

use crate::descriptor::{
    DEFAULT_WEIGHTS_ID, DORY1_0_4B_BF16_WEIGHTS_ID, DORY1_0_4B_CONTEXT_TOKENS,
    GLM53_CONTEXT_TOKENS, GLM53_FLASH_WEIGHTS_ID, INKLING2_CONTEXT_TOKENS,
    INKLING2_SMALL_WEIGHTS_ID, QWEN36_CONTEXT_TOKENS, QWEN36_FLASH_35B_A3B_FP8_WEIGHTS_ID,
    QWEN38_CONTEXT_TOKENS, QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID, QWEN38_FLASH_NEXT_WEIGHTS_ID,
    V41_CONTEXT_TOKENS,
};
use crate::ledger::{ArmLedger, ledger_for_arm};

/// The lanes a lane group steps when its operator names none: the CLI's default for a
/// `--tp4-rank` group whose arm opens its ranks as lanes, the dynamic dense arm's own default,
/// and what the web console writes into a lane group's plan.
pub const GROUP_LANES: usize = 4;

/// The control-plane engine selectors the web console launches an arm by ([Deployment::engine]).
pub mod engine {
    /// The TP=4 V4.1 engine.
    pub const DEEPSEEK: &str = "deepseek-v4.1-flash";
    /// The Qwen pack.
    pub const QWEN: &str = "qwen4-exp";
    /// The GLM-5.3-Flash pack.
    pub const GLM: &str = "glm5-next";
    /// The Inkling-2-Small pack.
    pub const INKLING: &str = "inkling2";
    /// The dynamic engine: the checkpoint's own arm, chosen when it is opened.
    pub const DYNAMIC: &str = "dynamic";
}

/// A prompt dialect: the codec that renders and tokenizes a chat, and the response engine that
/// reads the reply back into messages. The family's crate holds both; the serving path maps a
/// dialect to them in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// DeepSeek's own renderer, which its reference path normally serves.
    Deepseek,
    /// ChatML's `<|im_start|>`/`<|im_end|>` turns with Qwen's template.
    Qwen,
    /// GLM's `[gMASK]<sop>` and `<|user|>` turns.
    Glm,
    /// ChatML's turn tokens under Dory's released template; replies read as Qwen's.
    Dory,
    /// Inkling-2's typed messages: `<|message_model|><|content_thinking|>...<|end_message|>`.
    Inkling,
}

impl Dialect {
    /// The chat template's name, as the console prints it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Deepseek => "deepseek",
            Self::Qwen => "qwen",
            Self::Glm => "glm",
            Self::Dory => "dory",
            Self::Inkling => "inkling2",
        }
    }
}

/// The tokenizer and prompt dialect an arm reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tokenizer {
    /// The tokenizer's identity: the checkpoint id whose `tokenizer.json` the arm reads.
    pub tokenizer_id: &'static str,
    /// The dialect the arm declares, or `None` when the checkpoint's own tokenizer says (the
    /// shared dense arm serves whatever family lands on it).
    pub dialect: Option<Dialect>,
}

impl Tokenizer {
    /// The chat template the arm renders with, as the console prints it.
    #[must_use]
    pub fn chat_template(&self) -> &'static str {
        self.dialect.map_or("dynamic", Dialect::name)
    }
}

/// The converter a family's checkpoint goes through. The converters are the families' own code;
/// the CLI maps each variant to its function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Converter {
    /// DeepSeek's resident converter and its manifest.
    DeepSeek,
    /// The Qwen pack's walk-ordered rank files, with their NVFP4 scale pairs.
    Qwen,
    /// The GLM pack's walk-ordered rank files.
    Glm,
    /// Inkling-2's plan-ordered rank files, verified slot by slot against the checkpoint.
    Inkling,
    /// The shared architecture walk: every tensor the dynamic path reads, in walk order, at TP1.
    Walk,
}

/// How a checkpoint of this arm converts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conversion {
    /// The tensor-parallel world the released checkpoint converts at when the operator names
    /// none.
    pub world_size: usize,
    /// The converter the checkpoint goes through.
    pub converter: Converter,
    /// The model, as a conversion error names it.
    pub name: &'static str,
    /// Whether `--no-pad-moe-640` means something to this converter. The flag is DeepSeek's
    /// routed-expert layout; a pack whose files have one fixed layout refuses it.
    pub takes_no_pad: bool,
}

impl Conversion {
    /// Whether the converter has a released pack for this family (rather than the shared walk).
    #[must_use]
    pub fn released(&self) -> bool {
        self.converter != Converter::Walk
    }
}

/// The defaults a group of this arm serves by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupDefaults {
    /// The window the arm's device group admits when the operator names none, when that is
    /// narrower than the arm's whole window: every rank allocates its KV at the admitted context.
    pub context: Option<usize>,
    /// The lanes `serve` steps for a group of this arm when the operator names none. An arm
    /// whose group serves one request at a time unless told otherwise says 1.
    pub lanes: usize,
}

/// The web console's row for an arm: the deployment fields the model registry does not own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deployment {
    /// The engine selector that runs the arm's weights ([engine]).
    pub engine: &'static str,
    /// The default Hugging Face search term for this family.
    pub search: &'static str,
    /// How many ranks the console launches the family on.
    pub world_size: usize,
    /// The organization most of the family's checkpoints live under.
    pub org: &'static str,
    /// Whether launch needs a per-rank converted file.
    pub requires_conversion: bool,
    /// Whether launch loads the external cubin directory built by scripts/build-cubins.sh.
    pub requires_cubins: bool,
    /// Whether the four-rank group steps concurrent requests as fused lanes.
    pub lane_group: bool,
}

/// Which serving code boots an arm's checkpoint.
///
/// The arm states it, so the serving table cannot disagree with it: an arm that says [Self::Pack]
/// and has no row in the CLI's typed-pack table fails a test there instead of falling through to
/// the dynamic route at boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// The V4.1 reference/TP4 engine.
    Reference,
    /// A typed pack arm behind the generic host driver: a `Release` impl and one typed-pack row.
    Pack,
    /// The dynamic path: the checkpoint's own arm, opened from its configuration.
    Dynamic,
}

/// Where an arm's code lives and how the scripts name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Source {
    /// The crate that carries the arm.
    pub crate_name: &'static str,
    /// The serving code that boots the arm.
    pub route: Route,
    /// The prefixes of the environment switches that are the arm's own. Every rank of a group
    /// must see the same values, so a group launcher forwards each one.
    pub switches: &'static [&'static str],
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
    /// The tokenizer and prompt dialect the arm reads.
    fn tokenizer(&self) -> Tokenizer;
    /// How the arm's checkpoint converts.
    fn conversion(&self) -> Conversion;
    /// The defaults a group of this arm serves by.
    fn group(&self) -> GroupDefaults;
    /// The numerics report that holds the arm's tensor-core decode back, when its decode is the
    /// row GEMV ([check_profile]); `None` for an arm that takes `decode.tensor_cores`.
    fn row_decode_report(&self) -> Option<&'static str>;
    /// The web console's row, for an arm the console launches by a registry selector.
    fn deployment(&self) -> Option<Deployment>;
    /// The arm's crate and its own environment switches.
    fn source(&self) -> Source;
}

/// Refuses a profile `arm` does not serve: `decode.tensor_cores = true` on an arm whose decode
/// is the row GEMV, in the one message every such arm gives.
///
/// # Errors
/// As [crate::profile::refuse_tensor_core_decode].
pub fn check_profile(arm: &dyn Arm, profile: &crate::ModelProfile) -> Result<(), String> {
    match arm.row_decode_report() {
        Some(report) => {
            crate::profile::refuse_tensor_core_decode(arm.id(), arm.model_id(), report, profile)
        }
        None => Ok(()),
    }
}

/// The numerics registry's status for an arm: the same classification `sparkflash arms` prints.
#[must_use]
pub fn numerics_status(arm: &dyn Arm) -> &'static str {
    weights_numerics_status(arm.weights())
}

/// The numerics registry's status for a set of weight selectors: `captured` when any has a
/// capture, else whether any has an entry at all.
#[must_use]
pub fn weights_numerics_status(weights: &[&str]) -> &'static str {
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
        if weights.contains(&model.weights_id.as_str()) {
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
            dialect: Some(Dialect::Deepseek),
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 4,
            converter: Converter::DeepSeek,
            name: "DeepSeek-V4.1",
            takes_no_pad: true,
        }
    }
    fn group(&self) -> GroupDefaults {
        GroupDefaults {
            context: None,
            lanes: 1,
        }
    }
    fn row_decode_report(&self) -> Option<&'static str> {
        None
    }
    fn deployment(&self) -> Option<Deployment> {
        Some(Deployment {
            engine: engine::DEEPSEEK,
            search: "DeepSeek-V4.1",
            world_size: 4,
            // The V4.1 engine is the four-rank group; it has no narrower arm.
            org: "deepseek-ai",
            requires_conversion: true,
            requires_cubins: true,
            lane_group: false,
        })
    }
    fn source(&self) -> Source {
        Source {
            crate_name: "sparkflash-model-deepseek-v41",
            route: Route::Reference,
            switches: &[],
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
            dialect: Some(Dialect::Qwen),
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            converter: Converter::Qwen,
            name: "Qwen4-Exp",
            takes_no_pad: false,
        }
    }
    fn group(&self) -> GroupDefaults {
        GroupDefaults {
            context: None,
            lanes: GROUP_LANES,
        }
    }
    fn row_decode_report(&self) -> Option<&'static str> {
        None
    }
    fn deployment(&self) -> Option<Deployment> {
        Some(Deployment {
            engine: engine::QWEN,
            search: "Qwen3.8-Flash",
            world_size: 1,
            // The pack's single-node arm and its TP=2/TP=4 group (!205, !207).
            org: "Qwen",
            requires_conversion: true,
            requires_cubins: false,
            lane_group: false,
        })
    }
    fn source(&self) -> Source {
        Source {
            crate_name: "sparkflash-model-qwen4-exp",
            route: Route::Pack,
            switches: &["SPARKFLASH_QWEN4_"],
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
            dialect: Some(Dialect::Qwen),
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            converter: Converter::Qwen,
            name: "Qwen4-Exp",
            takes_no_pad: false,
        }
    }
    fn group(&self) -> GroupDefaults {
        GroupDefaults {
            context: None,
            lanes: GROUP_LANES,
        }
    }
    fn row_decode_report(&self) -> Option<&'static str> {
        None
    }
    fn deployment(&self) -> Option<Deployment> {
        Qwen.deployment()
    }
    fn source(&self) -> Source {
        Qwen.source()
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
            dialect: Some(Dialect::Qwen),
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            converter: Converter::Walk,
            name: "dynamic",
            takes_no_pad: true,
        }
    }
    fn group(&self) -> GroupDefaults {
        GroupDefaults {
            context: None,
            lanes: GROUP_LANES,
        }
    }
    fn row_decode_report(&self) -> Option<&'static str> {
        None
    }
    fn deployment(&self) -> Option<Deployment> {
        Some(Deployment {
            // The dynamic engine launches Qwen3.5-MoE from the checkpoint: the FP8 release on
            // the Qwen pack's device arm, a BF16 one on the shared CPU host reference, both at
            // one rank (sparkflash_ops::parts::dynamic_device_arm decides which).
            engine: engine::DYNAMIC,
            search: "Qwen3.6",
            world_size: 1,
            org: "Qwen",
            requires_conversion: false,
            requires_cubins: false,
            lane_group: false,
        })
    }
    fn source(&self) -> Source {
        Qwen.source()
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
            dialect: Some(Dialect::Glm),
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 4,
            converter: Converter::Glm,
            name: "GLM-5.3",
            takes_no_pad: false,
        }
    }
    fn group(&self) -> GroupDefaults {
        GroupDefaults {
            context: None,
            lanes: GROUP_LANES,
        }
    }
    fn row_decode_report(&self) -> Option<&'static str> {
        Some("docs/numerics/glm53/v3-gemv.md")
    }
    fn deployment(&self) -> Option<Deployment> {
        Some(Deployment {
            engine: engine::GLM,
            search: "GLM-5.3-Flash",
            world_size: 4,
            // The pack's single-node arm (which refuses the released 328 GB on memory, not on
            // capability) and its TP=2/TP=4 group.
            org: "zai-org",
            requires_conversion: true,
            requires_cubins: false,
            lane_group: true,
        })
    }
    fn source(&self) -> Source {
        Source {
            crate_name: "sparkflash-model-glm5-next",
            route: Route::Pack,
            switches: &["SPARKFLASH_GLM_"],
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
            dialect: Some(Dialect::Dory),
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            converter: Converter::Walk,
            name: "dynamic",
            takes_no_pad: true,
        }
    }
    fn group(&self) -> GroupDefaults {
        GroupDefaults {
            context: None,
            lanes: 1,
        }
    }
    fn row_decode_report(&self) -> Option<&'static str> {
        Some("docs/numerics/dory10/v2-gemv.md")
    }
    fn deployment(&self) -> Option<Deployment> {
        Some(Deployment {
            // The dynamic route opens Dory's device arm when its cubins are present, and its
            // CPU host reference otherwise. Dory's multi-rank group splits recurrent loops,
            // not layers, so it is not one of the console's tensor-parallel worlds.
            engine: engine::DYNAMIC,
            search: "Dory1.0",
            world_size: 1,
            org: "nvidia",
            requires_conversion: false,
            requires_cubins: false,
            lane_group: false,
        })
    }
    fn source(&self) -> Source {
        Source {
            crate_name: "sparkflash-model-dory",
            route: Route::Dynamic,
            // Its knobs are the profile's; SPARKFLASH_DORY_CUBIN is read when the kernels build.
            switches: &[],
        }
    }
}

/// The window Inkling's device group admits when the operator names none.
///
/// The release's window is 1M tokens; 256K is the arm's documented serving default, and the
/// difference is 6 GiB of KV a rank.
pub const INKLING2_GROUP_CONTEXT_TOKENS: usize = 262_144;

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
            dialect: Some(Dialect::Inkling),
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            converter: Converter::Inkling,
            name: "Inkling-2-Small",
            takes_no_pad: true,
        }
    }
    fn group(&self) -> GroupDefaults {
        GroupDefaults {
            context: Some(INKLING2_GROUP_CONTEXT_TOKENS),
            lanes: 1,
        }
    }
    fn row_decode_report(&self) -> Option<&'static str> {
        Some("docs/numerics/inkling2/v2-clean-head.md")
    }
    fn deployment(&self) -> Option<Deployment> {
        Some(Deployment {
            engine: engine::INKLING,
            search: "Inkling-2",
            world_size: 4,
            // The 393.5 GB checkpoint fits four Sparks; it does not fit one or two.
            org: "thinkingmachines",
            requires_conversion: true,
            // The arm's kernels are built into the binary (SPARKFLASH_INKLING_CUBIN at build
            // time); its group takes no cubin directory.
            requires_cubins: false,
            lane_group: true,
        })
    }
    fn source(&self) -> Source {
        Source {
            crate_name: "sparkflash-model-inkling2",
            route: Route::Dynamic,
            switches: &["SPARKFLASH_INKLING_"],
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
            dialect: None,
        }
    }
    fn conversion(&self) -> Conversion {
        Conversion {
            world_size: 1,
            converter: Converter::Walk,
            name: "dynamic",
            takes_no_pad: true,
        }
    }
    fn group(&self) -> GroupDefaults {
        GroupDefaults {
            context: None,
            lanes: GROUP_LANES,
        }
    }
    fn row_decode_report(&self) -> Option<&'static str> {
        None
    }
    fn deployment(&self) -> Option<Deployment> {
        None
    }
    fn source(&self) -> Source {
        Source {
            crate_name: "sparkflash-model-dense",
            route: Route::Dynamic,
            switches: &["SPARKFLASH_DENSE_"],
        }
    }
}

/// Every serving arm this build knows, in the console's order.
///
/// One entry per serving arm: the three Qwen releases are three entries because they are three
/// selectors with their own checkpoints, and they share the Qwen ledger and profile. The dense
/// arm is last because it serves no registry model.
pub static ARMS: &[&dyn Arm] = &[
    &DeepSeek, &Qwen, &QwenFp8, &Qwen36, &Glm, &Dory, &Inkling, &Dense,
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

/// The crate an arm's code lives in, by the arm's id, its model id, its crate's name or that
/// name without `sparkflash-model-`, ignoring case. `sparkflash arms --crate` prints it, and
/// `scripts/onboard-gate.sh` reads an arm's conformance crate through that, so the script keeps no list of its own.
#[must_use]
pub fn crate_for(name: &str) -> Option<&'static str> {
    ARMS.iter()
        .map(|arm| (arm, arm.source().crate_name))
        .find_map(|(arm, krate)| {
            let short = krate.strip_prefix("sparkflash-model-").unwrap_or(krate);
            [arm.id(), arm.model_id(), krate, short]
                .iter()
                .any(|known| known.eq_ignore_ascii_case(name))
                .then_some(krate)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every arm resolves to its own crate by its id, its model id and its crate's short name,
    /// and a name no arm has resolves to none.
    #[test]
    fn an_arm_names_its_crate() {
        for arm in ARMS {
            let krate = arm.source().crate_name;
            let short = krate.strip_prefix("sparkflash-model-").unwrap_or(krate);
            for name in [arm.id(), arm.model_id(), krate, short] {
                let found = crate_for(&name.to_ascii_uppercase());
                assert_eq!(found, Some(krate), "{name} resolves to {found:?}");
            }
        }
        assert_eq!(crate_for("glm"), Some("sparkflash-model-glm5-next"));
        assert_eq!(crate_for("no-such-arm"), None);
    }

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
            assert!(!arm.tokenizer().chat_template().is_empty());
            assert!(arm.conversion().world_size >= 1);
            assert!(!arm.conversion().name.is_empty());
            assert!(arm.group().lanes >= 1, "{} steps no lane", arm.id());
            assert!(
                arm.source().crate_name.starts_with("sparkflash-model-"),
                "{} names no model crate",
                arm.id()
            );
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
