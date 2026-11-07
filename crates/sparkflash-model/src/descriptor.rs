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

//! Model descriptors and the compile-time registry.

use std::sync::LazyLock;

use crate::capabilities::Capabilities;

/// The canonical id of the released V4.1-Flash checkpoint.
pub const DEFAULT_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

/// The control-plane weight selector for the released V4.1-Flash checkpoint.
pub const DEFAULT_WEIGHTS_ID: &str = "deepseek-v4.1-flash";

/// The served id of the NVIDIA NVFP4 Qwen3.8-Flash-Next checkpoint.
pub const QWEN38_FLASH_NEXT_ID: &str = "nvidia/Qwen3.8-Flash-Next-NVFP4";

/// The control-plane weight selector for the Qwen3.8-Flash-Next checkpoint.
pub const QWEN38_FLASH_NEXT_WEIGHTS_ID: &str = "qwen3.8-flash-next-nvfp4";

/// The served id of the official Qwen3.8-Flash-Next FP8 checkpoint.
pub const QWEN38_FLASH_NEXT_FP8_ID: &str = "Qwen/Qwen3.8-Flash-Next-FP8";

/// The control-plane weight selector for the official FP8 checkpoint.
pub const QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID: &str = "qwen3.8-flash-next-fp8";

/// The context window of the released Qwen3.8-Flash-Next checkpoint.
pub const QWEN38_CONTEXT_TOKENS: usize = 262_144;

/// The output ceiling this build serves for Qwen3.8-Flash-Next.
pub const QWEN38_MAX_OUTPUT_TOKENS: usize = 32_768;

/// The vocabulary of the released Qwen3.8-Flash-Next checkpoint (padded).
pub const QWEN38_VOCAB_SIZE: usize = 248_320;

/// The served id of the Z.ai GLM-5.3-Flash checkpoint.
pub const GLM53_FLASH_ID: &str = "zai-org/GLM-5.3-Flash";

/// The control-plane weight selector for the GLM-5.3-Flash FP8 checkpoint.
pub const GLM53_FLASH_WEIGHTS_ID: &str = "glm5.3-flash-fp8";

/// The context window of the released GLM-5.3-Flash checkpoint.
pub const GLM53_CONTEXT_TOKENS: usize = 1_048_576;

/// The per-request output ceiling this build serves for GLM-5.3-Flash.
///
/// The same 32,768 the other packed arm caps a request at; the context window and the dense KV
/// allocation, not this number, are what refuse a longer request.
pub const GLM53_MAX_OUTPUT_TOKENS: usize = 32_768;

/// The vocabulary of the released GLM-5.3-Flash checkpoint.
pub const GLM53_VOCAB_SIZE: usize = 154_880;

/// The model ids this build knows how to serve, in selector order.
///
/// Kept as a separate slice so the web control plane can expose it without constructing the
/// registry. A new pack adds its selector here and its descriptor below.
pub const WEIGHT_IDS: &[&str] = &[
    DEFAULT_WEIGHTS_ID,
    QWEN38_FLASH_NEXT_WEIGHTS_ID,
    QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID,
    GLM53_FLASH_WEIGHTS_ID,
];

/// The maximum context window of the released model.
pub const V41_CONTEXT_TOKENS: usize = 1_048_576;

/// The protocol upper bound on one request's generated tokens for the released model.
pub const V41_MAX_OUTPUT_TOKENS: usize = 393_216;

/// The vocabulary size of the released checkpoint.
pub const V41_VOCAB_SIZE: usize = 129_280;

/// What one model is and what it can serve.
///
/// This is data, not a trait: a model pack can expose a different descriptor without changing
/// the server. The descriptor deliberately does not carry weights or paths; those belong to the
/// deployment configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    /// The served model id, e.g. `deepseek-ai/DeepSeek-V4.1-Flash`.
    pub id: String,
    /// The model family, e.g. `deepseek-v4.1`. Used for artifact compatibility checks.
    pub family: String,
    /// The released `config.json` `model_type`, e.g. `deepseek_v41`. The control plane matches a
    /// checkpoint document on this value, so the two spellings of a family cannot drift.
    pub model_type: String,
    /// The short family label a control surface shows beside the model.
    pub label: String,
    /// The control-plane weight selector, e.g. `deepseek-v4.1-flash`.
    pub weights_id: String,
    /// The largest context this model can serve.
    pub context_tokens: usize,
    /// The protocol upper bound on one request's generated tokens.
    pub max_output_tokens: usize,
    /// The vocabulary the model scores against.
    pub vocab_size: usize,
    /// What the model family can do when fully loaded. An engine may narrow this (for example,
    /// a reference build without the vision tower), so the model card still asks the engine.
    pub capabilities: Capabilities,
}

impl ModelDescriptor {
    /// The released DeepSeek-V4.1-Flash descriptor.
    #[must_use]
    pub fn v41_flash() -> Self {
        Self {
            id: DEFAULT_MODEL_ID.to_owned(),
            family: "deepseek-v4.1".to_owned(),
            model_type: "deepseek_v41".to_owned(),
            label: "DeepSeek".to_owned(),
            weights_id: DEFAULT_WEIGHTS_ID.to_owned(),
            context_tokens: V41_CONTEXT_TOKENS,
            max_output_tokens: V41_MAX_OUTPUT_TOKENS,
            vocab_size: V41_VOCAB_SIZE,
            capabilities: Capabilities::ALL_WITH_VISION,
        }
    }

    /// The released nvidia/Qwen3.8-Flash-Next-NVFP4 descriptor.
    ///
    /// The device arm serves the text language model at TP=1; the vision tower and MTP are in the
    /// checkpoint but are not loaded, so vision is not declared.
    #[must_use]
    pub fn qwen38_flash_next_nvfp4() -> Self {
        Self {
            id: QWEN38_FLASH_NEXT_ID.to_owned(),
            family: "qwen4-exp".to_owned(),
            model_type: "qwen4_exp".to_owned(),
            label: "Qwen".to_owned(),
            weights_id: QWEN38_FLASH_NEXT_WEIGHTS_ID.to_owned(),
            context_tokens: QWEN38_CONTEXT_TOKENS,
            max_output_tokens: QWEN38_MAX_OUTPUT_TOKENS,
            vocab_size: QWEN38_VOCAB_SIZE,
            capabilities: Capabilities::ALL,
        }
    }

    /// The official Qwen3.8-Flash-Next FP8 descriptor.
    ///
    /// The architecture is the NVFP4 arm's; only the routed experts' storage differs, so the
    /// geometry, window and vocabulary are the same and only the identity changes.
    #[must_use]
    pub fn qwen38_flash_next_fp8() -> Self {
        Self {
            id: QWEN38_FLASH_NEXT_FP8_ID.to_owned(),
            weights_id: QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID.to_owned(),
            ..Self::qwen38_flash_next_nvfp4()
        }
    }

    /// The released zai-org/GLM-5.3-Flash descriptor.
    ///
    /// The vision tower is in the checkpoint but is not loaded, so vision is not declared. The
    /// context is the model's window; an endpoint reports the context it actually allocated in its
    /// own descriptor.
    #[must_use]
    pub fn glm53_flash() -> Self {
        Self {
            id: GLM53_FLASH_ID.to_owned(),
            family: "glm5-next".to_owned(),
            model_type: "glm5_next".to_owned(),
            label: "GLM".to_owned(),
            weights_id: GLM53_FLASH_WEIGHTS_ID.to_owned(),
            context_tokens: GLM53_CONTEXT_TOKENS,
            max_output_tokens: GLM53_MAX_OUTPUT_TOKENS,
            vocab_size: GLM53_VOCAB_SIZE,
            capabilities: Capabilities::ALL,
        }
    }

    /// A descriptor for any checkpoint whose shared architecture parsed, under the id it is
    /// served as.
    ///
    /// This is the dynamic path's constructor: every identity field the released presets hard-code
    /// (family, model type, window, vocabulary) comes from the checkpoint's own document, so a
    /// fine-tune or a previously unknown hybrid advertises what it actually is on `/v1/models`
    /// without an entry in this registry. `context` is the window the run allocated, capped by the
    /// checkpoint's own `max_position_embeddings`; zero means the document declares no window
    /// and the allocated context stands.
    ///
    /// The registry (`MODELS`) stays the list of canonical presets, not a closed gate: a descriptor
    /// built here carries the canonical `weights_id` when the id names one, and otherwise the id
    /// itself, which is what keeps a novel model honestly `unregistered` in the numerics registry
    /// (D67).
    #[must_use]
    pub fn from_architecture(
        id: impl Into<String>,
        architecture: &crate::Architecture,
        context: usize,
        capabilities: Capabilities,
    ) -> Self {
        let id = id.into();
        let canonical = find_model(&id);
        let declared = architecture.max_position_embeddings;
        let context_tokens = match (context, declared) {
            (0, 0) => 0,
            (0, window) => window,
            (context, 0) => context,
            (context, window) => context.min(window),
        };
        let family = architecture.family.clone();
        // A registry preset's family spelling is the control plane's; a novel architecture's is
        // its own `model_type`. Both are data from the checkpoint, never a match on this list.
        Self {
            id: id.clone(),
            family: canonical.map_or_else(|| family.clone(), |model| model.family.clone()),
            model_type: canonical.map_or_else(|| family.clone(), |model| model.model_type.clone()),
            label: canonical.map_or_else(
                || {
                    id.rsplit('/')
                        .next()
                        .unwrap_or(&id)
                        .split(['-', '_'])
                        .next()
                        .unwrap_or(&id)
                        .to_owned()
                },
                |model| model.label.clone(),
            ),
            weights_id: canonical.map_or_else(|| id.clone(), |model| model.weights_id.clone()),
            context_tokens,
            max_output_tokens: canonical
                .map_or(V41_MAX_OUTPUT_TOKENS, |model| model.max_output_tokens),
            vocab_size: architecture.vocab_size,
            capabilities,
        }
    }

    /// The V4.1-Flash shape under a caller-supplied id.
    ///
    /// This is what a weight-free test or a synthetic server uses when it names a model that is
    /// not necessarily the released checkpoint. It keeps the shape but makes the advertised id
    /// match the engine, so the HTTP surface cannot answer with two different ids.
    #[must_use]
    pub fn for_id(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::v41_flash()
        }
    }

    /// Whether this descriptor is the released V4.1-Flash model.
    #[must_use]
    pub fn is_v41_flash(&self) -> bool {
        self.weights_id == DEFAULT_WEIGHTS_ID
    }
}

/// Every model descriptor compiled into this build.
///
/// The registry is a `LazyLock` rather than a `const` slice because descriptors carry owned
/// strings and may later be built from a pack's registration function. Lookup is cheap and only
/// happens at startup or on a control-plane request.
pub static MODELS: LazyLock<Vec<ModelDescriptor>> = LazyLock::new(|| {
    vec![
        ModelDescriptor::v41_flash(),
        ModelDescriptor::qwen38_flash_next_nvfp4(),
        ModelDescriptor::qwen38_flash_next_fp8(),
        ModelDescriptor::glm53_flash(),
    ]
});

/// The default descriptor: the released V4.1-Flash.
///
/// # Panics
/// Panics only if the registry is empty, which is a build-time error in this crate.
#[must_use]
pub fn default_model() -> &'static ModelDescriptor {
    MODELS.first().expect("the model registry is not empty")
}

/// The canonical family spelling of a checkpoint's `model_type`.
///
/// A text-only release of a multimodal family names itself `<family>_text` at the document root
/// while the arm and registry descriptor are the multimodal family. Normalize that one alias
/// here so every caller (the registry lookup, pack detection, the boot plan check and the typed
/// Qwen bridge) agrees about which family the checkpoint is.
#[must_use]
pub fn canonical_model_type(model_type: &str) -> &str {
    let model_type = model_type.trim();
    if model_type.eq_ignore_ascii_case("qwen4_exp_text") {
        "qwen4_exp"
    } else {
        model_type
    }
}

/// Look up a descriptor by served id, weight selector, family, `model_type`, or the leaf of a
/// Hugging Face id.
///
/// Ids are matched case-insensitively, mirroring the HTTP router's `model_matches`.
#[must_use]
pub fn find_model(query: &str) -> Option<&'static ModelDescriptor> {
    let query = query.trim();
    if query.is_empty() {
        return None;
    }
    let canonical = canonical_model_type(query);
    let query_leaf = query.rsplit('/').next().unwrap_or(query);
    MODELS.iter().find(|model| {
        query.eq_ignore_ascii_case(&model.id)
            || query.eq_ignore_ascii_case(&model.weights_id)
            || query.eq_ignore_ascii_case(&model.family)
            || canonical.eq_ignore_ascii_case(&model.model_type)
            || query_leaf.eq_ignore_ascii_case(model.id.rsplit('/').next().unwrap_or(&model.id))
    })
}

/// Look up a descriptor by the checkpoint's `model_type`, with the text-only alias folded.
#[must_use]
pub fn find_model_by_type(model_type: &str) -> Option<&'static ModelDescriptor> {
    let canonical = canonical_model_type(model_type);
    MODELS
        .iter()
        .find(|model| canonical.eq_ignore_ascii_case(&model.model_type))
}

/// Look up a descriptor by the control plane's weight selector.
#[must_use]
pub fn find_model_by_weights(weights: &str) -> Option<&'static ModelDescriptor> {
    MODELS
        .iter()
        .find(|model| model.weights_id.eq_ignore_ascii_case(weights.trim()))
}

/// The context window of the model selected by `weights`, or `None` if unknown.
#[must_use]
pub fn model_context_tokens(weights: &str) -> Option<usize> {
    find_model_by_weights(weights).map(|model| model.context_tokens)
}

/// The output-token ceiling of the model selected by `weights`, or `None` if unknown.
#[must_use]
pub fn model_max_output_tokens(weights: &str) -> Option<usize> {
    find_model_by_weights(weights).map(|model| model.max_output_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_the_released_model_and_unique_selectors() {
        let models = &*MODELS;
        assert!(!models.is_empty());
        let mut ids: Vec<&str> = models
            .iter()
            .map(|model| model.weights_id.as_str())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), models.len(), "weight selectors must be unique");
        assert_eq!(default_model().weights_id, DEFAULT_WEIGHTS_ID);
        assert_eq!(default_model().id, DEFAULT_MODEL_ID);
        assert_eq!(
            WEIGHT_IDS.len(),
            models.len(),
            "WEIGHT_IDS and MODELS must describe the same registry"
        );
        for model in models {
            assert!(
                WEIGHT_IDS.contains(&model.weights_id.as_str()),
                "{} is missing from WEIGHT_IDS",
                model.weights_id
            );
        }
    }

    #[test]
    fn lookup_accepts_the_served_id_weight_selector_family_and_leaf() {
        for query in [
            DEFAULT_MODEL_ID,
            "deepseek-ai/deepseek-v4.1-flash",
            "DeepSeek-V4.1-Flash",
            DEFAULT_WEIGHTS_ID,
            "deepseek-v4.1",
        ] {
            let found = find_model(query).unwrap_or_else(|| panic!("{query} should resolve"));
            assert_eq!(found.weights_id, DEFAULT_WEIGHTS_ID);
        }
        assert!(find_model("llama-3.2-1b").is_none());
        assert!(find_model("   ").is_none());
    }

    #[test]
    fn the_text_only_qwen_model_type_resolves_to_its_multimodal_family() {
        assert_eq!(canonical_model_type("qwen4_exp_text"), "qwen4_exp");
        assert_eq!(canonical_model_type("QWEN4_EXP_TEXT"), "qwen4_exp");
        assert_eq!(canonical_model_type("glm5_next"), "glm5_next");
        for query in ["qwen4_exp", "qwen4_exp_text", "QWEN4_EXP_TEXT"] {
            let found = find_model(query).unwrap_or_else(|| panic!("{query} should resolve"));
            assert_eq!(found.weights_id, QWEN38_FLASH_NEXT_WEIGHTS_ID);
        }
        assert_eq!(
            find_model_by_type("qwen4_exp_text").map(|model| model.weights_id.as_str()),
            Some(QWEN38_FLASH_NEXT_WEIGHTS_ID)
        );
    }

    #[test]
    fn from_architecture_reads_the_checkpoint_instead_of_the_registry() {
        let architecture = crate::Architecture::from_json(
            r#"{"model_type":"acme_custom_flash","hidden_size":2560,"intermediate_size":16,"vocab_size":256,"max_position_embeddings":128,"num_hidden_layers":1,"num_attention_heads":1,"head_dim":16,"layer_types":["full_attention"]}"#,
        )
        .expect("a custom architecture");
        let descriptor = ModelDescriptor::from_architecture(
            "acme/Custom-Flash-Test",
            &architecture,
            64,
            Capabilities::ALL,
        );
        assert_eq!(descriptor.id, "acme/Custom-Flash-Test");
        assert_eq!(descriptor.family, "acme_custom_flash");
        assert_eq!(descriptor.model_type, "acme_custom_flash");
        assert_eq!(descriptor.weights_id, "acme/Custom-Flash-Test");
        assert_eq!(descriptor.context_tokens, 64);
        assert_eq!(descriptor.vocab_size, 256);
        // A novel id is not in the canonical registry, which is what keeps the numerics key
        // unregistered (D67) rather than borrowing a released model's claim.
        assert!(find_model(&descriptor.weights_id).is_none());
    }

    #[test]
    fn from_architecture_keeps_a_canonical_presets_identity_and_caps_its_window() {
        let architecture = crate::Architecture::from_json(
            r#"{"model_type":"qwen4_exp","hidden_size":2560,"intermediate_size":16,"vocab_size":248320,"max_position_embeddings":262144,"num_hidden_layers":1,"num_attention_heads":1,"head_dim":256,"layer_types":["full_attention"]}"#,
        )
        .expect("a canonical-shaped architecture");
        let descriptor = ModelDescriptor::from_architecture(
            QWEN38_FLASH_NEXT_FP8_ID,
            &architecture,
            1_000_000,
            Capabilities::ALL,
        );
        assert_eq!(descriptor.weights_id, QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID);
        assert_eq!(descriptor.family, "qwen4-exp");
        assert_eq!(descriptor.model_type, "qwen4_exp");
        assert_eq!(descriptor.context_tokens, QWEN38_CONTEXT_TOKENS);
        assert_eq!(descriptor.vocab_size, QWEN38_VOCAB_SIZE);
        assert!(find_model_by_weights(&descriptor.weights_id).is_some());
    }

    #[test]
    fn for_id_keeps_the_shape_and_changes_the_advertised_id() {
        let model = ModelDescriptor::for_id("test-model");
        assert_eq!(model.id, "test-model");
        assert_eq!(model.context_tokens, V41_CONTEXT_TOKENS);
        assert_eq!(model.vocab_size, V41_VOCAB_SIZE);
    }
}
