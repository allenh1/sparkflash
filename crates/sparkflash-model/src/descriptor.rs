//! Model descriptors and the compile-time registry.

use std::sync::LazyLock;

use crate::capabilities::Capabilities;

/// The canonical id of the released V4.1-Flash checkpoint.
pub const DEFAULT_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

/// The control-plane weight selector for the released V4.1-Flash checkpoint.
pub const DEFAULT_WEIGHTS_ID: &str = "deepseek-v4.1-flash";

/// The model ids this build knows how to serve, in selector order.
///
/// Kept as a separate slice so the web control plane can expose it without constructing the
/// registry. A new pack adds its selector here and its descriptor below.
pub const WEIGHT_IDS: &[&str] = &[DEFAULT_WEIGHTS_ID];

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
            weights_id: DEFAULT_WEIGHTS_ID.to_owned(),
            context_tokens: V41_CONTEXT_TOKENS,
            max_output_tokens: V41_MAX_OUTPUT_TOKENS,
            vocab_size: V41_VOCAB_SIZE,
            capabilities: Capabilities::ALL_WITH_VISION,
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
pub static MODELS: LazyLock<Vec<ModelDescriptor>> =
    LazyLock::new(|| vec![ModelDescriptor::v41_flash()]);

/// The default descriptor: the released V4.1-Flash.
///
/// # Panics
/// Panics only if the registry is empty, which is a build-time error in this crate.
#[must_use]
pub fn default_model() -> &'static ModelDescriptor {
    MODELS.first().expect("the model registry is not empty")
}

/// Look up a descriptor by served id, weight selector, family, or the leaf of a Hugging Face id.
///
/// Ids are matched case-insensitively, mirroring the HTTP router's `model_matches`.
#[must_use]
pub fn find_model(query: &str) -> Option<&'static ModelDescriptor> {
    let query = query.trim();
    if query.is_empty() {
        return None;
    }
    let query_leaf = query.rsplit('/').next().unwrap_or(query);
    MODELS.iter().find(|model| {
        query.eq_ignore_ascii_case(&model.id)
            || query.eq_ignore_ascii_case(&model.weights_id)
            || query.eq_ignore_ascii_case(&model.family)
            || query_leaf.eq_ignore_ascii_case(model.id.rsplit('/').next().unwrap_or(&model.id))
    })
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
    fn for_id_keeps_the_shape_and_changes_the_advertised_id() {
        let model = ModelDescriptor::for_id("test-model");
        assert_eq!(model.id, "test-model");
        assert_eq!(model.context_tokens, V41_CONTEXT_TOKENS);
        assert_eq!(model.vocab_size, V41_VOCAB_SIZE);
    }
}
