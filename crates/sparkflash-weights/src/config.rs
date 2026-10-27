//! Configuration errors shared by model packs and the generic host arithmetic.

/// What is wrong with a configuration.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// The JSON could not be parsed, or a required key was absent.
    #[error("could not parse the configuration: {0}")]
    Json(String),
    /// A parsed value contradicts another.
    #[error("{field}: {detail}")]
    Invalid {
        /// The field the complaint is about.
        field: &'static str,
        /// What is wrong with it.
        detail: String,
    },
}
