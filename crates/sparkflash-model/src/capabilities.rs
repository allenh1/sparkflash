//! What a model/engine can serve.
//!
//! This type lives with the model contract rather than the HTTP layer because both a model
//! descriptor and an engine report one. The HTTP layer re-exports it, so existing callers keep
//! using `sparkflash_api::engine::Capabilities`.

/// What an engine can serve beyond plain text.
///
/// Every field defaults to `false`: a channel an engine has not declared is treated as absent,
/// so `GET /v1/models` can never advertise a capability by omission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Capabilities {
    /// Whether the engine can emit tool calls.
    pub tools: bool,
    /// Whether the engine produces a reasoning channel.
    pub thinking: bool,
    /// Whether the engine honours a caller's `reasoning_effort`.
    pub reasoning_effort: bool,
    /// Whether the engine turns prepared images into visual tokens.
    pub vision: bool,
}

impl Capabilities {
    /// Every text channel, declared; vision stays opt-in because it depends on the tower being
    /// loaded by the engine.
    pub const ALL: Self = Self {
        tools: true,
        thinking: true,
        reasoning_effort: true,
        vision: false,
    };

    /// The same text channels plus vision.
    pub const ALL_WITH_VISION: Self = Self {
        vision: true,
        ..Self::ALL
    };
}
