//! The backend contract: how a model definition runs on one host.
//!
//! A model pack owns the architecture and the weights. A backend owns the state and the
//! arithmetic that turns a prompt into logits. This module names the seam between them so the
//! serving layer can drive a model without knowing whether the execution is the CPU reference,
//! a CUDA session, or a test double.
//!
//! The contract is deliberately one step at a time. Prefill consumes a prompt and returns the
//! last position's logits plus the token the backend chose; decode consumes one token and returns
//! the next step. A serving adapter dispatches at the step boundary, which is once per token, not
//! once per tensor or kernel.

use crate::capabilities::Capabilities;
use crate::descriptor::ModelDescriptor;

/// A backend failure.
///
/// The message is written for an operator; the code is the stable category a serving layer can
/// expose. Backend implementations decide which category their failure belongs to.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct BackendError {
    /// The stable machine-readable category.
    code: &'static str,
    /// The operator-facing explanation.
    message: String,
}

impl BackendError {
    /// A failure with a caller-supplied category.
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// The stable machine-readable category.
    #[must_use]
    pub fn code(&self) -> &'static str {
        self.code
    }

    /// The operator-facing explanation.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// What one prefill or decode step produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepOutput {
    /// The logits of the last position. Empty when the backend does not expose them.
    pub logits: Vec<f32>,
    /// The token the backend selected for the next position, or None when generation is done.
    pub next_token: Option<u32>,
    /// Draft tokens after the anchor, for a backend with speculative decoding. Empty when the
    /// backend drafts nothing; a generational adapter may use it as evidence, but must not treat
    /// a draft as an emitted token.
    pub draft: Vec<u32>,
}

/// The sampling configuration for one request.
///
/// This mirrors the protocol's sampling fields so a model backend can apply them with the
/// arithmetic and random source it owns. The serving layer sets it on a fresh session before
/// prefill; a backend that has no sampling stage may ignore it and return a deterministic token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// 0.0 is greedy; above it the backend's seeded distribution.
    pub temperature: f32,
    /// Nucleus threshold; 1.0 disables it.
    pub top_p: f32,
    /// Keep this many most likely tokens; 0 and -1 disable it.
    pub top_k: i32,
    /// Repetition penalty over the prompt and the generated tokens; 1.0 disables it.
    pub repetition_penalty: f32,
    /// Presence penalty over generated tokens; 0.0 disables it.
    pub presence_penalty: f32,
    /// Frequency penalty over generated tokens; 0.0 disables it.
    pub frequency_penalty: f32,
    /// The deterministic seed when the request named one.
    pub seed: u64,
    /// How many of the consumed tokens are the prompt.
    pub prompt_len: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            prompt_len: 0,
        }
    }
}

/// One sequence's backend state.
///
/// A session is created per request so two requests never share KV state. It is Send because
/// the serving adapter owns it on its engine thread.
pub trait ModelSession: Send {
    /// Applies this request's sampler before any step.
    ///
    /// The default keeps the backend's own deterministic choice, which is the right behavior for
    /// a test double and for a backend that has not implemented sampling yet.
    fn set_sampling(&mut self, sampling: Sampling) {
        let _ = sampling;
    }

    /// Consumes a prompt and returns the first step.
    ///
    /// # Errors
    /// A backend failure, such as a shape mismatch or an exhausted context.
    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError>;

    /// Consumes one generated token and returns the next step.
    ///
    /// # Errors
    /// A backend failure, such as a shape mismatch or an exhausted context.
    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError>;
}

/// How a model runs on one host.
///
/// A backend is a factory for sessions, not a session itself: it can be shared, while each
/// request gets its own state. The descriptor is the identity the serving layer reports; the
/// capabilities are what this backend can actually execute, which may be narrower than what the
/// model family supports when loaded fully.
pub trait ModelBackend: Send + Sync {
    /// The model this backend executes.
    fn descriptor(&self) -> &ModelDescriptor;

    /// What this backend can serve.
    ///
    /// The default takes the descriptor's capabilities. A backend that does not load an optional
    /// tower narrows the answer rather than advertising a channel it cannot run.
    fn capabilities(&self) -> Capabilities {
        self.descriptor().capabilities
    }

    /// Opens a fresh sequence.
    ///
    /// # Errors
    /// A backend failure, such as unavailable weights or an unsupported configuration.
    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_error_keeps_its_code_and_message() {
        let error = BackendError::new("weights", "the shard is missing");
        assert_eq!(error.code(), "weights");
        assert_eq!(error.message(), "the shard is missing");
        assert_eq!(error.to_string(), "the shard is missing");
    }

    #[test]
    fn default_sampling_is_the_disabled_configuration() {
        let sampling = Sampling::default();
        assert_eq!(sampling.temperature, 0.0);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.top_k, 0);
        assert_eq!(sampling.repetition_penalty, 1.0);
        assert_eq!(sampling.presence_penalty, 0.0);
        assert_eq!(sampling.frequency_penalty, 0.0);
    }
}
