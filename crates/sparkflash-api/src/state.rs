//! Shared axum application state.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::encoding::TokenizerSeam;
use crate::engine::InferenceEngine;
use crate::metrics::Metrics;

/// Everything the handlers share.
///
/// The engine is a trait object so the router is decoupled from the model runtime, and
/// the response-id counter plus the optional fixed clock make responses byte-for-byte
/// reproducible in tests.
#[derive(Clone)]
pub struct AppState {
    /// Model runtime.
    pub engine: Arc<dyn InferenceEngine>,
    /// Prometheus exporter.
    pub metrics: Arc<Metrics>,
    /// Process start time.
    pub started: Instant,
    id_seq: Arc<AtomicU64>,
    fixed_time_unix: Option<u64>,
    max_logprobs: usize,
    tokenizer: Option<Arc<TokenizerSeam>>,
}

impl AppState {
    /// Builds state around an engine with a fresh metric registry.
    #[must_use]
    pub fn new(engine: Arc<dyn InferenceEngine>) -> Self {
        Self {
            engine,
            metrics: Arc::new(Metrics::new()),
            started: Instant::now(),
            id_seq: Arc::new(AtomicU64::new(0)),
            fixed_time_unix: None,
            max_logprobs: crate::server::DEFAULT_MAX_LOGPROBS,
            tokenizer: None,
        }
    }

    /// Attaches the checkpoint tokenizer and V4.1 chat template.
    ///
    /// A production `sparkflash serve` always attaches one, so /v1/completions prompt text
    /// becomes the checkpoint's token ids and /v1/chat/completions renders the released
    /// server's template. Leaving it off keeps the Milestone 1 placeholder path that the
    /// weight-free router tests pin (`prompt.rs`).
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: Arc<TokenizerSeam>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// The attached tokenizer seam, if any.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&Arc<TokenizerSeam>> {
        self.tokenizer.as_ref()
    }

    /// Pins the `created`/`created_at` timestamp so tests can assert exact bytes.
    #[must_use]
    pub fn with_fixed_time(mut self, unix_seconds: u64) -> Self {
        self.fixed_time_unix = Some(unix_seconds);
        self
    }

    /// Sets the cap on requested logprobs, vLLM's `--max-logprobs`.
    ///
    /// [`usize::MAX`] means "no cap", which is what `sparkflash serve --max-logprobs -1` passes:
    /// vLLM spells unlimited that way too.
    #[must_use]
    pub fn with_max_logprobs(mut self, max_logprobs: usize) -> Self {
        self.max_logprobs = max_logprobs;
        self
    }

    /// The cap on requested logprobs.
    #[must_use]
    pub fn max_logprobs(&self) -> usize {
        self.max_logprobs
    }

    /// The engine's model id.
    #[must_use]
    pub fn model_id(&self) -> &str {
        self.engine.model_id()
    }

    /// Next deterministic response id for the given prefix.
    #[must_use]
    pub fn next_id(&self, prefix: &str) -> String {
        format!(
            "{prefix}{:08x}",
            self.id_seq.fetch_add(1, Ordering::Relaxed)
        )
    }

    /// Current Unix timestamp, or the pinned one in tests.
    #[must_use]
    pub fn now_unix(&self) -> u64 {
        self.fixed_time_unix.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |delta| delta.as_secs())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::MockEngine;

    #[test]
    fn ids_are_monotonic_and_deterministic() {
        let state = AppState::new(Arc::new(MockEngine::new("m")));
        assert_eq!(state.next_id("chatcmpl-dsv41-"), "chatcmpl-dsv41-00000000");
        assert_eq!(state.next_id("chatcmpl-dsv41-"), "chatcmpl-dsv41-00000001");
    }

    #[test]
    fn fixed_time_overrides_the_wall_clock() {
        let state = AppState::new(Arc::new(MockEngine::new("m"))).with_fixed_time(1_700_000_000);
        assert_eq!(state.now_unix(), 1_700_000_000);
    }
}
