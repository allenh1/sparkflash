//! Shared axum application state.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::device_token::DeviceTokenSource;
use crate::engine::InferenceEngine;
use crate::metrics::Metrics;
use sparkflash_model::{ModelDescriptor, PromptCodec};

/// The default interval between SSE keep-alive comment frames while an engine stream is idle.
///
/// Fifteen seconds is the conventional proxy-idle cadence for a streaming connection; a comment
/// frame keeps the connection open without adding a JSON event or changing the payload.
pub const DEFAULT_SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);

/// Draws the per-choice sampling seeds for a request that named no seed.
///
/// vLLM's contract for an unseeded request is "sample independently, no reproducibility"; the
/// OpenAI API only promises determinism when the caller passes `seed`. The router therefore has
/// to create the randomness itself for unseeded requests: with n > 1 every choice must be a
/// different draw, not n copies of seed 0 (which is what the reference driver would otherwise
/// generate). A source is a trait so tests can pin the exact draws while production uses
/// process entropy.
pub trait ChoiceSeedSource: Send + Sync {
    /// Returns one fresh seed. Successive calls are expected to differ.
    fn next_seed(&self) -> u64;
}

/// The production seed source: OS-seeded entropy, mixed with a monotone draw counter.
///
/// `std::collections::hash_map::RandomState` is seeded from the OS per process and per instance;
/// hashing the draw counter with those random keys yields a u64 that is neither predictable from
/// the process start time nor repeated across draws. No entropy crate is added for this.
#[derive(Debug, Default)]
pub struct RandomChoiceSeeds;

impl ChoiceSeedSource for RandomChoiceSeeds {
    fn next_seed(&self) -> u64 {
        static DRAW: AtomicU64 = AtomicU64::new(0);
        let random = RandomState::new();
        let mut hasher = random.build_hasher();
        hasher.write_u64(DRAW.fetch_add(1, Ordering::Relaxed));
        hasher.write_u128(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |delta| delta.as_nanos()),
        );
        hasher.finish()
    }
}

/// The browser cross-origin policy the router installs.
///
/// vLLM's server mounts a permissive CORS middleware by default (`access-control-allow-origin:
/// *`, every method, `max-age=600`), so a browser client works against the oracle out of the
/// box. SparkFlash matches that default and keeps an operator override: a listed origin set
/// answers only those origins with an echoed `Access-Control-Allow-Origin` and a `Vary: Origin`,
/// and `None` on [`AppState::with_cors`] removes the layer entirely (the pre-CORS wire bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorsConfig {
    /// `None` allows any origin (vLLM's default, `*`); `Some(list)` allows only those origins.
    allowed_origins: Option<Vec<String>>,
    /// How long a browser may cache one preflight result, in seconds.
    max_age_secs: u64,
}

impl CorsConfig {
    /// vLLM's default: any origin, `max-age=600`.
    #[must_use]
    pub fn any() -> Self {
        Self {
            allowed_origins: None,
            max_age_secs: 600,
        }
    }

    /// Only these origins are allowed; every other origin gets no CORS headers.
    #[must_use]
    pub fn allow_list(origins: Vec<String>) -> Self {
        Self {
            allowed_origins: Some(origins),
            max_age_secs: 600,
        }
    }

    /// Overrides the preflight cache lifetime.
    #[must_use]
    pub fn with_max_age_secs(mut self, seconds: u64) -> Self {
        self.max_age_secs = seconds;
        self
    }

    /// The allowed origin list; `None` means any origin.
    #[must_use]
    pub fn allowed_origins(&self) -> Option<&[String]> {
        self.allowed_origins.as_deref()
    }

    /// The preflight cache lifetime, in seconds.
    #[must_use]
    pub fn max_age_secs(&self) -> u64 {
        self.max_age_secs
    }
}

/// A deterministic seed source for tests: draws `start, start + 1, start + 2, ...`.
#[derive(Debug)]
pub struct SequentialChoiceSeeds {
    next: AtomicU64,
}

impl SequentialChoiceSeeds {
    /// A source whose first draw is `start`.
    #[must_use]
    pub fn starting_at(start: u64) -> Self {
        Self {
            next: AtomicU64::new(start),
        }
    }
}

impl ChoiceSeedSource for SequentialChoiceSeeds {
    fn next_seed(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }
}

/// Everything the handlers share.
///
/// The engine is a trait object so the router is decoupled from the model runtime, and
/// the response-id counter plus the optional fixed clock make responses byte-for-byte
/// reproducible in tests.
#[derive(Clone)]
pub struct AppState {
    /// Model runtime.
    pub engine: Arc<dyn InferenceEngine>,
    /// What this process is serving: id, context, output ceiling, vocabulary and capabilities.
    ///
    /// Defaults to the V4.1 descriptor under the engine's own id, so weight-free tests remain
    /// valid; a real serve installs the selected registry entry.
    model: Arc<ModelDescriptor>,
    /// Prometheus exporter.
    pub metrics: Arc<Metrics>,
    /// Process start time.
    pub started: Instant,
    id_seq: Arc<AtomicU64>,
    fixed_time_unix: Option<u64>,
    max_logprobs: usize,
    sse_keep_alive: Option<Duration>,
    tokenizer: Option<Arc<dyn PromptCodec>>,
    api_key: Option<Arc<str>>,
    cors: Option<CorsConfig>,
    device_token_source: Option<Arc<dyn DeviceTokenSource>>,
    choice_seeds: Arc<dyn ChoiceSeedSource>,
    draining: Arc<AtomicBool>,
}

impl AppState {
    /// Builds state around an engine with a fresh metric registry.
    #[must_use]
    pub fn new(engine: Arc<dyn InferenceEngine>) -> Self {
        let model = Arc::new(ModelDescriptor::for_id(engine.model_id()));
        Self {
            engine,
            model,
            metrics: Arc::new(Metrics::new()),
            started: Instant::now(),
            id_seq: Arc::new(AtomicU64::new(0)),
            fixed_time_unix: None,
            max_logprobs: crate::server::DEFAULT_MAX_LOGPROBS,
            sse_keep_alive: Some(DEFAULT_SSE_KEEP_ALIVE),
            tokenizer: None,
            api_key: None,
            cors: Some(CorsConfig::any()),
            device_token_source: None,
            choice_seeds: Arc::new(RandomChoiceSeeds),
            draining: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Installs the selected model descriptor.
    ///
    /// The descriptor is what the HTTP surface reports for context, output ceiling, vocabulary
    /// and the model card. The engine still owns what it can actually execute, so capabilities
    /// remain an engine answer.
    #[must_use]
    pub fn with_model_descriptor(mut self, descriptor: Arc<ModelDescriptor>) -> Self {
        self.model = descriptor;
        self
    }

    /// The selected model descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &ModelDescriptor {
        &self.model
    }

    /// Replaces the exporter so the engine driver and the HTTP surface share one registry.
    ///
    /// `sparkflash serve` builds the exporter before the engine driver, because the driver's
    /// request lifecycle is what feeds the scheduler gauges, the generator counter and the
    /// time-to-first-token histogram. Without this the router would export a second, empty
    /// registry and every one of those families would read zero.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Attaches the checkpoint tokenizer and V4.1 chat template.
    ///
    /// A production `sparkflash serve` always attaches one, so /v1/completions prompt text
    /// becomes the checkpoint's token ids and /v1/chat/completions renders the released
    /// server's template. Leaving it off keeps the Milestone 1 placeholder path that the
    /// weight-free router tests pin (`prompt.rs`).
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn PromptCodec>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// The attached tokenizer seam, if any.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&Arc<dyn PromptCodec>> {
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

    /// Sets the idle SSE keep-alive interval, or turns the comment frames off with None.
    ///
    /// The engine stream is the only source of SSE events, so a long engine pause is a long
    /// client-visible pause; the keep-alive comment fills it without adding a JSON event.
    #[must_use]
    pub fn with_sse_keep_alive(mut self, interval: Option<Duration>) -> Self {
        self.sse_keep_alive = interval;
        self
    }

    /// The idle SSE keep-alive interval, if comment frames are enabled.
    #[must_use]
    pub fn sse_keep_alive(&self) -> Option<Duration> {
        self.sse_keep_alive
    }

    /// Requires a bearer API key on every route except `/health` and `/metrics`.
    ///
    /// `None` is the default and preserves the pre-task-20 behaviour: the server is open. A key
    /// turns on the middleware in [`crate::server::router`], which answers a missing or wrong
    /// `Authorization: Bearer <key>` with the OpenAI 401 envelope.
    #[must_use]
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key.map(Arc::from);
        self
    }

    /// The bearer API key, when one is configured.
    #[must_use]
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// Sets the browser cross-origin policy, or removes the CORS layer with `None`.
    ///
    /// The default is [`CorsConfig::any`], which is what the reference vLLM server serves; `None`
    /// restores the pre-CORS wire surface (no `access-control-*` headers and no preflight answer).
    #[must_use]
    pub fn with_cors(mut self, cors: Option<CorsConfig>) -> Self {
        self.cors = cors;
        self
    }

    /// The browser cross-origin policy, when the layer is enabled.
    #[must_use]
    pub fn cors(&self) -> Option<&CorsConfig> {
        self.cors.as_ref()
    }

    /// Installs the device-chain token source for the S7/S8 serving scaffold.
    ///
    /// The default is `None`: the router then serves every request from the host twin engine,
    /// exactly as before the scaffold. When a source is installed, a plain (non-streaming or
    /// streaming) completion request may be served from a claimed four-rank device token envelope;
    /// an unclaimed envelope, a missing directory or an advanced request falls back to the engine.
    #[must_use]
    pub fn with_device_token_source(mut self, source: Option<Arc<dyn DeviceTokenSource>>) -> Self {
        self.device_token_source = source;
        self
    }

    /// The device-chain token source, when one is configured.
    #[must_use]
    pub fn device_token_source(&self) -> Option<&Arc<dyn DeviceTokenSource>> {
        self.device_token_source.as_ref()
    }

    /// Replaces the seed source for unseeded choices (tests use a deterministic source).
    #[must_use]
    pub fn with_choice_seeds(mut self, choice_seeds: Arc<dyn ChoiceSeedSource>) -> Self {
        self.choice_seeds = choice_seeds;
        self
    }

    /// Resolves one choice's sampling seed.
    ///
    /// The caller's seed is vLLM's per-choice derivation (`seed + i`), which keeps a seeded
    /// request reproducible and makes its choices distinct draws. An **unseeded** request leaves
    /// `EngineRequest::seed` as None when it asks for one choice (the pre-existing byte-compatible
    /// path), and draws a fresh seed for every choice when it asks for more than one: that is the
    /// difference between n independent samples and n copies of one deterministic stream.
    #[must_use]
    pub fn choice_seed(&self, seed: Option<u64>, choice: usize, n: usize) -> Option<u64> {
        match seed {
            Some(seed) => Some(seed.wrapping_add(choice as u64)),
            None if n > 1 => Some(self.choice_seeds.next_seed()),
            None => None,
        }
    }

    /// Marks the server as draining: readiness reports not-ready, requests already admitted
    /// continue to completion.
    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::SeqCst);
    }

    /// Whether the drain signal has been received.
    #[must_use]
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    /// The selected model id.
    ///
    /// This is the descriptor's id, which [`Self::new`] synchronises with the engine. A caller
    /// that installs a descriptor explicitly makes that descriptor authoritative.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model.id
    }

    /// The maximum sequence length this process serves.
    ///
    /// SPARK_MAX_SEQ_LEN still overrides the descriptor's context, but only within that model's
    /// own window.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        crate::engine::max_seq_len_for(self.model.context_tokens)
    }

    /// The output-token ceiling this process advertises.
    #[must_use]
    pub fn max_output_tokens(&self) -> usize {
        self.model.max_output_tokens
    }

    /// Next deterministic response id for the given prefix.
    #[must_use]
    pub fn next_id(&self, prefix: &str) -> String {
        format!(
            "{prefix}{:08x}",
            self.id_seq.fetch_add(1, Ordering::Relaxed)
        )
    }

    /// The shared response-id counter.
    ///
    /// A streaming response generator cannot hold an `&AppState` (the body has to be `'static`),
    /// so it takes the counter instead and formats tool-call ids from it the same way [`Self::next_id`]
    /// does.
    #[must_use]
    pub fn id_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.id_seq)
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

    #[test]
    fn the_default_descriptor_takes_its_id_from_the_engine() {
        let state = AppState::new(Arc::new(MockEngine::new("m")));
        assert_eq!(state.model_id(), "m");
        assert_eq!(state.descriptor().id, "m");
        assert_eq!(
            state.descriptor().context_tokens,
            sparkflash_model::V41_CONTEXT_TOKENS
        );
        assert_eq!(
            state.max_output_tokens(),
            sparkflash_model::V41_MAX_OUTPUT_TOKENS
        );
    }

    #[test]
    fn an_installed_descriptor_drives_context_output_and_id() {
        let descriptor = ModelDescriptor {
            id: "tiny-dense".to_owned(),
            family: "dense".to_owned(),
            weights_id: "tiny-dense".to_owned(),
            context_tokens: 4096,
            max_output_tokens: 1024,
            vocab_size: 32_000,
            capabilities: Default::default(),
        };
        let state = AppState::new(Arc::new(MockEngine::new("m")))
            .with_model_descriptor(Arc::new(descriptor));
        assert_eq!(state.model_id(), "tiny-dense");
        assert_eq!(state.max_seq_len(), 4096);
        assert_eq!(state.max_output_tokens(), 1024);
        assert_eq!(state.descriptor().vocab_size, 32_000);
    }
}
