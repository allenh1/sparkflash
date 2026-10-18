//! The engine seam: a weights-free boundary between the HTTP protocol layer and the
//! scheduler/model runtime.
//!
//! Everything the router needs from a running model is expressed by InferenceEngine, so
//! the whole HTTP surface is testable against MockEngine without a GPU, a checkpoint or a
//! network. EngineRequest carries exactly the fields the OpenAI/Anthropic front ends can
//! produce in Milestone 1; EngineEvent is the normalized decode stream both the SSE and
//! the buffered response encoders consume.

use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use sparkflash_engram::EngramTelemetrySnapshot;
use sparkflash_kv::{KvCacheStatsSnapshot, KvCacheTelemetry, PrefixMatchPlan};

use crate::loop_abort::{LoopAbort, LoopAbortConfig, LoopAbortDetector};
use crate::multimodal::ImagePart;
use crate::telemetry_types::{
    CompletionTokensDetails, DSparkStepStats, DeepSeekSparkRequestStats, OpenAiUsageWithCache,
    PromptTokensDetails, RequestTimingStats, RoceTransportStats, finite_or_zero,
};

/// Canonical Hugging Face model id served by this cluster.
pub const MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
/// Maximum context window of the model, in tokens.
pub const MODEL_CONTEXT_TOKENS: usize = 1_048_576;

/// Environment variable that overrides the sequence length the server actually serves.
pub const MAX_SEQ_LEN_ENV: &str = "SPARK_MAX_SEQ_LEN";

/// Smallest sequence length the server will serve.
///
/// The floor is one token: zero is not a window at all, while any positive value is
/// technically servable. We refuse only windows the engine provably cannot honour (zero,
/// or one wider than the model's), rather than second-guessing an operator who really
/// wants a deliberately tiny window.
pub const MIN_SEQ_LEN: usize = 1;

/// Parses a SPARK_MAX_SEQ_LEN value, returning None when it must be ignored.
///
/// Accepts a pure decimal integer within MIN_SEQ_LEN..=MODEL_CONTEXT_TOKENS. Surrounding
/// whitespace is tolerated because it survives shell quoting. Everything else is
/// rejected: a sign (usize::from_str would happily take "+131072"), underscore
/// separators, zero, and anything wider than the model's window - advertising
/// 18446744073709551615 tokens would be a number the engine cannot serve.
#[must_use]
pub fn parse_max_seq_len(raw: &str) -> Option<usize> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || !trimmed.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    trimmed
        .parse::<usize>()
        .ok()
        .filter(|value| (MIN_SEQ_LEN..=MODEL_CONTEXT_TOKENS).contains(value))
}

/// The maximum sequence length in use, per DECISIONS.md D5.
///
/// SPARK_MAX_SEQ_LEN wins when it is usable (see [`parse_max_seq_len`]). Anything else -
/// unset, unparsable, zero, or wider than the model's window - falls back to
/// [`MODEL_CONTEXT_TOKENS`] with a warning, so a typo can never make the server advertise
/// a window it cannot honour. /v1/models and /v1/cluster/stats report this value.
#[must_use]
pub fn max_seq_len() -> usize {
    let Ok(raw) = std::env::var(MAX_SEQ_LEN_ENV) else {
        return MODEL_CONTEXT_TOKENS;
    };
    match parse_max_seq_len(&raw) {
        Some(value) => value,
        None => {
            tracing::warn!(
                value = %raw,
                "ignoring {MAX_SEQ_LEN_ENV}: not a usable sequence length, serving {MODEL_CONTEXT_TOKENS}"
            );
            MODEL_CONTEXT_TOKENS
        }
    }
}
/// Default maximum number of generated tokens.
pub const MODEL_MAX_OUTPUT_TOKENS: usize = 65_536;
/// Value reported in the OpenAI `owned_by` field.
pub const MODEL_OWNED_BY: &str = "sparkflash";

/// Why generation stopped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of turn (EOS or a stop sequence).
    #[default]
    Stop,
    /// The token budget from `max_tokens` was exhausted.
    Length,
    /// The model emitted one or more tool calls.
    ToolCalls,
    /// The decode-side loop detector aborted a degenerate repetition.
    LoopAbort,
}

impl FinishReason {
    /// The OpenAI wire spelling of this reason.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::LoopAbort => "loop_abort",
        }
    }
}

/// A fully normalized inference request.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineRequest {
    /// Caller-supplied or router-generated request id.
    pub request_id: String,
    /// Resolved model id.
    pub model: String,
    /// Prompt token ids (placeholder ids in Milestone 1).
    pub prompt_tokens: Vec<u32>,
    /// Image content parts carried by the prompt, in request order.
    ///
    /// The router passes the parsed parts through and [`InferenceEngine::validate_request`] refuses
    /// any request that has one with [`MissingCapability::vision`] before generation: no engine in
    /// this workspace implements the image path, so there is no honest visual-token count to carry
    /// beside them. Inline parts carry the decoded payload bytes, so a vision front end can decode
    /// them to pixels and derive the ids, never from an assumed grid: the future wiring point has
    /// the payload it needs.
    pub images: Vec<ImagePart>,
    /// Hard cap on generated tokens.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Whether the reasoning channel is enabled.
    pub thinking: bool,
    /// Reasoning effort score in `1..=100`, resolved by crate::reasoning.
    pub reasoning_effort: Option<u8>,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Optional deterministic sampling seed.
    pub seed: Option<u64>,
    /// Scope mask armed by the `X-SparkFlash-Trace` request header.
    pub trace_scopes: u64,
    /// Engine steps for which `trace_scopes` stays armed.
    pub trace_steps: u32,
}

/// One element of the normalized decode stream.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// Emitted first, before any delta.
    Started {
        /// Total prompt tokens. Text only: a request carrying an image part is refused before
        /// generation, because no engine in this workspace implements the vision path.
        prompt_tokens: usize,
        /// Prompt tokens served from the prefix cache.
        cached_tokens: usize,
    },
    /// A delta in the reasoning channel.
    ReasoningDelta(String),
    /// A delta in the visible content channel.
    ContentDelta(String),
    /// A delta belonging to a streamed tool call.
    ToolCallDelta {
        /// Zero-based tool-call index.
        index: usize,
        /// Tool-call id; only present on the first delta of a call.
        id: Option<String>,
        /// Function name; only present on the first delta of a call.
        name: Option<String>,
        /// Fragment of the JSON arguments string.
        arguments_delta: String,
    },
    /// Terminal event carrying usage and telemetry.
    Finished {
        /// Why generation stopped.
        finish_reason: FinishReason,
        /// OpenAI usage block.
        usage: OpenAiUsageWithCache,
        /// Extended SparkFlash telemetry for the request.
        stats: Box<DeepSeekSparkRequestStats>,
    },
    /// Terminal error event.
    Error {
        /// Human-readable message.
        message: String,
        /// Machine-readable code.
        code: String,
    },
}

/// Per-node hardware telemetry for `GET /v1/cluster/stats`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeStats {
    /// Rank in the TP group (0..3).
    pub rank: u32,
    /// Host name.
    pub hostname: String,
    /// Total coherent UMA memory in GiB.
    pub uma_total_gib: f64,
    /// Free coherent UMA memory in GiB.
    pub uma_free_gib: f64,
    /// Locked SM clock in MHz.
    pub gpu_clock_mhz: u32,
    /// Installed NVIDIA driver version.
    pub driver_version: String,
    /// Running kernel version.
    pub kernel_version: String,
    /// Negotiated rail 0 link speed in Mb/s.
    pub rail0_speed_mbps: u32,
    /// Negotiated rail 1 link speed in Mb/s.
    pub rail1_speed_mbps: u32,
}

/// Cluster-wide telemetry snapshot returned by `GET /v1/cluster/stats`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterStatsSnapshot {
    /// Model id.
    pub model: String,
    /// Active fabric description.
    pub fabric_mode: String,
    /// Per-rank hardware stats.
    pub nodes: Vec<NodeStats>,
    /// KV cache counters.
    pub kv: KvCacheStatsSnapshot,
    /// Engram layer telemetry (layers 1 and 14).
    pub engram: Vec<EngramTelemetrySnapshot>,
    /// Transport counters.
    pub transport: RoceTransportStats,
    /// DSpark speculative counters.
    pub dspark: DSparkStepStats,
    /// Timing snapshot.
    pub timing: RequestTimingStats,
    /// Maximum sequence length in use: SPARK_MAX_SEQ_LEN, defaulting to
    /// [`MODEL_CONTEXT_TOKENS`] when it is unset or unusable (DECISIONS.md D5).
    pub max_seq_len: usize,
}

/// Zeroed KV cache telemetry snapshot.
#[must_use]
pub fn empty_kv_snapshot() -> KvCacheStatsSnapshot {
    KvCacheStatsSnapshot {
        requests_total: 0,
        prompt_tokens_total: 0,
        global_kv_uma_hit_tokens: 0,
        global_kv_nvme_hit_tokens: 0,
        global_kv_miss_tokens: 0,
        swa_tail_cache_hits: 0,
        swa_tail_cache_misses: 0,
        encoder_swa_replay_tokens_total: 0,
        decoder_swa_replay_tokens_total: 0,
        active_kv_pages: 0,
        total_kv_pages: 0,
        global_kv_hit_rate: 0.0,
    }
}

/// Zeroed telemetry for one Engram layer.
#[must_use]
pub fn default_engram_layer(layer_id: usize) -> EngramTelemetrySnapshot {
    EngramTelemetrySnapshot {
        layer_id,
        lookups_total: 0,
        tier1_drm_hits: 0,
        tier2_uma_hits: 0,
        tier3_nvme_misses: 0,
        hit_rate: 0.0,
        mean_prefetch_stall_us: 0.0,
        mean_gate_alpha: 0.0,
    }
}

/// Zeroed telemetry for the two Engram layers (1 and 14).
#[must_use]
pub fn default_engram_layers() -> [EngramTelemetrySnapshot; 2] {
    [default_engram_layer(1), default_engram_layer(14)]
}

impl Default for ClusterStatsSnapshot {
    fn default() -> Self {
        Self {
            model: String::new(),
            fabric_mode: String::new(),
            nodes: Vec::new(),
            kv: empty_kv_snapshot(),
            engram: default_engram_layers().to_vec(),
            transport: RoceTransportStats::default(),
            dspark: DSparkStepStats::default(),
            timing: RequestTimingStats::default(),
            max_seq_len: max_seq_len(),
        }
    }
}

impl ClusterStatsSnapshot {
    /// A realistic 4x DGX Spark snapshot using the numbers documented in spec section 5.7.
    #[must_use]
    pub fn sample() -> Self {
        let node = |rank: u32, hostname: &str, uma_free_gib: f64| NodeStats {
            rank,
            hostname: hostname.to_owned(),
            uma_total_gib: 128.0,
            uma_free_gib,
            gpu_clock_mhz: 2200,
            driver_version: "580.178.04".to_owned(),
            kernel_version: "6.17.0-1018-nvidia-64k".to_owned(),
            rail0_speed_mbps: 200_000,
            rail1_speed_mbps: 200_000,
        };
        Self {
            model: MODEL_ID.to_owned(),
            fabric_mode: "Switched200G_DualRail_RoCEnante".to_owned(),
            nodes: vec![
                node(0, "spark-0", 9.42),
                node(1, "spark-1", 9.58),
                node(2, "spark-2", 9.31),
                node(3, "spark-3", 9.44),
            ],
            kv: KvCacheStatsSnapshot {
                requests_total: 128,
                prompt_tokens_total: 4_194_304,
                global_kv_uma_hit_tokens: 3_670_016,
                global_kv_nvme_hit_tokens: 262_144,
                global_kv_miss_tokens: 262_144,
                swa_tail_cache_hits: 96,
                swa_tail_cache_misses: 32,
                encoder_swa_replay_tokens_total: 4096,
                decoder_swa_replay_tokens_total: 16_384,
                active_kv_pages: 8192,
                total_kv_pages: 16_384,
                global_kv_hit_rate: 0.9375,
            },
            engram: vec![
                EngramTelemetrySnapshot {
                    layer_id: 1,
                    lookups_total: 26_880,
                    tier1_drm_hits: 7120,
                    tier2_uma_hits: 12_890,
                    tier3_nvme_misses: 6870,
                    hit_rate: 0.7444,
                    mean_prefetch_stall_us: 0.0,
                    mean_gate_alpha: 0.312,
                },
                EngramTelemetrySnapshot {
                    layer_id: 14,
                    lookups_total: 26_880,
                    tier1_drm_hits: 6840,
                    tier2_uma_hits: 12_410,
                    tier3_nvme_misses: 7630,
                    hit_rate: 0.7161,
                    mean_prefetch_stall_us: 0.0,
                    mean_gate_alpha: 0.289,
                },
            ],
            transport: RoceTransportStats {
                fabric_mode: "Switched200G_DualRail_RoCEnante".to_owned(),
                rocenante_oneshot_calls: 5408,
                nccl_ring_calls: 20,
                rail0_rocep1s0f0_mb: 412.6,
                rail1_rocep2p1s0f0_mb: 412.6,
                l2_weight_prefetch_overlaps: 5408,
            },
            dspark: DSparkStepStats::from_counts(52, 260, 38, 204),
            timing: RequestTimingStats {
                ttft_ms: 698.4,
                prefill_tok_per_s: 5864.8,
                decode_tok_per_s: 134.2,
                min_rank_uma_free_gib: 9.42,
            },
            max_seq_len: max_seq_len(),
        }
    }
}

/// What an engine can actually serve.
///
/// Every field defaults to `false`, which is the most conservative answer there is: an engine that
/// has not declared a channel is treated as not having it, so `GET /v1/models` can never advertise
/// a capability by omission. A driver opts in by saying what it really does.
///
/// There is deliberately no `vision` (or `multimodal`) field. The image path is the 263
/// `vision.*`/`aligner.*` tensors plus the three image delimiters, and the tensor plan does not
/// load any of them (DECISIONS.md D18), so no engine in this workspace can serve an image. A
/// channel no driver can implement is not a capability that defaults to `false`; it is absent, and
/// a request that needs it is refused with [`MissingCapability`] rather than answered from its text
/// tokens or charged a visual-token count derived from an assumed image grid. When the vision path
/// is wired, the channel returns here as a field that is `true` only for an engine that really
/// carries the tensors - and `server::list_models` derives `multimodal`/`image_url` from it instead
/// of the constants there today, in the same change that overrides `validate_request`. The parsed
/// payload - including the decoded bytes of inline images - is already carried on
/// [`EngineRequest::images`], so only the vision forward (which decodes those bytes to pixels) and
/// the visual-token ids are missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Capabilities {
    /// Whether the engine can emit tool calls.
    pub tools: bool,
    /// Whether the engine produces a reasoning channel.
    pub thinking: bool,
    /// Whether the engine honours a caller's `reasoning_effort`.
    pub reasoning_effort: bool,
}

impl Capabilities {
    /// Every optional channel this workspace implements, declared: what an engine that does all of
    /// it answers. Vision is not in the list because no engine here implements it (see the type
    /// docs).
    pub const ALL: Self = Self {
        tools: true,
        thinking: true,
        reasoning_effort: true,
    };
}

/// A capability a request can require and an engine can lack.
///
/// This is the vocabulary of the *missing* capability. [`Capabilities`] says what an engine has;
/// a request that needs a channel the engine lacks is refused with [`MissingCapability`] before any
/// generation. Vision is the only such channel today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// An engine that can turn image parts into visual tokens: the vision tower, the aligner and the
    /// three image delimiters.
    Vision,
}

impl Capability {
    /// The stable spelling the Display impl and error messages use.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vision => "vision",
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A request needs a capability the engine does not implement.
///
/// Returned by [`InferenceEngine::validate_request`] before any generation, so the HTTP layer can
/// answer with a client error that names the missing channel instead of streaming an answer computed
/// from the request's text tokens alone. The only channel today is [`Capability::Vision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("this engine does not implement the {capability} capability: {detail}")]
pub struct MissingCapability {
    capability: Capability,
    detail: &'static str,
}

impl MissingCapability {
    /// The channel the request needed.
    #[must_use]
    pub fn capability(self) -> Capability {
        self.capability
    }

    /// The machine-readable code the HTTP layer answers with.
    #[must_use]
    pub fn code(self) -> &'static str {
        "unsupported_capability"
    }

    /// The missing vision capability, with the reason the image path cannot be served.
    ///
    /// The reason is the same one [`Capabilities`] cites: the image path is 263
    /// `vision.*`/`aligner.*` tensors plus the three image delimiters, and the tensor plan loads
    /// none of them (DECISIONS.md D18).
    #[must_use]
    pub fn vision() -> Self {
        Self {
            capability: Capability::Vision,
            detail: "the 263 vision/aligner tensors and the three image delimiters are not wired into \
                     the tensor plan (DECISIONS.md D18), so an image part cannot be turned into visual \
                     tokens; the request is refused rather than answered from its text tokens alone",
        }
    }
}

/// The model runtime behind the HTTP layer.
pub trait InferenceEngine: Send + Sync {
    /// Canonical model id served by this engine.
    fn model_id(&self) -> &str;

    /// Runs one request, yielding normalized decode events.
    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent>;

    /// Current cluster telemetry snapshot.
    fn cluster_stats(&self) -> ClusterStatsSnapshot;

    /// What this engine can serve beyond plain text.
    ///
    /// This is what `GET /v1/models` advertises. It defaults to [`Capabilities::default`], the most
    /// conservative answer: a channel is something a driver has to state, never something assumed on
    /// its behalf.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Checks that this engine can serve `request` before any generation.
    ///
    /// The default implementation refuses a request that carries an image part with
    /// [`MissingCapability::vision`]: no engine in this workspace implements the image path, and
    /// [`Capabilities`] deliberately has no vision channel to consult. An engine that really can
    /// serve images overrides this method together with the capability surface, so the refusal and
    /// the advertisement stay the same fact.
    ///
    /// # Errors
    /// [`MissingCapability`] naming the first channel the request needs and the engine lacks.
    fn validate_request(&self, request: &EngineRequest) -> Result<(), MissingCapability> {
        if !request.images.is_empty() {
            return Err(MissingCapability::vision());
        }
        Ok(())
    }
}

/// One scripted tool call for MockEngine.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MockToolCall {
    /// Tool-call id reported on the first delta.
    pub id: String,
    /// Function name reported on the first delta.
    pub name: String,
    /// Argument fragments, each emitted as its own delta.
    pub arguments: Vec<String>,
}

impl MockToolCall {
    /// Builds a tool call from its id, name and argument fragments.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: &[&str]) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.iter().map(|a| (*a).to_owned()).collect(),
        }
    }
}

/// The scripted decode stream of a MockEngine.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MockResponse {
    /// Reasoning-channel token pieces.
    pub reasoning: Vec<String>,
    /// Visible-content token pieces.
    pub content: Vec<String>,
    /// Tool calls, in index order.
    pub tool_calls: Vec<MockToolCall>,
    /// Finish reason used when the script is not truncated or aborted.
    pub finish_reason: FinishReason,
}

/// Deterministic, weights-free InferenceEngine used by the router tests.
///
/// The emitted event sequence is a pure function of the request and the engine
/// configuration, which is what lets the SSE tests assert exact frames.
#[derive(Debug, Clone)]
pub struct MockEngine {
    model: String,
    response: MockResponse,
    delay: Option<Duration>,
    cluster: ClusterStatsSnapshot,
    kv: Arc<KvCacheTelemetry>,
    uma_matched: usize,
    nvme_matched: usize,
    swa_tail_hit: bool,
    accepted_prediction_tokens: usize,
    rejected_prediction_tokens: usize,
    loop_guard: Option<LoopAbortConfig>,
    /// What this mock answers for [`InferenceEngine::capabilities`].
    capabilities: Capabilities,
}

impl MockEngine {
    /// Creates a mock engine with an empty script and sample cluster telemetry.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            response: MockResponse::default(),
            delay: None,
            cluster: ClusterStatsSnapshot::sample(),
            kv: Arc::new(KvCacheTelemetry::default()),
            uma_matched: 0,
            nvme_matched: 0,
            swa_tail_hit: false,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
            loop_guard: None,
            capabilities: Capabilities::default(),
        }
    }

    /// Sets what this mock says it can serve.
    ///
    /// The mock never actually does any of it: these are the answers
    /// [`InferenceEngine::capabilities`] gives, and nothing more, so that the capability line of
    /// `/v1/models` can be pinned in every direction. The default is [`Capabilities::default`], the
    /// most conservative set there is.
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Replaces the whole scripted response.
    #[must_use]
    pub fn with_response(mut self, response: MockResponse) -> Self {
        self.response = response;
        self
    }

    /// Replaces the reasoning-channel pieces.
    #[must_use]
    pub fn reasoning(mut self, pieces: &[&str]) -> Self {
        self.response.reasoning = pieces.iter().map(|p| (*p).to_owned()).collect();
        self
    }

    /// Replaces the visible-content pieces.
    #[must_use]
    pub fn content(mut self, pieces: &[&str]) -> Self {
        self.response.content = pieces.iter().map(|p| (*p).to_owned()).collect();
        self
    }

    /// Appends one scripted tool call.
    #[must_use]
    pub fn tool_call(mut self, id: &str, name: &str, arguments: &[&str]) -> Self {
        self.response
            .tool_calls
            .push(MockToolCall::new(id, name, arguments));
        self
    }

    /// Sets the finish reason used when the script completes normally.
    #[must_use]
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.response.finish_reason = reason;
        self
    }

    /// Injects a delay before every event after the first.
    #[must_use]
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    /// Configures the simulated prefix-cache match for every request.
    #[must_use]
    pub fn with_cache(
        mut self,
        uma_matched: usize,
        nvme_matched: usize,
        swa_tail_hit: bool,
    ) -> Self {
        self.uma_matched = uma_matched;
        self.nvme_matched = nvme_matched;
        self.swa_tail_hit = swa_tail_hit;
        self
    }

    /// Replaces the cluster telemetry template (and therefore the per-request stats that
    /// are not derived from the request itself).
    #[must_use]
    pub fn with_cluster_stats(mut self, snapshot: ClusterStatsSnapshot) -> Self {
        self.cluster = snapshot;
        self
    }

    /// Configures the accepted/rejected speculative-token counters in the usage block.
    #[must_use]
    pub fn with_prediction_tokens(mut self, accepted: usize, rejected: usize) -> Self {
        self.accepted_prediction_tokens = accepted;
        self.rejected_prediction_tokens = rejected;
        self
    }

    /// Runs a loop-abort detector over the scripted pieces; a fire ends the stream with
    /// FinishReason::LoopAbort.
    #[must_use]
    pub fn with_loop_abort(mut self, config: LoopAbortConfig) -> Self {
        self.loop_guard = Some(config);
        self
    }

    /// The live KV telemetry counters feeding cluster_stats().
    #[must_use]
    pub fn kv_telemetry(&self) -> Arc<KvCacheTelemetry> {
        Arc::clone(&self.kv)
    }

    fn engram_pair(&self) -> [EngramTelemetrySnapshot; 2] {
        let mut layers = default_engram_layers();
        for (slot, source) in layers.iter_mut().zip(self.cluster.engram.iter()) {
            *slot = source.clone();
        }
        layers
    }

    fn script(&self, request: &EngineRequest) -> Vec<EngineEvent> {
        let prompt_tokens = request.prompt_tokens.len();
        let plan = self.kv.plan_request(
            prompt_tokens,
            self.uma_matched,
            self.nvme_matched,
            self.swa_tail_hit,
        );

        let mut events = Vec::new();
        events.push(EngineEvent::Started {
            prompt_tokens,
            cached_tokens: plan.global_kv_cached_tokens,
        });

        let mut budget = request.max_tokens;
        let mut completion_tokens = 0usize;
        let mut reasoning_tokens = 0usize;
        let mut truncated = false;
        let mut abort: Option<LoopAbort> = None;
        let mut detector = self.loop_guard.map(LoopAbortDetector::new);

        'script: {
            for piece in &self.response.reasoning {
                if budget == 0 {
                    truncated = true;
                    break 'script;
                }
                if let Some(reason) = detector.as_mut().and_then(|guard| guard.push(piece)) {
                    abort = Some(reason);
                    break 'script;
                }
                budget -= 1;
                completion_tokens += 1;
                reasoning_tokens += 1;
                events.push(EngineEvent::ReasoningDelta(piece.clone()));
            }
            for piece in &self.response.content {
                if budget == 0 {
                    truncated = true;
                    break 'script;
                }
                if let Some(reason) = detector.as_mut().and_then(|guard| guard.push(piece)) {
                    abort = Some(reason);
                    break 'script;
                }
                budget -= 1;
                completion_tokens += 1;
                events.push(EngineEvent::ContentDelta(piece.clone()));
            }
            for (index, call) in self.response.tool_calls.iter().enumerate() {
                if budget == 0 {
                    truncated = true;
                    break 'script;
                }
                let (first, rest): (&str, &[String]) = match call.arguments.split_first() {
                    Some((head, tail)) => (head.as_str(), tail),
                    None => ("", &[]),
                };
                budget -= 1;
                completion_tokens += 1;
                events.push(EngineEvent::ToolCallDelta {
                    index,
                    id: Some(call.id.clone()),
                    name: Some(call.name.clone()),
                    arguments_delta: first.to_owned(),
                });
                for argument in rest {
                    if budget == 0 {
                        truncated = true;
                        break 'script;
                    }
                    budget -= 1;
                    completion_tokens += 1;
                    events.push(EngineEvent::ToolCallDelta {
                        index,
                        id: None,
                        name: None,
                        arguments_delta: argument.clone(),
                    });
                }
            }
        }

        let finish_reason = if abort.is_some() {
            FinishReason::LoopAbort
        } else if truncated {
            FinishReason::Length
        } else {
            self.response.finish_reason
        };

        let usage = OpenAiUsageWithCache {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: plan.global_kv_cached_tokens,
                uma_cached_tokens: plan.uma_hit_tokens,
                nvme_cached_tokens: plan.nvme_restored_tokens,
                // No engine here produces visual tokens. The router refuses an image request via
                // validate_request; a direct caller that bypasses that check still gets no
                // fabricated count from this mock.
                image_tokens: 0,
            },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens,
                accepted_prediction_tokens: self.accepted_prediction_tokens.min(completion_tokens),
                rejected_prediction_tokens: self.rejected_prediction_tokens.min(completion_tokens),
            },
        };

        let stats = DeepSeekSparkRequestStats {
            kv_cache_hit_rate: finite_or_zero(plan.hit_rate()),
            kv_cache: plan,
            engram_layers: self.engram_pair(),
            dspark_speculative: self.cluster.dspark,
            transport: self.cluster.transport.clone(),
            timing: self.cluster.timing,
        };

        events.push(EngineEvent::Finished {
            finish_reason,
            usage,
            stats: Box::new(stats),
        });
        events
    }
}

impl InferenceEngine for MockEngine {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        let events = self.script(&request);
        match self.delay {
            None => Box::pin(futures::stream::iter(events)),
            Some(delay) => Box::pin(async_stream::stream! {
                let mut first = true;
                for event in events {
                    if !first {
                        tokio::time::sleep(delay).await;
                    }
                    first = false;
                    yield event;
                }
            }),
        }
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        let mut snapshot = self.cluster.clone();
        snapshot.model = self.model.clone();
        snapshot.kv = self.kv.snapshot();
        // Refresh the configured window so the endpoint reports the value in use now.
        snapshot.max_seq_len = max_seq_len();
        snapshot
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }
}

/// Convenience: the prefix-match plan a request produced, for tests and callers that only
/// care about the cache half of the telemetry.
#[must_use]
pub fn plan_for(
    prompt_tokens: usize,
    uma_matched: usize,
    nvme_matched: usize,
    swa_tail_hit: bool,
) -> PrefixMatchPlan {
    KvCacheTelemetry::default().plan_request(prompt_tokens, uma_matched, nvme_matched, swa_tail_hit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn request() -> EngineRequest {
        EngineRequest {
            request_id: "req-1".to_owned(),
            model: MODEL_ID.to_owned(),
            prompt_tokens: vec![1, 2, 3],
            images: Vec::new(),
            max_tokens: 64,
            temperature: 1.0,
            top_p: 1.0,
            thinking: true,
            reasoning_effort: Some(75),
            stop: Vec::new(),
            seed: None,
            trace_scopes: 0,
            trace_steps: 0,
        }
    }

    fn collect(engine: &MockEngine, request: EngineRequest) -> Vec<EngineEvent> {
        futures::executor::block_on(engine.generate(request).collect::<Vec<_>>())
    }

    /// An engine that declares nothing: it answers with the trait's conservative default.
    struct SilentEngine;

    impl InferenceEngine for SilentEngine {
        fn model_id(&self) -> &str {
            "silent"
        }

        fn generate(&self, _request: EngineRequest) -> BoxStream<'static, EngineEvent> {
            Box::pin(futures::stream::empty())
        }

        fn cluster_stats(&self) -> ClusterStatsSnapshot {
            ClusterStatsSnapshot::default()
        }
    }

    #[test]
    fn an_engine_that_declares_nothing_answers_with_the_conservative_default() {
        assert_eq!(SilentEngine.capabilities(), Capabilities::default());
        assert_eq!(
            MockEngine::new(MODEL_ID).capabilities(),
            Capabilities::default()
        );
        assert_eq!(
            MockEngine::new(MODEL_ID)
                .with_capabilities(Capabilities::ALL)
                .capabilities(),
            Capabilities::ALL
        );
    }

    #[test]
    fn the_capability_set_has_no_vision_channel() {
        // A struct literal, so re-adding a `vision` field fails to compile until this test and the
        // refusal in `validate_request` are updated together: there is no channel to advertise while
        // the 263 vision/aligner tensors are unwired (DECISIONS.md D18, D27).
        let every_implemented_channel = Capabilities {
            tools: true,
            thinking: true,
            reasoning_effort: true,
        };
        assert_eq!(Capabilities::ALL, every_implemented_channel);
    }

    #[test]
    fn a_request_with_an_image_part_is_refused_with_the_missing_capability() {
        let engine = MockEngine::new(MODEL_ID);
        let mut request = request();
        request.images = vec![ImagePart::Placeholder];
        let error = engine
            .validate_request(&request)
            .expect_err("no engine in this workspace implements the vision path");
        assert_eq!(error.capability(), Capability::Vision);
        assert_eq!(error.code(), "unsupported_capability");
        let message = error.to_string();
        assert!(message.contains("vision"), "names the channel: {message}");
        assert!(
            message.contains("263"),
            "names the unwired tensors: {message}"
        );
    }

    #[test]
    fn a_text_request_passes_the_capability_check() {
        let engine = MockEngine::new(MODEL_ID);
        assert_eq!(engine.validate_request(&request()), Ok(()));
    }

    #[test]
    fn a_scripted_request_with_an_image_part_charges_no_fabricated_visual_tokens() {
        // The mock is handed a request that bypassed the router's refusal. It still charges only the
        // prompt it was given: there is no 32x32 grid to fall back on (D27).
        let engine = MockEngine::new(MODEL_ID).content(&["ok"]);
        let mut request = request();
        request.images = vec![ImagePart::Placeholder];
        let events = collect(&engine, request);
        match events.first() {
            Some(EngineEvent::Started { prompt_tokens, .. }) => assert_eq!(*prompt_tokens, 3),
            other => panic!("expected Started, got {other:?}"),
        }
        match events.last() {
            Some(EngineEvent::Finished { usage, .. }) => {
                assert_eq!(usage.prompt_tokens, 3);
                assert_eq!(usage.prompt_tokens_details.image_tokens, 0);
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }

    #[test]
    fn scripted_stream_is_reproducible_and_ordered() {
        let engine = MockEngine::new(MODEL_ID)
            .reasoning(&["think "])
            .content(&["Hello", " world"])
            .with_cache(3, 0, true);
        let sample = ClusterStatsSnapshot::sample();
        let events = collect(&engine, request());
        assert_eq!(
            events,
            vec![
                EngineEvent::Started {
                    prompt_tokens: 3,
                    cached_tokens: 0
                },
                EngineEvent::ReasoningDelta("think ".to_owned()),
                EngineEvent::ContentDelta("Hello".to_owned()),
                EngineEvent::ContentDelta(" world".to_owned()),
                EngineEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    usage: OpenAiUsageWithCache {
                        prompt_tokens: 3,
                        completion_tokens: 3,
                        total_tokens: 6,
                        prompt_tokens_details: PromptTokensDetails {
                            cached_tokens: 0,
                            uma_cached_tokens: 0,
                            nvme_cached_tokens: 0,
                            image_tokens: 0,
                        },
                        completion_tokens_details: CompletionTokensDetails {
                            reasoning_tokens: 1,
                            accepted_prediction_tokens: 0,
                            rejected_prediction_tokens: 0,
                        },
                    },
                    stats: Box::new(DeepSeekSparkRequestStats {
                        kv_cache_hit_rate: 0.0,
                        kv_cache: plan_for(3, 3, 0, true),
                        engram_layers: [sample.engram[0].clone(), sample.engram[1].clone()],
                        dspark_speculative: sample.dspark,
                        transport: sample.transport.clone(),
                        timing: sample.timing,
                    }),
                },
            ]
        );
        // A second run is byte-identical.
        assert_eq!(collect(&engine, request()), events);
    }

    #[test]
    fn max_tokens_truncates_and_reports_length() {
        let engine = MockEngine::new(MODEL_ID).content(&["a", "b", "c", "d"]);
        let mut request = request();
        request.max_tokens = 2;
        let events = collect(&engine, request);
        let deltas: Vec<&EngineEvent> = events
            .iter()
            .filter(|event| matches!(event, EngineEvent::ContentDelta(_)))
            .collect();
        assert_eq!(deltas.len(), 2);
        match events.last() {
            Some(EngineEvent::Finished {
                finish_reason,
                usage,
                ..
            }) => {
                assert_eq!(*finish_reason, FinishReason::Length);
                assert_eq!(usage.completion_tokens, 2);
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }

    #[test]
    fn tool_calls_stream_their_argument_fragments() {
        let engine = MockEngine::new(MODEL_ID)
            .tool_call("call_1", "get_weather", &["{\"loc", "ation\": \"SF\"}"])
            .with_finish_reason(FinishReason::ToolCalls);
        let events = collect(&engine, request());
        let deltas: Vec<&EngineEvent> = events
            .iter()
            .filter(|event| matches!(event, EngineEvent::ToolCallDelta { .. }))
            .collect();
        assert_eq!(deltas.len(), 2);
        match deltas[0] {
            EngineEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                assert_eq!(*index, 0);
                assert_eq!(id.as_deref(), Some("call_1"));
                assert_eq!(name.as_deref(), Some("get_weather"));
                assert_eq!(arguments_delta, "{\"loc");
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn loop_guard_finishes_with_loop_abort() {
        let config = LoopAbortConfig {
            ngram: 2,
            max_repeats: 3,
            ..LoopAbortConfig::default()
        };
        let engine = MockEngine::new(MODEL_ID)
            .content(&["a", "b", "a", "b", "a", "b"])
            .with_loop_abort(config);
        let events = collect(&engine, request());
        match events.last() {
            Some(EngineEvent::Finished {
                finish_reason,
                usage,
                ..
            }) => {
                assert_eq!(*finish_reason, FinishReason::LoopAbort);
                assert_eq!(usage.completion_tokens, 5);
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }

    #[test]
    fn cluster_stats_reports_the_model_and_live_kv_counters() {
        let engine = MockEngine::new(MODEL_ID).content(&["ok"]);
        let snapshot = engine.cluster_stats();
        assert_eq!(snapshot.model, MODEL_ID);
        assert_eq!(snapshot.kv.requests_total, 0);
        assert_eq!(snapshot.nodes.len(), 4);
        let _ = collect(&engine, request());
        assert_eq!(engine.cluster_stats().kv.requests_total, 1);
        assert_eq!(engine.cluster_stats().kv.prompt_tokens_total, 3);
    }
}
