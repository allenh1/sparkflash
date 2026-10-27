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
use sparkflash_model::telemetry::{
    EngramTelemetrySnapshot, KvCacheStatsSnapshot, KvCacheTelemetry, PrefixMatchPlan,
};

use crate::loop_abort::{LoopAbort, LoopAbortConfig, LoopAbortDetector};
use crate::multimodal::ImagePart;
use crate::telemetry_types::{
    CompletionTokensDetails, DSparkStepStats, DeepSeekSparkRequestStats, OpenAiUsageWithCache,
    PromptTokensDetails, RequestTimingStats, RoceTransportStats, finite_or_zero,
};
use crate::vision::VisionImage;

/// Canonical Hugging Face model id served by this cluster.
///
/// Kept as a compatibility constant for the V4.1 default. A selected model's descriptor is the
/// source of truth on the HTTP path; see [`crate::state::AppState::descriptor`].
pub const MODEL_ID: &str = sparkflash_model::DEFAULT_MODEL_ID;
/// Maximum context window of the V4.1 model, in tokens.
///
/// A selected model's [`sparkflash_model::ModelDescriptor::context_tokens`] is the source of
/// truth; this constant is the default used by weight-free tests and the legacy helpers below.
pub const MODEL_CONTEXT_TOKENS: usize = sparkflash_model::V41_CONTEXT_TOKENS;

/// Environment variable that overrides the sequence length the server actually serves.
pub const MAX_SEQ_LEN_ENV: &str = "SPARK_MAX_SEQ_LEN";

/// Smallest sequence length the server will serve.
///
/// The floor is one token: zero is not a window at all, while any positive value is
/// technically servable. We refuse only windows the engine provably cannot honour (zero,
/// or one wider than the model's), rather than second-guessing an operator who really
/// wants a deliberately tiny window.
pub const MIN_SEQ_LEN: usize = 1;

/// Parses a SPARK_MAX_SEQ_LEN value against the V4.1 window.
///
/// This is the compatibility wrapper; a model-aware caller uses [`parse_max_seq_len_within`]
/// with its descriptor's `context_tokens`.
#[must_use]
pub fn parse_max_seq_len(raw: &str) -> Option<usize> {
    parse_max_seq_len_within(raw, MODEL_CONTEXT_TOKENS)
}

/// Parses a SPARK_MAX_SEQ_LEN value, returning None when it must be ignored.
///
/// Accepts a pure decimal integer within `MIN_SEQ_LEN..=max_context`. Surrounding whitespace is
/// tolerated because it survives shell quoting. Everything else is rejected: a sign
/// (`usize::from_str` would happily take `"+131072"`), underscore separators, zero, and anything
/// wider than the model's window.
#[must_use]
pub fn parse_max_seq_len_within(raw: &str, max_context: usize) -> Option<usize> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || !trimmed.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    trimmed
        .parse::<usize>()
        .ok()
        .filter(|value| (MIN_SEQ_LEN..=max_context).contains(value))
}

/// The maximum sequence length in use for the V4.1 default, per DECISIONS.md D5.
#[must_use]
pub fn max_seq_len() -> usize {
    max_seq_len_for(MODEL_CONTEXT_TOKENS)
}

/// The maximum sequence length in use for a model with `max_context` tokens.
///
/// SPARK_MAX_SEQ_LEN wins when it is usable (see [`parse_max_seq_len_within`]). Anything else -
/// unset, unparsable, zero, or wider than the model's window - falls back to `max_context` with a
/// warning, so a typo can never make the server advertise a window it cannot honour.
#[must_use]
pub fn max_seq_len_for(max_context: usize) -> usize {
    let Ok(raw) = std::env::var(MAX_SEQ_LEN_ENV) else {
        return max_context;
    };
    match parse_max_seq_len_within(&raw, max_context) {
        Some(value) => value,
        None => {
            tracing::warn!(
                value = %raw,
                "ignoring {MAX_SEQ_LEN_ENV}: not a usable sequence length, serving {max_context}"
            );
            max_context
        }
    }
}
/// The recipe-contract upper bound on a requested `max_tokens`.
///
/// The recipe uses 393,216 as its protocol upper bound; the actual generation is
/// still lowered to the context left after the prompt by the router. The CLI's
/// `--max-tokens` default uses the same ceiling so the serve cap cannot silently
/// truncate a request the API advertises as legal.
///
/// A selected model's descriptor may narrow this; the HTTP path asks the descriptor.
pub const MODEL_MAX_OUTPUT_TOKENS: usize = sparkflash_model::V41_MAX_OUTPUT_TOKENS;
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

/// How many alternative logprobs one position reports.
///
/// The wire values are the ones vLLM accepts: a non-negative top-k count, or `-1` for the whole
/// vocabulary. The router resolves the wire integer to this type (and validates it against the
/// engine's vocabulary size and the server's logprob cap) before an [`EngineRequest`] exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogprobSelection {
    /// The `n` most likely alternatives. The position's own token is always reported in addition,
    /// which is why the OpenAI completion object can carry `n + 1` entries.
    Top(u32),
    /// The whole vocabulary, in token-id order and unranked: vLLM's `-1`.
    All,
}

impl LogprobSelection {
    /// The number of alternatives a `Top` selection asks the engine to select.
    #[must_use]
    pub fn top(self) -> Option<usize> {
        match self {
            Self::Top(count) => Some(count as usize),
            Self::All => None,
        }
    }
}

/// The logprobs one request asks for.
///
/// This is the engine side of the OpenAI `logprobs` / `top_logprobs` / `prompt_logprobs` /
/// `logprob_token_ids` fields; the router resolves and validates the wire values before it builds
/// this. The default asks for nothing, and an engine that is not asked does no logprob work and
/// emits no logprob event, so the cost only exists on requests that requested it. Prompt logprobs
/// are the expensive half for the reference driver (see `sparkflash-cli`'s `serve`: it has no
/// all-position logit API, so it feeds the prompt token by token to score each position).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogprobsRequest {
    /// Per generated token: how many alternatives the engine reports, if any.
    pub sample: Option<LogprobSelection>,
    /// Per prompt token: how many alternatives the engine reports, if any. Position 0 always
    /// reports `None` because it has no conditioning token to be scored against.
    pub prompt: Option<LogprobSelection>,
    /// Specific vocabulary ids whose logprobs are wanted at every generated position, in request
    /// order. When non-empty this takes precedence over `sample`'s natural top-k, as vLLM's
    /// `logprob_token_ids` does.
    pub token_ids: Vec<u32>,
}

impl LogprobsRequest {
    /// Whether this request asks for per-token sample logprobs (natural top-k or explicit ids).
    #[must_use]
    pub fn wants_sample(&self) -> bool {
        self.sample.is_some() || !self.token_ids.is_empty()
    }

    /// Whether this request asks for prompt logprobs.
    #[must_use]
    pub fn wants_prompt(&self) -> bool {
        self.prompt.is_some()
    }

    /// Whether any logprob work is requested at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.wants_sample() && !self.wants_prompt()
    }
}

/// One token and its logprob, as reported for one position.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// The vocabulary id of the token this position is about (the sampled token for a generated
    /// position, the prompt token for a prompt position).
    pub token_id: u32,
    /// Its decoded string. The engine owns the tokenizer, so the router never decodes a token id
    /// itself: this is the string that goes into `tokens[]`, `top_logprobs` keys and
    /// `decoded_token`.
    pub token: String,
    /// Its log probability (natural log).
    pub logprob: f32,
    /// Its 1-based vocabulary rank: the number of tokens with a strictly greater logprob, plus one.
    /// `None` when the request asked for the whole vocabulary, which vLLM leaves unranked.
    pub rank: Option<u32>,
    /// The alternatives, in vLLM's order: the position's own token first, then the selected
    /// alternatives. For a natural top-k that is the top-k in descending logprob with duplicates
    /// collapsed (the position's own token keeps first place); for `All` it is the whole
    /// vocabulary in token-id order; for explicit `token_ids` it is the requested ids in request
    /// order.
    pub top: Vec<LogprobAlternative>,
}

/// One alternative in a [`TokenLogprob`].
#[derive(Debug, Clone, PartialEq)]
pub struct LogprobAlternative {
    /// Vocabulary id.
    pub token_id: u32,
    /// Decoded string.
    pub token: String,
    /// Log probability.
    pub logprob: f32,
    /// vLLM's rank for the entry: the true competition rank for the leading (own-token) entry and
    /// the top-k positional rank for the selected alternatives; `None` for an unranked `All`
    /// payload.
    pub rank: Option<u32>,
}

/// The decoded string of one prompt token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenString {
    /// Vocabulary id.
    pub token_id: u32,
    /// Decoded string, decoded the way vLLM decodes a logprob token: one id at a time, with the
    /// preceding sequence ids as context so a byte-fallback token gets the character it completes
    /// instead of a replacement glyph.
    pub token: String,
}

/// Prompt logprobs, one entry per prompt token.
///
/// `positions[0]` is always `None`: the first prompt token has no conditioning token, so there is
/// no distribution to score it under. vLLM reports the same `None` in the completion response.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptLogprobs {
    /// The decoded string of every prompt token, including position 0, because the `echo`
    /// response lists all of them in `tokens[]`.
    pub tokens: Vec<TokenString>,
    /// One logprob per prompt token; position 0 is `None`.
    pub positions: Vec<Option<TokenLogprob>>,
}

/// A fully normalized inference request.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineRequest {
    /// Caller-supplied or router-generated request id.
    pub request_id: String,
    /// Resolved model id.
    pub model: String,
    /// Prompt token ids (placeholder ids in Milestone 1). Each image's span is already expanded in
    /// place: every position of it is the model's image token id, and [`Self::vision`]
    /// says where each span starts and what fills it.
    pub prompt_tokens: Vec<u32>,
    /// The logprobs this request asks for. The default asks for none: no log-softmax, no top-k and
    /// no extra event, so a request that did not ask for logprobs pays nothing for the feature.
    pub logprobs: LogprobsRequest,
    /// Image content parts the router did not prepare, in request order.
    ///
    /// These are the parts of a request served without the vision path (the placeholder prompt
    /// path, or an engine without [`Capabilities::vision`]): there is no canvas and no span for
    /// them, and [`InferenceEngine::validate_request`] refuses any request that has one with
    /// [`MissingCapability::vision`] before generation, rather than answer from the text alone or
    /// charge a visual-token count from an assumed grid.
    pub images: Vec<ImagePart>,
    /// The prepared images, in prompt order: each span's place in [`Self::prompt_tokens`] and its
    /// canvas (DECISIONS.md D59). Filled only for an engine that declares [`Capabilities::vision`].
    pub vision: Vec<VisionImage>,
    /// Hard cap on generated tokens.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Number of most likely tokens the candidate filter keeps: 0 and -1 disable it, values above
    /// zero keep that many. vLLM refuses values below -1 (`sampling_params.py`: "top_k must be 0
    /// (disable), or at least 1").
    pub top_k: i32,
    /// Repetition penalty over the prompt and the generated tokens so far: 1.0 disables it, above 1
    /// discourages a repeated token, below 1 encourages it (`sampling_params.py` lines 248-253).
    pub repetition_penalty: f32,
    /// Presence penalty over generated tokens: subtracted once from every token that already
    /// appears in the generated text (`sampling_params.py` lines 240-243). 0.0 disables it.
    pub presence_penalty: f32,
    /// Frequency penalty over generated tokens: subtracted once per occurrence from every token
    /// in the generated text (`sampling_params.py` lines 244-246). 0.0 disables it.
    pub frequency_penalty: f32,
    /// Whether special tokens are hidden from the generated text; vLLM's OpenAI default is true
    /// and an explicit false passes the raw decoded text through.
    pub skip_special_tokens: bool,
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
    /// Logprobs for one generated token, emitted after that token's own content delta and only
    /// when the request asked for sample logprobs. One of these is emitted per generated token.
    TokenLogprobs(Box<TokenLogprob>),
    /// Prompt logprobs, emitted once after `Started` and before any delta or token logprob, only
    /// when the request asked for them. The payload's `positions[0]` is `None`.
    PromptLogprobs(Box<PromptLogprobs>),
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
/// The type lives in `sparkflash-model` because a model descriptor and an engine report the same
/// fact. It is re-exported here so every existing caller and test keeps its path.
///
/// Every field defaults to `false`: an engine that has not declared a channel is treated as not
/// having it, so `GET /v1/models` can never advertise a capability by omission. `vision` is
/// `true` only for an engine with the tower, aligner and image delimiters loaded (DECISIONS.md
/// D59, amending D18).
pub use sparkflash_model::Capabilities;

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
    /// The reason is the one [`Capabilities::vision`] stands for: the image path is 263
    /// `vision.*`/`aligner.*` tensors plus the three image delimiters, and this engine has not
    /// loaded them (DECISIONS.md D18, D59).
    #[must_use]
    pub fn vision() -> Self {
        Self {
            capability: Capability::Vision,
            detail: "the 263 vision/aligner tensors and the three image delimiters are not loaded by \
                     this engine (DECISIONS.md D18, D59), so an image part cannot be turned into \
                     visual tokens; the request is refused rather than answered from its text tokens \
                     alone",
        }
    }
}

/// The model runtime behind the HTTP layer.
pub trait InferenceEngine: Send + Sync {
    /// Canonical model id served by this engine.
    fn model_id(&self) -> &str;

    /// Runs one request, yielding normalized decode events.
    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent>;

    /// The vocabulary size this engine scores against, when it knows it.
    ///
    /// The router uses it to validate the wire logprob bounds the way vLLM validates against the
    /// model configuration: `-1` means the whole vocabulary, and explicit `logprob_token_ids`
    /// must name ids inside it. The default `None` means "not stated", and the router refuses the
    /// requests it cannot check rather than guessing a bound.
    fn vocab_size(&self) -> Option<usize> {
        None
    }

    /// Whether one request's multiple choices can be served as independent samples.
    ///
    /// The reference engine and the mock take one generate call per request, so repeated calls
    /// for one response are independent draws: the default is true. An engine whose sampling
    /// state is shared (the TP=4 group runs its four ranks as one lockstep sequence and its host
    /// sampler is temperature+seed only) returns false, and the router refuses n > 1 with a typed
    /// error instead of returning choices it cannot claim were drawn independently.
    fn independent_choices(&self) -> bool {
        true
    }

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

    /// Whether the engine can still serve: `/health` and `/readyz` answer 503 with the reason when
    /// it cannot (its loop failed, or has made no progress for too long).
    ///
    /// # Errors
    /// Why the engine cannot serve.
    fn health(&self) -> Result<(), String> {
        Ok(())
    }

    /// Checks that this engine can serve `request` before any generation.
    ///
    /// The default implementation refuses, with [`MissingCapability::vision`], a request that
    /// carries an unprepared image part, or prepared images when [`Self::capabilities`] does not
    /// declare vision: the refusal and the advertisement are the same fact.
    ///
    /// # Errors
    /// [`MissingCapability`] naming the first channel the request needs and the engine lacks.
    fn validate_request(&self, request: &EngineRequest) -> Result<(), MissingCapability> {
        if !request.images.is_empty() || (!request.vision.is_empty() && !self.capabilities().vision)
        {
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
    /// Scripted sample logprobs, one per content piece, emitted only when the request asks for
    /// sample logprobs. This is how the router tests drive the response assembly without a model.
    token_logprobs: Vec<TokenLogprob>,
    /// Scripted prompt logprobs, emitted only when the request asks for prompt logprobs.
    prompt_logprobs: Option<PromptLogprobs>,
    /// The vocabulary size this mock reports to `logprobs=-1` validation; `None` means unstated.
    vocab_size: Option<usize>,
    /// What this mock answers for [`InferenceEngine::health`]; `None` is healthy.
    unhealthy: Option<String>,
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
            token_logprobs: Vec::new(),
            prompt_logprobs: None,
            vocab_size: None,
            unhealthy: None,
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

    /// Scripts one sample logprob per content piece, in order.
    ///
    /// The mock emits these (one [`EngineEvent::TokenLogprobs`] after each content delta) only when
    /// the request actually asks for sample logprobs, so a test can pin both the gated delivery and
    /// the assembled response shape.
    #[must_use]
    pub fn with_token_logprobs(mut self, logprobs: Vec<TokenLogprob>) -> Self {
        self.token_logprobs = logprobs;
        self
    }

    /// Scripts the prompt logprobs payload, emitted (once, after `Started`) only when the request
    /// asks for prompt logprobs.
    #[must_use]
    pub fn with_prompt_logprobs(mut self, logprobs: PromptLogprobs) -> Self {
        self.prompt_logprobs = Some(logprobs);
        self
    }

    /// Reports a vocabulary size, which is what lets `logprobs=-1` and `logprob_token_ids` be
    /// validated by the router.
    #[must_use]
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = Some(vocab_size);
        self
    }

    /// Reports the engine unable to serve, for `reason`, on [`InferenceEngine::health`].
    #[must_use]
    pub fn with_unhealthy(mut self, reason: impl Into<String>) -> Self {
        self.unhealthy = Some(reason.into());
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
        // Prompt logprobs are one payload for the whole prompt, delivered once at the end of
        // prefill, exactly where vLLM delivers them.
        if request.logprobs.wants_prompt()
            && let Some(prompt_logprobs) = self.prompt_logprobs.clone()
        {
            events.push(EngineEvent::PromptLogprobs(Box::new(prompt_logprobs)));
        }

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
            for (content_index, piece) in self.response.content.iter().enumerate() {
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
                if request.logprobs.wants_sample()
                    && let Some(logprob) = self.token_logprobs.get(content_index)
                {
                    events.push(EngineEvent::TokenLogprobs(Box::new(logprob.clone())));
                }
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
                // The prepared spans' positions, which a vision engine reports; unprepared parts are
                // refused by validate_request and never counted.
                image_tokens: crate::vision::image_tokens(&request.vision),
            },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens,
                accepted_prediction_tokens: self.accepted_prediction_tokens.min(completion_tokens),
                rejected_prediction_tokens: self.rejected_prediction_tokens.min(completion_tokens),
            },
            prompt_cache_hit_tokens: plan.global_kv_cached_tokens,
            prompt_cache_miss_tokens: prompt_tokens.saturating_sub(plan.global_kv_cached_tokens),
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

    fn health(&self) -> Result<(), String> {
        self.unhealthy.clone().map_or(Ok(()), Err)
    }

    fn vocab_size(&self) -> Option<usize> {
        self.vocab_size
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
            logprobs: LogprobsRequest::default(),
            images: Vec::new(),
            vision: Vec::new(),
            max_tokens: 64,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            skip_special_tokens: true,
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
    fn all_declares_every_text_channel_and_leaves_vision_out() {
        // A struct literal, so a new channel fails to compile until this test says whether `ALL`
        // covers it. Vision is declared by an engine with the tower loaded, never by `ALL` (D59).
        let every_text_channel = Capabilities {
            tools: true,
            thinking: true,
            reasoning_effort: true,
            vision: false,
        };
        assert_eq!(Capabilities::ALL, every_text_channel);
    }

    fn prepared_image() -> VisionImage {
        let grid = sparkflash_model::vision::ImageGrid {
            n_llm_h: 1,
            n_llm_w: 1,
            best_height: 42,
            best_width: 42,
        };
        VisionImage {
            span: sparkflash_model::vision::ImageSpan {
                start: 1,
                grid,
                key: [0; 32],
            },
            pixels: vec![0; 42 * 42 * 3].into(),
        }
    }

    #[test]
    fn prepared_images_pass_only_an_engine_that_declares_vision() {
        let mut request = request();
        request.vision = vec![prepared_image()];
        let text = MockEngine::new(MODEL_ID).with_capabilities(Capabilities::ALL);
        assert_eq!(
            text.validate_request(&request).map_err(|e| e.capability()),
            Err(Capability::Vision)
        );
        let vision = MockEngine::new(MODEL_ID).with_capabilities(Capabilities {
            vision: true,
            ..Capabilities::ALL
        });
        assert_eq!(vision.validate_request(&request), Ok(()));
        // An unprepared part has no canvas and no span, whatever the engine declares.
        request.images = vec![ImagePart::Placeholder];
        assert!(vision.validate_request(&request).is_err());
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
                        prompt_cache_hit_tokens: 0,
                        prompt_cache_miss_tokens: 3,
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
