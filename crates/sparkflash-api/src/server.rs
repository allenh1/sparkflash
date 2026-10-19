//! The axum HTTP surface: OpenAI chat/completions/responses, Anthropic messages, cluster
//! telemetry, scoped tracing administration, trace export and Prometheus metrics.
//!
//! Every route is mounted by crate::server::router, and every response shape is derived
//! from the engine seam in crate::engine, so the whole surface is exercised by
//! tower::ServiceExt::oneshot tests without weights, GPUs or sockets.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::io::Write as _;
use std::sync::{Arc, OnceLock};

use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::StreamExt;
use serde_json::{Map, Value, json};
use sparkflash_trace::{
    GpuTraceRing, TraceAdminRequest, TraceAdminResponse, TraceExportOptions, TraceScope,
    active_mask, arm_with_gpu, bind_current_request, disarm_with_gpu, drain_cpu_events,
    parse_scopes_with_aliases, set_request_filter, summarize, to_chrome_trace_json,
};

use crate::engine::{
    EngineEvent, EngineRequest, FinishReason, InferenceEngine, LogprobSelection, LogprobsRequest,
    MODEL_MAX_OUTPUT_TOKENS, MODEL_OWNED_BY, PromptLogprobs, TokenLogprob, max_seq_len,
};
use crate::error::ApiError;
use crate::prompt::{
    PromptSummary, find_image_carrier, summarize_anthropic_messages, summarize_chat_messages,
    summarize_prompt_text, summarize_responses_input,
};
use crate::reasoning::{self, ReasoningSpec};
use crate::schema::{
    ChatCompletionRequest, CompletionRequest, MessagesRequest, PromptSpec, ResponsesRequest,
    StopSpec,
};
use crate::state::AppState;
use crate::telemetry_types::{DeepSeekSparkRequestStats, OpenAiUsageWithCache};

/// Decode-side trace request header.
pub const TRACE_HEADER: &str = "x-sparkflash-trace";
/// Response header echoing the armed trace step count.
pub const TRACE_STEPS_HEADER: &str = "x-sparkflash-trace-steps";
/// Response header echoing the armed scope names.
pub const TRACE_SCOPES_HEADER: &str = "x-sparkflash-trace-scopes";
/// Optional client-supplied id that correlates a request with a trace filter.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Default token budget when a request does not set `max_tokens`.
pub const DEFAULT_MAX_TOKENS: usize = 4096;

/// The default cap on requested logprobs, matching vLLM's `--max-logprobs` default.
///
/// The cap exists because a request for the whole vocabulary multiplies the response size by the
/// 129,280-token vocabulary per position. `sparkflash serve --max-logprobs N` changes it, and
/// `-1` lifts it, exactly as vLLM's flag does.
pub const DEFAULT_MAX_LOGPROBS: usize = 20;

/// vLLM's cap on the length of `logprob_token_ids`.
pub const MAX_LOGPROB_TOKEN_IDS: usize = 128;

/// Capacity of the host-side GPU trace ring, in events.
///
/// Milestone 1 has no CUDA graph yet, so the ring is a coherent host-side mirror: the CPU
/// mask is published into its header by arm_with_gpu, which is exactly what an in-graph
/// trace point reads once the graph exists.
pub const GPU_TRACE_RING_CAPACITY: usize = 4096;

/// The process-wide GPU trace ring.
///
/// This is an owned host allocation ([GpuTraceRing::new]), not device-visible memory: the
/// M1 server has no GPU producer, so no kernel writes slots. A real producer must
/// allocate this memory with cudaHostAllocMapped and adopt that pointer with
/// [GpuTraceRing::from_mapped_ptr]; until then /v1/admin/trace/dump reports
/// mapped: false rather than pretending the ring is device-coherent.
#[must_use]
pub fn gpu_trace_ring() -> &'static GpuTraceRing {
    static RING: OnceLock<GpuTraceRing> = OnceLock::new();
    RING.get_or_init(|| GpuTraceRing::new(GPU_TRACE_RING_CAPACITY))
}

/// Arms CPU span points and the mirrored in-graph GPU points together.
pub fn arm_traces(scopes: u64, steps: u32) -> u64 {
    arm_with_gpu(gpu_trace_ring().header(), scopes, steps)
}

/// Disarms the CPU scopes and the mirrored GPU ring mask together and drops any
/// single-request trace filter.
pub fn disarm_traces() {
    disarm_with_gpu(gpu_trace_ring().header());
    let _ = set_request_filter(None);
}

/// Binds a request id to the current thread for as long as the guard lives.
///
/// `sparkflash_trace` records a span only when the thread that would emit it is bound to
/// the id an installed trace filter targets, so a single-request capture follows the
/// request instead of every worker on the box. Dropping the guard puts back the id that
/// was bound before, which keeps reused worker threads and nested scopes correct.
pub struct RequestTraceBinding {
    previous: Option<String>,
}

impl RequestTraceBinding {
    /// Binds `request_id` and remembers the previous binding.
    pub fn new(request_id: Option<String>) -> Self {
        Self {
            previous: bind_current_request(request_id),
        }
    }
}

impl Drop for RequestTraceBinding {
    fn drop(&mut self) {
        bind_current_request(self.previous.take());
    }
}

/// The id a request is traced under: the caller's `X-Request-Id` when present and
/// non-empty, otherwise the server-generated id for the response.
#[must_use]
pub fn trace_request_id(headers: &HeaderMap, generated: &str) -> Option<String> {
    let client = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    Some(client.map_or_else(|| generated.to_owned(), str::to_owned))
}

/// Everything the SSE encoders need beyond the engine request itself.
struct StreamContext {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    trace_id: Option<String>,
}

/// A parsed `X-SparkFlash-Trace` directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceDirective {
    /// Scope mask to arm.
    pub scopes: u64,
    /// Engine steps for which the mask stays armed.
    pub steps: u32,
}

impl TraceDirective {
    /// Step count used when the header names scopes only.
    pub const DEFAULT_STEPS: u32 = 64;

    /// Parses `"<scopes>"`, `"<scopes>;steps=45"` or `"steps=45;<scopes>"`.
    ///
    /// # Errors
    /// Returns a 400 ApiError when the step count is not a number, when no scope list is
    /// present, or when the scope list contains an unknown name.
    pub fn parse(raw: &str) -> Result<Self, ApiError> {
        let mut scope_spec: Option<&str> = None;
        let mut steps = Self::DEFAULT_STEPS;
        for part in raw.split(';') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some(value) = part.strip_prefix("steps=") {
                steps = value.trim().parse::<u32>().map_err(|_| {
                    invalid_trace(format!("invalid X-SparkFlash-Trace step count: {value:?}"))
                })?;
            } else if scope_spec.is_none() {
                scope_spec = Some(part);
            } else {
                return Err(invalid_trace(
                    "X-SparkFlash-Trace carries more than one scope list".to_owned(),
                ));
            }
        }
        let spec = scope_spec.ok_or_else(|| {
            invalid_trace(
                "X-SparkFlash-Trace requires a scope list, e.g. \"engram;steps=45\"".to_owned(),
            )
        })?;
        let scopes = parse_scopes_with_aliases(spec).map_err(|error| {
            invalid_trace(format!("invalid X-SparkFlash-Trace scopes: {error}"))
        })?;
        Ok(Self { scopes, steps })
    }

    /// Arms the CPU scopes and mirrors them into the GPU ring for this many engine
    /// steps, returning the effective mask.
    pub fn arm(&self) -> u64 {
        arm_traces(self.scopes, self.steps)
    }
}

fn invalid_trace(message: String) -> ApiError {
    ApiError::invalid_request(
        message,
        Some("scopes".to_owned()),
        Some("invalid_trace_scopes".to_owned()),
    )
}

/// Builds the complete route table.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/v1/models", get(list_models))
        .route("/v1/cluster/stats", get(cluster_stats))
        .route("/v1/admin/trace", post(admin_trace))
        .route("/v1/admin/trace/dump", get(trace_dump))
        .route("/metrics", get(metrics))
        .with_state(state)
}

// ---------------------------------------------------------------------------------------
// Request helpers
// ---------------------------------------------------------------------------------------

fn parse_json(body: &Bytes) -> Result<Value, ApiError> {
    if body.is_empty() {
        return Err(ApiError::bad_request("request body is empty"));
    }
    serde_json::from_slice(body).map_err(ApiError::malformed_json)
}

fn invalid_body(error: serde_json::Error) -> ApiError {
    ApiError::invalid_request(
        format!("invalid request body: {error}"),
        None,
        Some("invalid_request".to_owned()),
    )
}

/// Refuses an image carrier found in the raw request body before generation.
///
/// This is D27's structural backstop: [`find_image_carrier`] walks every object and array, so
/// an image spelling no typed parser knows (the documented `computer_screenshot` output, a
/// `file_id`-only screenshot, a `code_interpreter_call` output, a case-variant
/// `type`) is refused with the same HTTP 400 `unsupported_capability` as the capability
/// seam, and the message names the shape and JSON path found. It runs before
/// `validate_request` because the typed seam only sees the parts a parser understood.
fn refuse_image_carrier(carrier: Option<String>) -> Result<(), ApiError> {
    let Some(shape) = carrier else {
        return Ok(());
    };
    Err(ApiError::invalid_request(
        format!(
            "this engine does not implement the vision capability: the request body carries an \
             image at {shape}, and the 263 vision/aligner tensors and the three image delimiters \
             are not wired into the tensor plan (DECISIONS.md D18), so it cannot be turned into \
             visual tokens; the request is refused rather than answered from its text tokens alone"
        ),
        None,
        Some("unsupported_capability".to_owned()),
    ))
}

fn resolve_reasoning(value: &Value) -> Result<ReasoningSpec, ApiError> {
    // The Responses API nests the effort in `reasoning.effort`; everything else uses the
    // top-level `reasoning_effort` that crate::reasoning understands.
    if value.get("reasoning_effort").is_none()
        && let Some(effort) = value
            .get("reasoning")
            .and_then(|reasoning| reasoning.get("effort"))
    {
        let mut synthetic = Map::new();
        synthetic.insert("reasoning_effort".to_owned(), effort.clone());
        for key in ["enable_thinking", "chat_template_kwargs"] {
            if let Some(flag) = value.get(key) {
                synthetic.insert(key.to_owned(), flag.clone());
            }
        }
        return reasoning::resolve(&Value::Object(synthetic)).map_err(ApiError::from);
    }
    reasoning::resolve(value).map_err(ApiError::from)
}

fn resolve_model(state: &AppState, requested: Option<&str>) -> Result<String, ApiError> {
    let engine_id = state.model_id();
    match requested.map(str::trim).filter(|model| !model.is_empty()) {
        None => Ok(engine_id.to_owned()),
        Some(requested) if model_matches(requested, engine_id) => Ok(engine_id.to_owned()),
        Some(requested) => Err(ApiError::model_not_found(requested)),
    }
}

fn model_matches(requested: &str, engine_id: &str) -> bool {
    if requested.eq_ignore_ascii_case(engine_id) {
        return true;
    }
    let requested_leaf = requested.rsplit('/').next().unwrap_or(requested);
    let engine_leaf = engine_id.rsplit('/').next().unwrap_or(engine_id);
    requested_leaf.eq_ignore_ascii_case(engine_leaf)
}

/// Resolves the requested output-token budget.
///
/// An absent budget means [`DEFAULT_MAX_TOKENS`]; an explicit `0` or a value above the
/// advertised ceiling [`MODEL_MAX_OUTPUT_TOKENS`] is a client error rather than being
/// silently clamped.
fn resolve_max_tokens(
    primary: Option<usize>,
    primary_name: &'static str,
    secondary: Option<usize>,
    secondary_name: &'static str,
) -> Result<usize, ApiError> {
    let (field, value) = match (primary, secondary) {
        (Some(value), _) => (primary_name, value),
        (None, Some(value)) => (secondary_name, value),
        (None, None) => return Ok(DEFAULT_MAX_TOKENS),
    };
    if value == 0 {
        return Err(ApiError::invalid_request(
            format!("{field} must be at least 1"),
            Some(field.to_owned()),
            Some("invalid_max_tokens".to_owned()),
        ));
    }
    if value > MODEL_MAX_OUTPUT_TOKENS {
        return Err(ApiError::invalid_request(
            format!("{field} must not exceed {MODEL_MAX_OUTPUT_TOKENS}"),
            Some(field.to_owned()),
            Some("invalid_max_tokens".to_owned()),
        ));
    }
    Ok(value)
}

fn normalize_stop(stop: Option<&StopSpec>) -> Vec<String> {
    match stop {
        None => Vec::new(),
        Some(StopSpec::One(sequence)) => vec![sequence.clone()],
        Some(StopSpec::Many(sequences)) => sequences.clone(),
    }
}

struct RequestBuilder {
    id: String,
    model: String,
    summary: PromptSummary,
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    spec: ReasoningSpec,
    stop: Vec<String>,
    seed: Option<u64>,
    logprobs: LogprobsRequest,
    trace: Option<TraceDirective>,
}

impl RequestBuilder {
    /// Builds the seam request. The parsed image parts are passed through - never turned into a
    /// fabricated visual-token count - so the engine's `validate_request` can refuse them (see
    /// [`crate::engine::MissingCapability`]) and a future vision engine has the payload it needs:
    /// inline parts carry the decoded bytes, not only a length. The handler calls
    /// `validate_request` before generation.
    fn build(self) -> EngineRequest {
        EngineRequest {
            request_id: self.id,
            model: self.model,
            prompt_tokens: self.summary.token_ids,
            images: self.summary.images,
            max_tokens: self.max_tokens,
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            thinking: self.spec.thinking,
            reasoning_effort: self.spec.effort,
            stop: self.stop,
            seed: self.seed,
            logprobs: self.logprobs,
            trace_scopes: self.trace.map_or(0, |directive| directive.scopes),
            trace_steps: self.trace.map_or(0, |directive| directive.steps),
        }
    }
}

fn trace_context(headers: &HeaderMap) -> Result<Option<TraceDirective>, ApiError> {
    let Some(raw) = headers.get(TRACE_HEADER) else {
        return Ok(None);
    };
    let raw = raw
        .to_str()
        .map_err(|_| ApiError::bad_request("X-SparkFlash-Trace header is not valid UTF-8"))?;
    let directive = TraceDirective::parse(raw)?;
    directive.arm();
    Ok(Some(directive))
}

fn apply_trace_headers(response: &mut Response, directive: &TraceDirective) {
    if let Ok(steps) = HeaderValue::from_str(&directive.steps.to_string()) {
        response.headers_mut().insert(TRACE_STEPS_HEADER, steps);
    }
    if let Ok(scopes) = HeaderValue::from_str(&TraceScope::describe(directive.scopes)) {
        response.headers_mut().insert(TRACE_SCOPES_HEADER, scopes);
    }
}

fn json_response(body: Value, trace: Option<TraceDirective>) -> Response {
    let mut response = Json(body).into_response();
    if let Some(directive) = trace.as_ref() {
        apply_trace_headers(&mut response, directive);
    }
    response
}

fn sse_response(
    stream: impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
    trace: Option<TraceDirective>,
) -> Response {
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    if let Some(directive) = trace.as_ref() {
        apply_trace_headers(&mut response, directive);
    }
    response
}

fn sse_frame(value: &Value) -> Bytes {
    let mut text = String::from("data: ");
    text.push_str(&serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned()));
    text.push_str("\n\n");
    Bytes::from(text)
}

fn sse_done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

// ---------------------------------------------------------------------------------------
// Engine driving
// ---------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
struct BufferedToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug)]
struct EngineOutput {
    reasoning: String,
    content: String,
    tool_calls: Vec<BufferedToolCall>,
    /// One logprob per generated token, in generation order. Empty when the engine was not asked
    /// for sample logprobs; an engine that is asked for them emits one
    /// [`EngineEvent::TokenLogprobs`] per generated token.
    token_logprobs: Vec<TokenLogprob>,
    /// Prompt logprobs, present when the engine was asked for them.
    prompt_logprobs: Option<PromptLogprobs>,
    finish_reason: FinishReason,
    usage: OpenAiUsageWithCache,
    stats: Box<DeepSeekSparkRequestStats>,
}

async fn drive_engine(
    engine: &dyn InferenceEngine,
    request: EngineRequest,
) -> Result<EngineOutput, ApiError> {
    let mut stream = engine.generate(request);
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut tool_calls: BTreeMap<usize, BufferedToolCall> = BTreeMap::new();
    let mut token_logprobs: Vec<TokenLogprob> = Vec::new();
    let mut prompt_logprobs: Option<PromptLogprobs> = None;
    let mut finished = None;
    while let Some(event) = stream.next().await {
        match event {
            EngineEvent::Started { .. } => {}
            EngineEvent::ReasoningDelta(text) => reasoning.push_str(&text),
            EngineEvent::ContentDelta(text) => content.push_str(&text),
            EngineEvent::PromptLogprobs(payload) => prompt_logprobs = Some(*payload),
            EngineEvent::TokenLogprobs(logprob) => token_logprobs.push(*logprob),
            EngineEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                let entry = tool_calls.entry(index).or_default();
                if let Some(id) = id {
                    entry.id = id;
                }
                if let Some(name) = name {
                    entry.name = name;
                }
                entry.arguments.push_str(&arguments_delta);
            }
            EngineEvent::Finished {
                finish_reason,
                usage,
                stats,
            } => {
                finished = Some((finish_reason, usage, stats));
            }
            EngineEvent::Error { message, code } => {
                return Err(ApiError::service_unavailable(message).with_code(code));
            }
        }
    }
    let Some((finish_reason, usage, stats)) = finished else {
        return Err(ApiError::service_unavailable(
            "engine stream ended without a finish event",
        ));
    };
    Ok(EngineOutput {
        reasoning,
        content,
        tool_calls: tool_calls.into_values().collect(),
        token_logprobs,
        prompt_logprobs,
        finish_reason,
        usage,
        stats,
    })
}

// ---------------------------------------------------------------------------------------
// Logprobs
// ---------------------------------------------------------------------------------------
//
// The wire layer mirrors vLLM's `logprobs_mode="raw_logprobs"` protocol: the engine reports the
// log-softmax of the raw logits (before temperature and penalties), the router only shapes it.
// The JSON shapes below are the ones vLLM emits, including its two oddities that clients depend
// on: the completion `top_logprobs` dict carries the sampled token in addition to the requested
// top-k entries, and the chat `top_logprobs` list is the first `n` entries of that dict (so it
// starts with the sampled token). `echo` with no explicit `prompt_logprobs` computes prompt
// logprobs with `logprobs`'s value, which is vLLM's rule.

/// What the response layer needs to know about the logprobs a request asked for.
#[derive(Debug, Clone, Copy, Default)]
struct LogprobInfo {
    /// Per generated token selection; `None` when the request asked for none.
    sample: Option<LogprobSelection>,
    /// Whether explicit `logprob_token_ids` selected the alternatives: vLLM then returns every
    /// entry of the position, truncation only applies to a natural top-k.
    explicit_token_ids: bool,
}

impl LogprobInfo {
    /// Whether generated-token logprobs were requested at all.
    fn wants_sample(self) -> bool {
        self.sample.is_some() || self.explicit_token_ids
    }
}

/// vLLM's clamp for a reported logprob: nothing below `-9999`, and a non-finite value (an
/// all-masked row, in principle) becomes the floor rather than an unrepresentable JSON number.
fn clamped_logprob(value: f32) -> f64 {
    if value.is_finite() {
        f64::from(value.max(-9999.0))
    } else {
        -9999.0
    }
}

/// The UTF-8 bytes of a decoded token, as a JSON array of byte values: OpenAI's and vLLM's
/// `bytes` field for chat logprobs.
fn utf8_bytes(text: &str) -> Value {
    Value::Array(text.as_bytes().iter().map(|byte| json!(byte)).collect())
}

/// Resolves one wire logprob count against the server cap and the engine's vocabulary size.
///
/// vLLM's rules: a negative value other than `-1` is refused; `-1` asks for the whole
/// vocabulary and must fit under `--max-logprobs`; a positive value must fit too. The error
/// strings are vLLM's spellings so a client that matches on them keeps working.
fn resolve_logprob_selection(
    raw: Option<i32>,
    field: &'static str,
    requested_for: &'static str,
    max_logprobs: usize,
    vocab_size: Option<usize>,
) -> Result<Option<LogprobSelection>, ApiError> {
    let Some(value) = raw else {
        return Ok(None);
    };
    if value < -1 {
        return Err(ApiError::invalid_request(
            format!("`{field}` must be a positive value or -1."),
            Some(field.to_owned()),
            Some("invalid_logprobs".to_owned()),
        ));
    }
    if value == -1 {
        let Some(vocab) = vocab_size else {
            return Err(ApiError::invalid_request(
                format!(
                    "`{field}` = -1 requests the whole vocabulary, but this engine does not report its vocabulary size, so the request cannot be bounded"
                ),
                Some(field.to_owned()),
                Some("invalid_logprobs".to_owned()),
            ));
        };
        if max_logprobs != usize::MAX && vocab > max_logprobs {
            return Err(ApiError::invalid_request(
                format!(
                    "Requested {requested_for} logprobs of {vocab}, which is greater than max allowed: {max_logprobs}"
                ),
                Some(field.to_owned()),
                Some("invalid_logprobs".to_owned()),
            ));
        }
        return Ok(Some(LogprobSelection::All));
    }
    let want = value as usize;
    if max_logprobs != usize::MAX && want > max_logprobs {
        return Err(ApiError::invalid_request(
            format!(
                "Requested {requested_for} logprobs of {want}, which is greater than max allowed: {max_logprobs}"
            ),
            Some(field.to_owned()),
            Some("invalid_logprobs".to_owned()),
        ));
    }
    Ok(Some(LogprobSelection::Top(value as u32)))
}

/// Resolves and validates `logprob_token_ids`.
///
/// vLLM's rules: the ids need `logprobs` to be set, the list is capped at
/// [`MAX_LOGPROB_TOKEN_IDS`], and every id must be inside the vocabulary. The sampling-parameter
/// rule that `logprobs` must equal the list length is *not* enforced: both OpenAI protocol layers
/// drop `logprobs` as soon as `logprob_token_ids` is set, so it is unreachable over HTTP.
fn resolve_logprob_token_ids(
    raw: Option<Vec<u32>>,
    logprobs: Option<LogprobSelection>,
    vocab_size: Option<usize>,
) -> Result<Vec<u32>, ApiError> {
    let Some(ids) = raw else {
        return Ok(Vec::new());
    };
    if ids.is_empty() {
        return Ok(ids);
    }
    if logprobs.is_none() {
        return Err(ApiError::invalid_request(
            "when using `logprob_token_ids`, `logprobs` must be set.",
            Some("logprob_token_ids".to_owned()),
            Some("invalid_logprobs".to_owned()),
        ));
    }
    if ids.len() > MAX_LOGPROB_TOKEN_IDS {
        return Err(ApiError::invalid_request(
            format!(
                "Requested logprob_token_ids of length {}, which is greater than max allowed: {MAX_LOGPROB_TOKEN_IDS}",
                ids.len()
            ),
            Some("logprob_token_ids".to_owned()),
            Some("invalid_logprobs".to_owned()),
        ));
    }
    if let Some(vocab) = vocab_size {
        let invalid: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| *id as usize >= vocab)
            .collect();
        if !invalid.is_empty() {
            let listed = invalid
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ApiError::invalid_request(
                format!(
                    "token_id(s) [{listed}] in logprob_token_ids contain out-of-vocab token ids. Vocabulary size: {vocab}"
                ),
                Some("logprob_token_ids".to_owned()),
                Some("invalid_logprobs".to_owned()),
            ));
        }
    }
    Ok(ids)
}

/// Refuses an explicit `prompt_logprobs` on a streaming request, which vLLM does.
///
/// The check is on the request field only, and only for a value that asks for logprobs at all: vLLM
/// rejects `> 0` and `-1`, while `0` is accepted and dropped (a stream chunk has no
/// `prompt_logprobs` field). An `echo`-implied request is allowed to stream, and the first
/// completion chunk carries the prompt logprobs.
fn reject_streaming_prompt_logprobs(stream: bool, explicit: Option<i32>) -> Result<(), ApiError> {
    if stream
        && let Some(value) = explicit
        && (value > 0 || value == -1)
    {
        return Err(ApiError::invalid_request(
            "`prompt_logprobs` are not available when `stream=True`.",
            Some("prompt_logprobs".to_owned()),
            Some("unsupported_parameter".to_owned()),
        ));
    }
    Ok(())
}

/// One position of an OpenAI completion's parallel logprob arrays.
struct CompletionPosition<'a> {
    /// The decoded token string.
    token: &'a str,
    /// The logprob entry, or `None` for prompt position 0, which has no conditioning token.
    logprob: Option<&'a TokenLogprob>,
}

/// The prompt positions of an `echo` completion: every prompt token, with `None` at position 0.
fn prompt_positions(prompt: &PromptLogprobs) -> Vec<CompletionPosition<'_>> {
    prompt
        .tokens
        .iter()
        .enumerate()
        .map(|(index, token)| CompletionPosition {
            token: &token.token,
            logprob: prompt.positions.get(index).and_then(Option::as_ref),
        })
        .collect()
}

/// The generated positions of a completion.
fn generated_positions(output: &EngineOutput) -> Vec<CompletionPosition<'_>> {
    output
        .token_logprobs
        .iter()
        .map(|logprob| CompletionPosition {
            token: &logprob.token,
            logprob: Some(logprob),
        })
        .collect()
}

/// Builds the OpenAI completion `logprobs` object for a position list.
///
/// vLLM's `top_logprobs` entry is the position's whole map, in vLLM's insertion order: the
/// sampled/actual token first, then the selected alternatives. Each value is the clamped logprob.
fn completion_logprobs_value(positions: &[CompletionPosition<'_>], text_base: usize) -> Value {
    let mut text_offset = Vec::with_capacity(positions.len());
    let mut token_logprobs = Vec::with_capacity(positions.len());
    let mut tokens = Vec::with_capacity(positions.len());
    let mut top_logprobs = Vec::with_capacity(positions.len());
    let mut offset = text_base;
    for position in positions {
        text_offset.push(json!(offset));
        offset += position.token.chars().count();
        tokens.push(Value::String(position.token.to_owned()));
        match position.logprob {
            None => {
                token_logprobs.push(Value::Null);
                top_logprobs.push(Value::Null);
            }
            Some(logprob) => {
                token_logprobs.push(json!(clamped_logprob(logprob.logprob)));
                let mut map = Map::new();
                for alternative in &logprob.top {
                    map.insert(
                        alternative.token.clone(),
                        json!(clamped_logprob(alternative.logprob)),
                    );
                }
                top_logprobs.push(Value::Object(map));
            }
        }
    }
    json!({
        "text_offset": text_offset,
        "token_logprobs": token_logprobs,
        "tokens": tokens,
        "top_logprobs": top_logprobs,
    })
}

/// Builds the id-keyed `prompt_logprobs` array: one entry per prompt token, `null` at position 0.
///
/// vLLM's `Logprob` for prompt positions is `{logprob, rank, decoded_token}`, keyed by token id.
fn prompt_logprobs_value(prompt: &PromptLogprobs) -> Value {
    let mut entries = Vec::with_capacity(prompt.positions.len());
    for position in &prompt.positions {
        let Some(logprob) = position else {
            entries.push(Value::Null);
            continue;
        };
        let mut map = Map::new();
        for alternative in &logprob.top {
            let mut entry = Map::new();
            entry.insert(
                "logprob".to_owned(),
                json!(clamped_logprob(alternative.logprob)),
            );
            entry.insert(
                "rank".to_owned(),
                alternative.rank.map_or(Value::Null, |rank| json!(rank)),
            );
            entry.insert(
                "decoded_token".to_owned(),
                Value::String(alternative.token.clone()),
            );
            map.insert(alternative.token_id.to_string(), Value::Object(entry));
        }
        entries.push(Value::Object(map));
    }
    Value::Array(entries)
}

/// The alternatives a chat logprobs entry lists.
///
/// vLLM's chat `top_logprobs` is the first `n` entries of the position's map (whose first entry is
/// always the sampled token), the whole map for `-1` or `logprob_token_ids`, and empty for `0`.
fn chat_top_entries(
    logprob: &TokenLogprob,
    info: LogprobInfo,
) -> &[crate::engine::LogprobAlternative] {
    if info.explicit_token_ids {
        return &logprob.top;
    }
    match info.sample {
        None => &[],
        Some(LogprobSelection::All) => &logprob.top,
        Some(LogprobSelection::Top(count)) => {
            let count = (count as usize).min(logprob.top.len());
            &logprob.top[..count]
        }
    }
}

/// One entry of the chat `logprobs.content` array.
fn chat_logprobs_entry(logprob: &TokenLogprob, info: LogprobInfo) -> Value {
    let alternatives: Vec<Value> = chat_top_entries(logprob, info)
        .iter()
        .map(|alternative| {
            json!({
                "token": alternative.token,
                "logprob": clamped_logprob(alternative.logprob),
                "bytes": utf8_bytes(&alternative.token),
            })
        })
        .collect();
    json!({
        "token": logprob.token,
        "logprob": clamped_logprob(logprob.logprob),
        "bytes": utf8_bytes(&logprob.token),
        "top_logprobs": alternatives,
    })
}

/// Builds the OpenAI chat `logprobs` object: `{content: [{token, logprob, bytes, top_logprobs}]}`.
fn chat_logprobs_value(output: &EngineOutput, info: LogprobInfo) -> Value {
    let content: Vec<Value> = output
        .token_logprobs
        .iter()
        .map(|logprob| chat_logprobs_entry(logprob, info))
        .collect();
    json!({ "content": content })
}

// ---------------------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------------------

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let value = parse_json(&body)?;
    let spec = resolve_reasoning(&value)?;
    let image_carrier = find_image_carrier(&value);
    let request: ChatCompletionRequest = serde_json::from_value(value).map_err(invalid_body)?;
    if request.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "\"messages\" must be a non-empty array",
            Some("messages".to_owned()),
            Some("invalid_messages".to_owned()),
        ));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid_request(
            "only \"n\": 1 is supported",
            Some("n".to_owned()),
            Some("unsupported_parameter".to_owned()),
        ));
    }
    // Logprobs are resolved and validated before generation, exactly where vLLM validates them:
    // an impossible request (too many alternatives, out-of-vocabulary ids, streaming prompt
    // logprobs) is a 400, never a truncated answer.
    let vocab_size = state.engine.vocab_size();
    let logprobs_requested = request.logprobs.unwrap_or(false);
    let explicit_token_ids = request
        .logprob_token_ids
        .as_ref()
        .is_some_and(|ids| !ids.is_empty());
    // vLLM's chat protocol drops `top_logprobs` from the sampling parameters as soon as
    // `logprob_token_ids` is set, so the explicit list takes precedence and `logprobs` only has to
    // be true. Without ids, `logprobs: true` alone asks for the sampled token (vLLM's
    // `logprobs: 0`), which is why the resolved selection defaults to `Top(0)`.
    let sample = if explicit_token_ids {
        if !logprobs_requested {
            return Err(ApiError::invalid_request(
                "when using `logprob_token_ids`, `logprobs` must be set to true.",
                Some("logprob_token_ids".to_owned()),
                Some("invalid_logprobs".to_owned()),
            ));
        }
        if let Some(top_logprobs) = request.top_logprobs
            && top_logprobs < -1
        {
            return Err(ApiError::invalid_request(
                "`top_logprobs` must be a positive value or -1.",
                Some("top_logprobs".to_owned()),
                Some("invalid_logprobs".to_owned()),
            ));
        }
        Some(LogprobSelection::Top(0))
    } else if logprobs_requested {
        Some(
            resolve_logprob_selection(
                request.top_logprobs,
                "top_logprobs",
                "sample",
                state.max_logprobs(),
                vocab_size,
            )?
            .unwrap_or(LogprobSelection::Top(0)),
        )
    } else {
        if let Some(top_logprobs) = request.top_logprobs
            && top_logprobs != 0
        {
            return Err(ApiError::invalid_request(
                "when using `top_logprobs`, `logprobs` must be set to true.",
                Some("top_logprobs".to_owned()),
                Some("invalid_logprobs".to_owned()),
            ));
        }
        None
    };
    // vLLM's chat `echo` implication is deliberately not implemented: this router does not render
    // a chat template into the prompt yet (see prompt.rs), so "the final message" is not a thing
    // it could echo. Explicit prompt_logprobs are supported, and are refused for streams below.
    let prompt_logprobs = resolve_logprob_selection(
        request.prompt_logprobs,
        "prompt_logprobs",
        "prompt",
        state.max_logprobs(),
        vocab_size,
    )?;
    reject_streaming_prompt_logprobs(request.stream, request.prompt_logprobs)?;
    let token_ids = resolve_logprob_token_ids(request.logprob_token_ids, sample, vocab_size)?;
    let logprobs = LogprobsRequest {
        sample,
        prompt: prompt_logprobs,
        token_ids,
    };
    let logprob_info = LogprobInfo {
        sample,
        explicit_token_ids,
    };
    let model = resolve_model(&state, request.model.as_deref())?;
    let trace = trace_context(&headers)?;
    let summary = summarize_chat_messages(&request.messages)?;
    refuse_image_carrier(image_carrier)?;
    let id = state.next_id("chatcmpl-dsv41-");
    let created = state.now_unix();
    let trace_id = trace_request_id(&headers, &id);
    let engine_request = RequestBuilder {
        id: id.clone(),
        model: model.clone(),
        summary,
        max_tokens: resolve_max_tokens(
            request.max_completion_tokens,
            "max_completion_tokens",
            request.max_tokens,
            "max_tokens",
        )?,
        temperature: request.temperature,
        top_p: request.top_p,
        spec,
        stop: normalize_stop(request.stop.as_ref()),
        seed: request.seed,
        logprobs,
        trace,
    }
    .build();
    // The capability boundary: an engine that cannot serve a channel the request needs refuses
    // here, with a typed error naming it, before any token is generated or streamed.
    state
        .engine
        .validate_request(&engine_request)
        .map_err(ApiError::from)?;

    if request.stream {
        let include_usage = request
            .stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false);
        Ok(chat_stream_response(
            &state,
            engine_request,
            StreamContext {
                id,
                created,
                model,
                include_usage,
                trace_id,
            },
            logprob_info,
            trace,
        ))
    } else {
        let _binding = RequestTraceBinding::new(trace_id);
        let output = drive_engine(state.engine.as_ref(), engine_request).await?;
        state
            .metrics
            .record_request(&output.usage, &output.stats.kv_cache);
        Ok(json_response(
            chat_completion_body(&output, &id, created, &model, logprob_info),
            trace,
        ))
    }
}

fn chat_completion_body(
    output: &EngineOutput,
    id: &str,
    created: u64,
    model: &str,
    logprob_info: LogprobInfo,
) -> Value {
    let mut message = Map::new();
    message.insert("role".to_owned(), Value::String("assistant".to_owned()));
    if output.content.is_empty() && !output.tool_calls.is_empty() {
        message.insert("content".to_owned(), Value::Null);
    } else {
        message.insert("content".to_owned(), Value::String(output.content.clone()));
    }
    if !output.reasoning.is_empty() {
        message.insert(
            "reasoning_content".to_owned(),
            Value::String(output.reasoning.clone()),
        );
    }
    if !output.tool_calls.is_empty() {
        let calls: Vec<Value> = output
            .tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    "index": index,
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect();
        message.insert("tool_calls".to_owned(), Value::Array(calls));
    }
    // `logprobs` is always present (null when not requested, like the OpenAI schema and vLLM);
    // `prompt_logprobs` is vLLM's extension and is only emitted when the request asked for it.
    let logprobs = if logprob_info.wants_sample() {
        chat_logprobs_value(output, logprob_info)
    } else {
        Value::Null
    };
    let mut choice = Map::new();
    choice.insert("index".to_owned(), json!(0));
    choice.insert("message".to_owned(), Value::Object(message));
    choice.insert("logprobs".to_owned(), logprobs);
    choice.insert(
        "finish_reason".to_owned(),
        serde_json::to_value(output.finish_reason).unwrap_or(Value::Null),
    );
    let mut body = Map::new();
    body.insert("id".to_owned(), Value::String(id.to_owned()));
    body.insert(
        "object".to_owned(),
        Value::String("chat.completion".to_owned()),
    );
    body.insert("created".to_owned(), json!(created));
    body.insert("model".to_owned(), Value::String(model.to_owned()));
    body.insert(
        "choices".to_owned(),
        Value::Array(vec![Value::Object(choice)]),
    );
    if let Some(prompt_logprobs) = output.prompt_logprobs.as_ref() {
        body.insert(
            "prompt_logprobs".to_owned(),
            prompt_logprobs_value(prompt_logprobs),
        );
    }
    body.insert(
        "usage".to_owned(),
        serde_json::to_value(output.usage).unwrap_or(Value::Null),
    );
    body.insert(
        "x_deepseek_spark_stats".to_owned(),
        serde_json::to_value(output.stats.as_ref()).unwrap_or(Value::Null),
    );
    Value::Object(body)
}

fn chat_chunk(
    id: &str,
    model: &str,
    created: u64,
    delta: Value,
    finish_reason: Option<FinishReason>,
    logprobs: Option<Value>,
    stats: Option<&DeepSeekSparkRequestStats>,
) -> Value {
    let mut choice = json!({"index": 0, "delta": delta, "finish_reason": finish_reason});
    if let (Some(logprobs), Some(map)) = (logprobs, choice.as_object_mut()) {
        map.insert("logprobs".to_owned(), logprobs);
    }
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    });
    if let Some(stats) = stats
        && let Some(map) = chunk.as_object_mut()
    {
        map.insert(
            "x_deepseek_spark_stats".to_owned(),
            serde_json::to_value(stats).unwrap_or(Value::Null),
        );
    }
    chunk
}

fn chat_usage_chunk(id: &str, model: &str, created: u64, usage: &OpenAiUsageWithCache) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
    })
}

fn tool_call_delta(
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
) -> Value {
    let mut call = Map::new();
    call.insert("index".to_owned(), json!(index));
    if let Some(id) = id {
        call.insert("id".to_owned(), Value::String(id));
        call.insert("type".to_owned(), Value::String("function".to_owned()));
    }
    let mut function = Map::new();
    if let Some(name) = name {
        function.insert("name".to_owned(), Value::String(name));
    }
    function.insert("arguments".to_owned(), Value::String(arguments));
    call.insert("function".to_owned(), Value::Object(function));
    json!({"tool_calls": [Value::Object(call)]})
}

fn chat_stream_response(
    state: &AppState,
    request: EngineRequest,
    context: StreamContext,
    logprob_info: LogprobInfo,
    trace: Option<TraceDirective>,
) -> Response {
    let engine = Arc::clone(&state.engine);
    let metrics = Arc::clone(&state.metrics);
    let StreamContext {
        id,
        created,
        model,
        include_usage,
        trace_id,
    } = context;
    let stream = async_stream::stream! {
        // The spans are emitted while this generator is polled, so the binding has to be
        // taken here rather than on the handler's stack.
        let _binding = RequestTraceBinding::new(trace_id);
        yield Ok::<Bytes, Infallible>(sse_frame(&chat_chunk(
            &id,
            &model,
            created,
            json!({"role": "assistant"}),
            None,
            None,
            None,
        )));
        // When logprobs are requested the text has to travel with the logprob entry of the token
        // it belongs to, so content deltas are held here until their `TokenLogprobs` arrives.
        let mut buffered = String::new();
        let mut events = engine.generate(request);
        while let Some(event) = events.next().await {
            match event {
                EngineEvent::Started { .. } => {}
                // Explicit prompt_logprobs are refused for streams, so this cannot arrive here.
                EngineEvent::PromptLogprobs(_) => {}
                EngineEvent::ReasoningDelta(text) => {
                    yield Ok(sse_frame(&chat_chunk(
                        &id,
                        &model,
                        created,
                        json!({"reasoning_content": text}),
                        None,
                        None,
                        None,
                    )));
                }
                EngineEvent::ContentDelta(text) => {
                    if logprob_info.wants_sample() {
                        buffered.push_str(&text);
                    } else {
                        yield Ok(sse_frame(&chat_chunk(
                            &id,
                            &model,
                            created,
                            json!({"content": text}),
                            None,
                            None,
                            None,
                        )));
                    }
                }
                EngineEvent::TokenLogprobs(logprob) => {
                    if logprob_info.wants_sample() {
                        let text = std::mem::take(&mut buffered);
                        let logprobs =
                            json!({"content": [chat_logprobs_entry(&logprob, logprob_info)]});
                        yield Ok(sse_frame(&chat_chunk(
                            &id,
                            &model,
                            created,
                            json!({"content": text}),
                            None,
                            Some(logprobs),
                            None,
                        )));
                    }
                }
                EngineEvent::ToolCallDelta { index, id: call_id, name, arguments_delta } => {
                    yield Ok(sse_frame(&chat_chunk(
                        &id,
                        &model,
                        created,
                        tool_call_delta(index, call_id, name, arguments_delta),
                        None,
                        None,
                        None,
                    )));
                }
                EngineEvent::Finished { finish_reason, usage, stats } => {
                    metrics.record_request(&usage, &stats.kv_cache);
                    // An engine that was asked for logprobs but emitted none for its last token
                    // has its text flushed here, without a logprob entry rather than with a
                    // fabricated one.
                    if logprob_info.wants_sample() && !buffered.is_empty() {
                        let text = std::mem::take(&mut buffered);
                        yield Ok(sse_frame(&chat_chunk(
                            &id,
                            &model,
                            created,
                            json!({"content": text}),
                            None,
                            None,
                            None,
                        )));
                    }
                    yield Ok(sse_frame(&chat_chunk(
                        &id,
                        &model,
                        created,
                        json!({}),
                        Some(finish_reason),
                        None,
                        Some(stats.as_ref()),
                    )));
                    if include_usage {
                        yield Ok(sse_frame(&chat_usage_chunk(&id, &model, created, &usage)));
                    }
                }
                EngineEvent::Error { message, code } => {
                    yield Ok(sse_frame(&json!({
                        "error": {"message": message, "type": "server_error", "param": null, "code": code},
                    })));
                }
            }
        }
        yield Ok(sse_done());
    };
    sse_response(stream, trace)
}

// ---------------------------------------------------------------------------------------
// POST /v1/completions
// ---------------------------------------------------------------------------------------

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let value = parse_json(&body)?;
    let spec = resolve_reasoning(&value)?;
    let image_carrier = find_image_carrier(&value);
    let request: CompletionRequest = serde_json::from_value(value).map_err(invalid_body)?;
    let model = resolve_model(&state, request.model.as_deref())?;
    let trace = trace_context(&headers)?;
    let prompt_text = match request.prompt.as_ref() {
        Some(PromptSpec::Text(text)) => text.clone(),
        Some(PromptSpec::Many(items)) => items.join("\n"),
        None => {
            return Err(ApiError::invalid_request(
                "\"prompt\" is required",
                Some("prompt".to_owned()),
                Some("invalid_prompt".to_owned()),
            ));
        }
    };
    let summary = summarize_prompt_text(&prompt_text);
    refuse_image_carrier(image_carrier)?;
    let id = state.next_id("cmpl-dsv41-");
    let created = state.now_unix();
    let trace_id = trace_request_id(&headers, &id);
    // vLLM's echo rules: `echo` with no explicit `prompt_logprobs` computes prompt logprobs with
    // `logprobs`'s value, and `echo` with `max_tokens == 0` is the echo-without-generation
    // request (text is the prompt, no tokens are generated, finish_reason is "length").
    let echo = request.echo.unwrap_or(false);
    let max_tokens = if echo && request.max_tokens == Some(0) {
        0
    } else {
        resolve_max_tokens(None, "max_tokens", request.max_tokens, "max_tokens")?
    };
    let vocab_size = state.engine.vocab_size();
    let explicit_token_ids = request
        .logprob_token_ids
        .as_ref()
        .is_some_and(|ids| !ids.is_empty());
    // vLLM's completion protocol drops `logprobs` from the sampling parameters as soon as
    // `logprob_token_ids` is set, so the explicit list takes precedence: the number only has to be
    // present, and is neither capped nor compared against the list length.
    let sample = if explicit_token_ids {
        let Some(value) = request.logprobs else {
            return Err(ApiError::invalid_request(
                "when using `logprob_token_ids`, `logprobs` must be set.",
                Some("logprob_token_ids".to_owned()),
                Some("invalid_logprobs".to_owned()),
            ));
        };
        if value < -1 {
            return Err(ApiError::invalid_request(
                "`logprobs` must be a positive value or -1.",
                Some("logprobs".to_owned()),
                Some("invalid_logprobs".to_owned()),
            ));
        }
        Some(LogprobSelection::Top(0))
    } else {
        resolve_logprob_selection(
            request.logprobs,
            "logprobs",
            "sample",
            state.max_logprobs(),
            vocab_size,
        )?
    };
    let mut prompt_logprobs_raw = request.prompt_logprobs;
    if prompt_logprobs_raw.is_none() && echo {
        prompt_logprobs_raw = request.logprobs;
    }
    let prompt_logprobs = resolve_logprob_selection(
        prompt_logprobs_raw,
        "prompt_logprobs",
        "prompt",
        state.max_logprobs(),
        vocab_size,
    )?;
    reject_streaming_prompt_logprobs(request.stream, request.prompt_logprobs)?;
    let token_ids = resolve_logprob_token_ids(request.logprob_token_ids, sample, vocab_size)?;
    let logprobs = LogprobsRequest {
        sample,
        prompt: prompt_logprobs,
        token_ids,
    };
    let logprob_info = LogprobInfo {
        sample,
        explicit_token_ids,
    };
    let engine_request = RequestBuilder {
        id: id.clone(),
        model: model.clone(),
        summary,
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        spec,
        stop: normalize_stop(request.stop.as_ref()),
        seed: request.seed,
        logprobs,
        trace,
    }
    .build();
    state
        .engine
        .validate_request(&engine_request)
        .map_err(ApiError::from)?;

    if request.stream {
        let include_usage = request
            .stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false);
        Ok(completions_stream_response(
            &state,
            engine_request,
            StreamContext {
                id,
                created,
                model,
                include_usage,
                trace_id,
            },
            logprob_info,
            echo,
            prompt_text,
            trace,
        ))
    } else {
        let _binding = RequestTraceBinding::new(trace_id);
        let output = drive_engine(state.engine.as_ref(), engine_request).await?;
        state
            .metrics
            .record_request(&output.usage, &output.stats.kv_cache);
        let body = completion_body(
            &output,
            &id,
            created,
            &model,
            logprob_info,
            echo,
            &prompt_text,
        );
        Ok(json_response(body, trace))
    }
}

/// Builds the OpenAI /v1/completions body, applying vLLM's echo and logprob rules.
///
/// With `echo`, `text` is the prompt text plus the generated text and the logprob arrays cover the
/// prompt tokens first; `prompt_logprobs` is always present (null unless computed), because that is
/// the field a vLLM client checks.
fn completion_body(
    output: &EngineOutput,
    id: &str,
    created: u64,
    model: &str,
    logprob_info: LogprobInfo,
    echo: bool,
    prompt_text: &str,
) -> Value {
    let text = if echo {
        format!("{prompt_text}{}", output.content)
    } else {
        output.content.clone()
    };
    let logprobs = if logprob_info.wants_sample() {
        let mut positions = Vec::new();
        if echo && let Some(prompt) = output.prompt_logprobs.as_ref() {
            positions.extend(prompt_positions(prompt));
        }
        positions.extend(generated_positions(output));
        completion_logprobs_value(&positions, 0)
    } else {
        Value::Null
    };
    let prompt_logprobs = output
        .prompt_logprobs
        .as_ref()
        .map_or(Value::Null, prompt_logprobs_value);
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": logprobs,
            "finish_reason": output.finish_reason,
            "prompt_logprobs": prompt_logprobs,
        }],
        "usage": output.usage,
        "x_deepseek_spark_stats": output.stats,
    })
}

fn completion_chunk(
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    finish_reason: Option<FinishReason>,
    logprobs: Option<Value>,
) -> Value {
    let mut choice = json!({"index": 0, "text": text, "finish_reason": finish_reason});
    if let (Some(logprobs), Some(map)) = (logprobs, choice.as_object_mut()) {
        map.insert("logprobs".to_owned(), logprobs);
    } else if let Some(map) = choice.as_object_mut() {
        map.insert("logprobs".to_owned(), Value::Null);
    }
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [choice],
    })
}

fn completions_stream_response(
    state: &AppState,
    request: EngineRequest,
    context: StreamContext,
    logprob_info: LogprobInfo,
    echo: bool,
    prompt_text: String,
    trace: Option<TraceDirective>,
) -> Response {
    let engine = Arc::clone(&state.engine);
    let metrics = Arc::clone(&state.metrics);
    let StreamContext {
        id,
        created,
        model,
        include_usage,
        trace_id,
    } = context;
    let stream = async_stream::stream! {
        // The spans are emitted while this generator is polled, so the binding has to be
        // taken here rather than on the handler's stack.
        let _binding = RequestTraceBinding::new(trace_id);
        // With logprobs requested the chunk has to carry the text the logprob belongs to, so
        // content deltas are held until their `TokenLogprobs` event arrives.
        let mut buffered = String::new();
        let mut prompt: Option<PromptLogprobs> = None;
        // vLLM starts a streamed chunk's `text_offset` at the length of the text it has already
        // sent, not at zero for every chunk.
        let mut emitted_chars = 0usize;
        let mut echo_sent = false;
        let mut events = engine.generate(request);
        while let Some(event) = events.next().await {
            match event {
                EngineEvent::Started { .. } => {}
                EngineEvent::PromptLogprobs(payload) => prompt = Some(*payload),
                EngineEvent::ReasoningDelta(_) => {}
                EngineEvent::ContentDelta(text) => {
                    if logprob_info.wants_sample() {
                        buffered.push_str(&text);
                    } else {
                        emitted_chars += text.chars().count();
                        yield Ok(sse_frame(&completion_chunk(&id, &model, created, &text, None, None)));
                    }
                }
                EngineEvent::TokenLogprobs(logprob) => {
                    if logprob_info.wants_sample() {
                        let mut positions = Vec::new();
                        let text = if echo && !echo_sent {
                            echo_sent = true;
                            if let Some(prompt) = prompt.as_ref() {
                                positions.extend(prompt_positions(prompt));
                            }
                            format!("{prompt_text}{}", std::mem::take(&mut buffered))
                        } else {
                            std::mem::take(&mut buffered)
                        };
                        positions.push(CompletionPosition {
                            token: &logprob.token,
                            logprob: Some(&logprob),
                        });
                        let logprobs = completion_logprobs_value(&positions, emitted_chars);
                        emitted_chars += text.chars().count();
                        yield Ok(sse_frame(&completion_chunk(
                            &id,
                            &model,
                            created,
                            &text,
                            None,
                            Some(logprobs),
                        )));
                    }
                }
                EngineEvent::ToolCallDelta { .. } => {}
                EngineEvent::Finished { finish_reason, usage, stats } => {
                    metrics.record_request(&usage, &stats.kv_cache);
                    // The last text a logprobs request has (a token the engine did not logprob,
                    // or the echo prompt of an echo-without-generation stream) goes out once here.
                    if logprob_info.wants_sample() {
                        let mut positions = Vec::new();
                        let mut text = std::mem::take(&mut buffered);
                        if echo && !echo_sent {
                            echo_sent = true;
                            if let Some(prompt) = prompt.as_ref() {
                                positions.extend(prompt_positions(prompt));
                            }
                            text = format!("{prompt_text}{text}");
                        }
                        if !text.is_empty() || !positions.is_empty() {
                            let logprobs = if positions.is_empty() {
                                None
                            } else {
                                Some(completion_logprobs_value(&positions, emitted_chars))
                            };
                            yield Ok(sse_frame(&completion_chunk(
                                &id,
                                &model,
                                created,
                                &text,
                                None,
                                logprobs,
                            )));
                        }
                    }
                    yield Ok(sse_frame(&completion_chunk(
                        &id,
                        &model,
                        created,
                        "",
                        Some(finish_reason),
                        None,
                    )));
                    if include_usage {
                        yield Ok(sse_frame(&json!({
                            "id": id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": usage,
                        })));
                    }
                }
                EngineEvent::Error { message, code } => {
                    yield Ok(sse_frame(&json!({
                        "error": {"message": message, "type": "server_error", "param": null, "code": code},
                    })));
                }
            }
        }
        yield Ok(sse_done());
    };
    sse_response(stream, trace)
}

// ---------------------------------------------------------------------------------------
// POST /v1/responses
// ---------------------------------------------------------------------------------------

async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let value = parse_json(&body)?;
    let spec = resolve_reasoning(&value)?;
    let image_carrier = find_image_carrier(&value);
    let request: ResponsesRequest = serde_json::from_value(value).map_err(invalid_body)?;
    if request.stream.unwrap_or(false) {
        return Err(ApiError::invalid_request(
            "/v1/responses does not implement SSE yet; send \"stream\": false",
            Some("stream".to_owned()),
            Some("unsupported_parameter".to_owned()),
        ));
    }
    let model = resolve_model(&state, request.model.as_deref())?;
    let trace = trace_context(&headers)?;

    let mut summary = PromptSummary::default();
    if let Some(instructions) = request.instructions.as_deref() {
        summary.absorb(summarize_prompt_text(instructions));
    }
    if let Some(input) = request.input.as_ref() {
        summary.absorb(summarize_responses_input(input)?);
    }
    refuse_image_carrier(image_carrier)?;

    let response_id = state.next_id("resp_dsv41_");
    let message_id = state.next_id("msg_dsv41_");
    let reasoning_id = state.next_id("rs_dsv41_");
    let created = state.now_unix();
    let trace_id = trace_request_id(&headers, &response_id);
    let engine_request = RequestBuilder {
        id: response_id.clone(),
        model: model.clone(),
        summary,
        max_tokens: resolve_max_tokens(
            None,
            "max_output_tokens",
            request.max_output_tokens,
            "max_output_tokens",
        )?,
        temperature: request.temperature,
        top_p: request.top_p,
        spec,
        stop: Vec::new(),
        seed: None,
        logprobs: LogprobsRequest::default(),
        trace,
    }
    .build();
    state
        .engine
        .validate_request(&engine_request)
        .map_err(ApiError::from)?;
    let _binding = RequestTraceBinding::new(trace_id);
    let output = drive_engine(state.engine.as_ref(), engine_request).await?;
    state
        .metrics
        .record_request(&output.usage, &output.stats.kv_cache);

    let mut items = Vec::new();
    if !output.reasoning.is_empty() {
        items.push(json!({
            "type": "reasoning",
            "id": reasoning_id,
            "summary": [{"type": "summary_text", "text": output.reasoning}],
        }));
    }
    items.push(json!({
        "type": "message",
        "id": message_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": output.content, "annotations": []}],
    }));

    let body = json!({
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": items,
        "output_text": output.content,
        "usage": {
            "input_tokens": output.usage.prompt_tokens,
            "input_tokens_details": {"cached_tokens": output.usage.prompt_tokens_details.cached_tokens},
            "output_tokens": output.usage.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": output.usage.completion_tokens_details.reasoning_tokens},
            "total_tokens": output.usage.total_tokens,
        },
        "x_deepseek_spark_stats": output.stats,
    });
    Ok(json_response(body, trace))
}

// ---------------------------------------------------------------------------------------
// POST /v1/messages (Anthropic)
// ---------------------------------------------------------------------------------------

async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let value = parse_json(&body)?;
    let spec = resolve_reasoning(&value)?;
    let image_carrier = find_image_carrier(&value);
    let request: MessagesRequest = serde_json::from_value(value).map_err(invalid_body)?;
    if request.stream.unwrap_or(false) {
        return Err(ApiError::invalid_request(
            "/v1/messages does not implement SSE yet; send \"stream\": false",
            Some("stream".to_owned()),
            Some("unsupported_parameter".to_owned()),
        ));
    }
    if request.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "\"messages\" must be a non-empty array",
            Some("messages".to_owned()),
            Some("invalid_messages".to_owned()),
        ));
    }
    let model = resolve_model(&state, request.model.as_deref())?;
    let trace = trace_context(&headers)?;
    let summary = summarize_anthropic_messages(request.system.as_ref(), &request.messages)?;
    refuse_image_carrier(image_carrier)?;
    let stop = request.stop_sequences.clone().unwrap_or_default();
    let id = state.next_id("msg_dsv41_");
    let trace_id = trace_request_id(&headers, &id);
    let engine_request = RequestBuilder {
        id: id.clone(),
        model: model.clone(),
        summary,
        max_tokens: resolve_max_tokens(None, "max_tokens", request.max_tokens, "max_tokens")?,
        temperature: request.temperature,
        top_p: request.top_p,
        spec,
        stop,
        seed: None,
        logprobs: LogprobsRequest::default(),
        trace,
    }
    .build();
    state
        .engine
        .validate_request(&engine_request)
        .map_err(ApiError::from)?;
    let _binding = RequestTraceBinding::new(trace_id);
    let output = drive_engine(state.engine.as_ref(), engine_request).await?;
    state
        .metrics
        .record_request(&output.usage, &output.stats.kv_cache);

    let mut content = Vec::new();
    if !output.reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": output.reasoning, "signature": ""}));
    }
    content.push(json!({"type": "text", "text": output.content}));
    let body = json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": anthropic_stop_reason(output.finish_reason),
        "stop_sequence": null,
        "usage": {
            "input_tokens": output.usage.prompt_tokens,
            "output_tokens": output.usage.completion_tokens,
            "cache_read_input_tokens": output.usage.prompt_tokens_details.cached_tokens,
            "cache_creation_input_tokens": 0,
        },
        "x_deepseek_spark_stats": output.stats,
    });
    Ok(json_response(body, trace))
}

fn anthropic_stop_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::LoopAbort => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ToolCalls => "tool_use",
    }
}

// ---------------------------------------------------------------------------------------
// GET /v1/models, /v1/cluster/stats, /v1/admin/trace/dump, /metrics
// ---------------------------------------------------------------------------------------

async fn list_models(State(state): State<AppState>) -> Response {
    let created = state.now_unix();
    // Most of the capability line is the engine's own answer, because a server must not advertise a
    // channel its engine cannot serve: `tools`, `thinking` and the reasoning-effort schema come from
    // `Capabilities`, `tool_choice` rides along with tools, and the schema is published only when
    // the engine honours it.
    //
    // `multimodal` and `image_url` are the exception: they are fixed `false` rather than read from
    // a capability field. The image path is 263 `vision.*`/`aligner.*` tensors plus the three image
    // delimiters that the tensor plan does not load (DECISIONS.md D18), so no engine in this
    // workspace can serve an image part. A request carrying one is refused with
    // `unsupported_capability` before generation (D27); advertising `true` would be a claim no
    // engine can honour.
    //
    // This block is the single wiring point for a future vision engine: re-add `vision` to
    // `Capabilities`, derive `multimodal` and `image_url` from it here, and override
    // `InferenceEngine::validate_request` with the image forward - which consumes the
    // `EngineRequest::images` parts already carried through the seam (including the decoded bytes
    // of inline images) and produces the visual-token ids - in the same reviewed change. Until then no code path can print `true`, so the constants
    // cannot drift from the refusal.
    let capabilities = state.engine.capabilities();
    let mut advertised = json!({
        "streaming": true,
        "tools": capabilities.tools,
        "tool_choice": capabilities.tools,
        "multimodal": false,
        "image_url": false,
        "thinking": capabilities.thinking,
        "chat_template_kwargs": if capabilities.thinking { json!(["thinking"]) } else { json!([]) },
        "endpoints": [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1/cluster/stats",
            "/v1/admin/trace",
            "/metrics",
        ],
    });
    if capabilities.reasoning_effort {
        advertised["reasoning_effort"] = json!({
            "type": "integer",
            "min": 1,
            "max": 100,
            "default": 75,
            "names": ["low", "minimal", "medium", "high", "xhigh", "max"],
        });
    }
    let body = json!({
        "object": "list",
        "data": [{
            "id": state.model_id(),
            "object": "model",
            "created": created,
            "owned_by": MODEL_OWNED_BY,
            "context_length": max_seq_len(),
            "max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
            "fabric_mode": state.engine.cluster_stats().fabric_mode,
            "capabilities": advertised,
        }],
    });
    Json(body).into_response()
}

async fn cluster_stats(State(state): State<AppState>) -> Response {
    Json(state.engine.cluster_stats()).into_response()
}

/// Arms the requested scopes and installs (or clears) the single-request trace filter.
///
/// The `request_id` member of the response body is the installed filter: when it is
/// present it restricts subsequent captures to that request id; `null` means the filter is
/// cleared and every armed span is recorded again.
async fn admin_trace(body: Bytes) -> Result<Response, ApiError> {
    let request: TraceAdminRequest =
        serde_json::from_slice(&body).map_err(ApiError::malformed_json)?;
    let mask = request.resolve_mask().map_err(|error| {
        ApiError::invalid_request(
            format!("invalid trace scope list: {error}"),
            Some("scopes".to_owned()),
            Some("invalid_trace_scopes".to_owned()),
        )
    })?;
    let header = gpu_trace_ring().header();
    let effective = arm_with_gpu(header, mask, request.steps);
    // Passing None clears any filter left by an earlier capture, otherwise that capture
    // would silently swallow every later trace.
    let _ = set_request_filter(request.request_id.clone());
    let snapshot = TraceAdminResponse::snapshot(Some(header));
    let mut body = serde_json::to_value(snapshot).unwrap_or_else(|_| json!({}));
    if let Some(map) = body.as_object_mut() {
        map.insert("active_mask".to_owned(), json!(effective));
        // The installed filter: present means "only this request id is captured".
        map.insert(
            "request_id".to_owned(),
            request
                .request_id
                .clone()
                .map_or(Value::Null, Value::String),
        );
    }
    Ok(Json(body).into_response())
}

async fn trace_dump() -> Result<Response, ApiError> {
    let events = drain_cpu_events();
    let summaries = summarize(&events);
    let ring = gpu_trace_ring();
    // SAFETY: the ring is an owned host allocation in this process, so no kernel can be
    // writing slots while the host reads them; M1 has no GPU producer at all. If that
    // changes, this read must be sequenced after the graph's completion event.
    let gpu_events = unsafe { ring.collect() };
    let json_text = to_chrome_trace_json(&events, &gpu_events, TraceExportOptions::default());
    let mut document: Value =
        serde_json::from_str(&json_text).unwrap_or_else(|_| json!({"traceEvents": []}));
    if let Some(metadata) = document.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.insert(
            "scope_summaries".to_owned(),
            serde_json::to_value(&summaries).unwrap_or(Value::Null),
        );
        metadata.insert(
            "gpu_ring".to_owned(),
            json!({
                "capacity": ring.capacity(),
                "events_included": gpu_events.len(),
                "mapped": !ring.is_owned(),
                "note": "owned host allocation; a real GPU producer must map it with \
                    cudaHostAllocMapped and adopt it via GpuTraceRing::from_mapped_ptr",
            }),
        );
    }
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder
        .write_all(document.to_string().as_bytes())
        .map_err(|error| ApiError::internal(format!("gzip encode failed: {error}")))?;
    let bytes = encoder
        .finish()
        .map_err(|error| ApiError::internal(format!("gzip finish failed: {error}")))?;
    let mut response = Response::new(Body::from(bytes));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/gzip"),
    );
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"trace.json.gz\""),
    );
    Ok(response)
}

async fn metrics(State(state): State<AppState>) -> Response {
    let snapshot = state.engine.cluster_stats();
    state.metrics.observe_cluster(&snapshot, active_mask());
    let text = state.metrics.render();
    let mut response = Response::new(Body::from(text));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

/// Convenience re-export used by tests: the HTTP status an ApiError carries.
#[must_use]
pub fn error_status(error: &ApiError) -> StatusCode {
    error.status()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_trace_binding_binds_and_restores_the_previous_id() {
        let previous = bind_current_request(Some("outer".to_owned()));
        assert_eq!(
            sparkflash_trace::current_request().as_deref(),
            Some("outer")
        );
        {
            let _inner = RequestTraceBinding::new(Some("inner".to_owned()));
            assert_eq!(
                sparkflash_trace::current_request().as_deref(),
                Some("inner")
            );
        }
        assert_eq!(
            sparkflash_trace::current_request().as_deref(),
            Some("outer")
        );
        {
            let _none = RequestTraceBinding::new(None);
            assert_eq!(sparkflash_trace::current_request(), None);
        }
        assert_eq!(
            sparkflash_trace::current_request().as_deref(),
            Some("outer")
        );
        bind_current_request(previous);
        assert_eq!(sparkflash_trace::current_request(), None);
    }

    #[test]
    fn the_client_request_id_header_wins_over_the_generated_id() {
        let mut headers = HeaderMap::new();
        assert_eq!(
            trace_request_id(&headers, "chatcmpl-1").as_deref(),
            Some("chatcmpl-1")
        );
        headers.insert(REQUEST_ID_HEADER, HeaderValue::from_static("req-7"));
        assert_eq!(
            trace_request_id(&headers, "chatcmpl-1").as_deref(),
            Some("req-7")
        );
        headers.insert(REQUEST_ID_HEADER, HeaderValue::from_static("   "));
        assert_eq!(
            trace_request_id(&headers, "chatcmpl-1").as_deref(),
            Some("chatcmpl-1")
        );
    }
}
