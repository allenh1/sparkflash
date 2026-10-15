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
    EngineEvent, EngineRequest, FinishReason, InferenceEngine, MODEL_MAX_OUTPUT_TOKENS,
    MODEL_OWNED_BY, max_seq_len,
};
use crate::error::ApiError;
use crate::prompt::{
    PromptSummary, summarize_anthropic_messages, summarize_chat_messages, summarize_prompt_text,
    summarize_responses_input,
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
    trace: Option<TraceDirective>,
}

impl RequestBuilder {
    fn build(self) -> EngineRequest {
        EngineRequest {
            request_id: self.id,
            model: self.model,
            prompt_tokens: self.summary.token_ids,
            image_tokens: self.summary.image_tokens,
            max_tokens: self.max_tokens,
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            thinking: self.spec.thinking,
            reasoning_effort: self.spec.effort,
            stop: self.stop,
            seed: self.seed,
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
    let mut finished = None;
    while let Some(event) = stream.next().await {
        match event {
            EngineEvent::Started { .. } => {}
            EngineEvent::ReasoningDelta(text) => reasoning.push_str(&text),
            EngineEvent::ContentDelta(text) => content.push_str(&text),
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
        finish_reason,
        usage,
        stats,
    })
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
    let model = resolve_model(&state, request.model.as_deref())?;
    let trace = trace_context(&headers)?;
    let summary = summarize_chat_messages(&request.messages)?;
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
        trace,
    }
    .build();

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
            trace,
        ))
    } else {
        let _binding = RequestTraceBinding::new(trace_id);
        let output = drive_engine(state.engine.as_ref(), engine_request).await?;
        state
            .metrics
            .record_request(&output.usage, &output.stats.kv_cache);
        Ok(json_response(
            chat_completion_body(&output, &id, created, &model),
            trace,
        ))
    }
}

fn chat_completion_body(output: &EngineOutput, id: &str, created: u64, model: &str) -> Value {
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
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": output.finish_reason,
        }],
        "usage": output.usage,
        "x_deepseek_spark_stats": output.stats,
    })
}

fn chat_chunk(
    id: &str,
    model: &str,
    created: u64,
    delta: Value,
    finish_reason: Option<FinishReason>,
    stats: Option<&DeepSeekSparkRequestStats>,
) -> Value {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
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
        )));
        let mut events = engine.generate(request);
        while let Some(event) = events.next().await {
            match event {
                EngineEvent::Started { .. } => {}
                EngineEvent::ReasoningDelta(text) => {
                    yield Ok(sse_frame(&chat_chunk(
                        &id,
                        &model,
                        created,
                        json!({"reasoning_content": text}),
                        None,
                        None,
                    )));
                }
                EngineEvent::ContentDelta(text) => {
                    yield Ok(sse_frame(&chat_chunk(
                        &id,
                        &model,
                        created,
                        json!({"content": text}),
                        None,
                        None,
                    )));
                }
                EngineEvent::ToolCallDelta { index, id: call_id, name, arguments_delta } => {
                    yield Ok(sse_frame(&chat_chunk(
                        &id,
                        &model,
                        created,
                        tool_call_delta(index, call_id, name, arguments_delta),
                        None,
                        None,
                    )));
                }
                EngineEvent::Finished { finish_reason, usage, stats } => {
                    metrics.record_request(&usage, &stats.kv_cache);
                    yield Ok(sse_frame(&chat_chunk(
                        &id,
                        &model,
                        created,
                        json!({}),
                        Some(finish_reason),
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
    let id = state.next_id("cmpl-dsv41-");
    let created = state.now_unix();
    let trace_id = trace_request_id(&headers, &id);
    let engine_request = RequestBuilder {
        id: id.clone(),
        model: model.clone(),
        summary,
        max_tokens: resolve_max_tokens(None, "max_tokens", request.max_tokens, "max_tokens")?,
        temperature: request.temperature,
        top_p: request.top_p,
        spec,
        stop: normalize_stop(request.stop.as_ref()),
        seed: request.seed,
        trace,
    }
    .build();

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
            trace,
        ))
    } else {
        let _binding = RequestTraceBinding::new(trace_id);
        let output = drive_engine(state.engine.as_ref(), engine_request).await?;
        state
            .metrics
            .record_request(&output.usage, &output.stats.kv_cache);
        let body = json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "text": output.content,
                "logprobs": null,
                "finish_reason": output.finish_reason,
            }],
            "usage": output.usage,
            "x_deepseek_spark_stats": output.stats,
        });
        Ok(json_response(body, trace))
    }
}

fn completion_chunk(
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    finish_reason: Option<FinishReason>,
) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish_reason}],
    })
}

fn completions_stream_response(
    state: &AppState,
    request: EngineRequest,
    context: StreamContext,
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
        let mut events = engine.generate(request);
        while let Some(event) = events.next().await {
            match event {
                EngineEvent::Started { .. } => {}
                EngineEvent::ReasoningDelta(_) => {}
                EngineEvent::ContentDelta(text) => {
                    yield Ok(sse_frame(&completion_chunk(&id, &model, created, &text, None)));
                }
                EngineEvent::ToolCallDelta { .. } => {}
                EngineEvent::Finished { finish_reason, usage, stats } => {
                    metrics.record_request(&usage, &stats.kv_cache);
                    yield Ok(sse_frame(&completion_chunk(&id, &model, created, "", Some(finish_reason))));
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
        trace,
    }
    .build();
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
        trace,
    }
    .build();
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
            "capabilities": {
                "streaming": true,
                "tools": true,
                "tool_choice": true,
                "multimodal": true,
                "image_url": true,
                "thinking": true,
                "reasoning_effort": {
                    "type": "integer",
                    "min": 1,
                    "max": 100,
                    "default": 75,
                    "names": ["low", "minimal", "medium", "high", "xhigh", "max"],
                },
                "chat_template_kwargs": ["thinking"],
                "endpoints": [
                    "/v1/chat/completions",
                    "/v1/completions",
                    "/v1/responses",
                    "/v1/messages",
                    "/v1/cluster/stats",
                    "/v1/admin/trace",
                    "/metrics",
                ],
            },
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
