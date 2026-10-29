// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::StreamExt;
use serde_json::{Map, Value, json};
use sparkflash_model::telemetry::PrefixMatchPlan;
use sparkflash_trace::{
    GpuTraceRing, TraceAdminRequest, TraceAdminResponse, TraceExportOptions, TraceScope,
    active_mask, arm_with_gpu, bind_current_request, disarm_with_gpu, drain_cpu_events,
    parse_scopes_with_aliases, set_request_filter, summarize, to_chrome_trace_json,
};

use crate::device_token::{
    DeviceEndpoint, DeviceResponseContext, DeviceTokenClaim, DeviceTokenRequest, DeviceTokenStream,
    device_chat_body, device_chat_frames, device_completions_body, device_completions_frames,
};
use crate::engine::{
    EngineEvent, EngineRequest, FinishReason, InferenceEngine, LogprobSelection, LogprobsRequest,
    MODEL_MAX_OUTPUT_TOKENS, MODEL_OWNED_BY, PromptLogprobs, TokenLogprob,
};
use crate::error::ApiError;
use crate::prompt::{
    PromptSummary, find_image_carrier, summarize_chat_messages, summarize_prompt_text,
};
use crate::reasoning::{self, ReasoningSpec};
use crate::schema::{
    ChatCompletionRequest, ChatMessage, CompletionRequest, MessagesRequest, PromptSpec,
    ResponseFormat, ResponsesRequest, StopSpec,
};
use crate::state::{AppState, CorsConfig};
use crate::telemetry_types::{
    DSparkStepStats, DeepSeekSparkRequestStats, OpenAiUsageWithCache, safe_ratio,
};
use crate::tool_parser;
use crate::vision::{
    ChatImage, VisionImage, fetch_enabled as image_fetch_enabled, place_images, prepare_images,
    refuse_unread_image, take_chat_images,
};
use sparkflash_model::{PromptCodec, PromptError};

/// Decode-side trace request header.
pub const TRACE_HEADER: &str = "x-sparkflash-trace";
/// Response header echoing the armed trace step count.
pub const TRACE_STEPS_HEADER: &str = "x-sparkflash-trace-steps";
/// Response header echoing the armed scope names.
pub const TRACE_SCOPES_HEADER: &str = "x-sparkflash-trace-scopes";
/// Optional client-supplied id that correlates a request with a trace filter.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Default `max_tokens` outside thinking mode (deepseek-recipe 0.2 contract).
pub const DEFAULT_NON_THINKING_MAX_TOKENS: usize = 8 * 1024;
/// Default `max_tokens` in thinking mode, except at `max` effort.
pub const DEFAULT_THINKING_MAX_TOKENS: usize = 64 * 1024;
/// Default `max_tokens` in thinking mode at `max` effort.
pub const DEFAULT_MAX_EFFORT_MAX_TOKENS: usize = 128 * 1024;

/// The default cap on requested logprobs, matching vLLM's `--max-logprobs` default.
///
/// The cap exists because a request for the whole vocabulary multiplies the response size by the
/// 129,280-token vocabulary per position. `sparkflash serve --max-logprobs N` changes it, and
/// `-1` lifts it, exactly as vLLM's flag does.
pub const DEFAULT_MAX_LOGPROBS: usize = 20;

/// vLLM's cap on the length of `logprob_token_ids`.
pub const MAX_LOGPROB_TOKEN_IDS: usize = 128;

/// The largest request body the API reads, in bytes.
///
/// axum's extractors stop at 2 MB by default, which a 1M-token prompt passes several times over
/// (about 4 MB as plain text, 8 MB as token ids, more again once JSON-escaped). vLLM's server has
/// no body cap, so this one is sized for the model's whole context with room to spare, and for the
/// recipe's 64 MiB of images a request sent inline (about 86 MiB once base64-encoded).
pub const MAX_REQUEST_BODY_BYTES: usize = 128 << 20;

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
    /// Whether the client offered tools, which is what turns on the streaming DSML parser.
    tools_offered: bool,
    /// Whether more than one parsed call may be reported (`parallel_tool_calls`, default true).
    parallel_tool_calls: bool,
    /// Whether every chunk carries the running usage block (`stream_options.continuous_usage_stats`).
    continuous_usage_stats: bool,
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
///
/// When [`AppState::with_api_key`] set a key, every route except `/health`, `/readyz` and
/// `/metrics` requires a matching `Authorization: Bearer <key>`. Without a key the middleware
/// returns before touching the request, so the pre-auth behaviour is byte-for-byte unchanged.
pub fn router(state: AppState) -> Router {
    let auth_state = state.clone();
    let router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/v1/models", get(list_models))
        // The model id itself contains a slash (`deepseek-ai/DeepSeek-V4.1-Flash`), so the retrieve
        // route needs the rest-of-path wildcard rather than a single segment.
        .route("/v1/models/{*model}", get(retrieve_model))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/health", get(health))
        .route("/readyz", get(readyz))
        .route("/v1/cluster/stats", get(cluster_stats))
        .route("/v1/admin/trace", post(admin_trace))
        .route("/v1/admin/trace/dump", get(trace_dump))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            auth_state,
            require_bearer_auth,
        ));
    // The CORS layer is added last, so it is the outermost layer: a browser preflight carries no
    // Authorization header, so it has to be answered before the bearer middleware (vLLM's
    // middleware stack answers preflight first too). Without a policy the router is unchanged.
    let router = match state.cors() {
        Some(config) => router.layer(cors_layer(config)),
        None => router,
    };
    router.with_state(state)
}

/// The browser cross-origin layer, mirroring the reference server's permissive default.
///
/// vLLM mounts a CORS middleware that answers `access-control-allow-origin: *`, every method and
/// `max-age: 600`; the allow-headers list is explicit rather than a wildcard because the Fetch
/// standard's `*` wildcard does not cover `Authorization`, which an OpenAI client sends on every
/// request. A configured origin list answers only those origins and adds `Vary: Origin`.
fn cors_layer(config: &CorsConfig) -> tower_http::cors::CorsLayer {
    let layer = tower_http::cors::CorsLayer::new()
        .allow_methods([
            Method::GET,
            Method::HEAD,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            header::ACCEPT,
            header::ORIGIN,
            HeaderName::from_static(REQUEST_ID_HEADER),
            HeaderName::from_static(TRACE_HEADER),
            HeaderName::from_static(TRACE_STEPS_HEADER),
            HeaderName::from_static(TRACE_SCOPES_HEADER),
        ])
        .max_age(Duration::from_secs(config.max_age_secs()));
    match config.allowed_origins() {
        None => layer.allow_origin(tower_http::cors::AllowOrigin::any()),
        Some(origins) => layer.allow_origin(tower_http::cors::AllowOrigin::list(
            origins
                .iter()
                .filter_map(|origin| HeaderValue::from_str(origin).ok())
                .collect::<Vec<_>>(),
        )),
    }
}

/// The three routes reachable without the bearer key.
///
/// `/health` is the liveness probe a supervisor cannot give a secret to, `/readyz` is the
/// readiness probe (drain-aware: it turns 503 as soon as the drain signal is received), and
/// `/metrics` is the local scrape endpoint; vLLM's health route is exempt too. Everything else,
/// including `/v1/models` and the admin routes, is behind the key.
fn is_public_route(path: &str) -> bool {
    path == "/health" || path == "/readyz" || path == "/metrics"
}

/// Rejects a missing or wrong bearer key with the OpenAI 401 envelope.
///
/// The comparison is constant-time so a caller cannot learn the key byte by byte; the scheme is
/// matched case-insensitively because RFC 7235 says it is, while the token itself is exact.
async fn require_bearer_auth(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.api_key() else {
        return next.run(request).await;
    };
    if is_public_route(request.uri().path()) {
        return next.run(request).await;
    }
    let authorized = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| bearer_token_matches(value, expected));
    if authorized {
        next.run(request).await
    } else {
        ApiError::unauthorized().into_response()
    }
}

/// Whether a header value is `Bearer <expected>`, compared without an early exit.
fn bearer_token_matches(header: &str, expected: &str) -> bool {
    let Some((scheme, token)) = header.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("Bearer") {
        return false;
    }
    constant_time_eq(token.as_bytes(), expected.as_bytes())
}

/// Compares two byte strings in time independent of their contents (length still leaks).
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut difference = 0u8;
    for (left, right) in left.iter().zip(right) {
        difference |= left ^ right;
    }
    difference == 0
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
/// This is D27's structural backstop: [find_image_carrier] walks every object and array, so
/// an image spelling no typed parser knows (the documented computer_screenshot output, a
/// file_id-only screenshot, a code_interpreter_call output, a case-variant type) is refused with
/// the same HTTP 400 unsupported_capability as the capability seam, and the message names the
/// shape and JSON path found. It runs before validate_request because the typed seam only sees
/// the parts a parser understood.
///
/// With vision (the engine has the tower loaded) this only fires for a carrier in a shape the
/// route does not read, and the message names the shapes it does read instead of blaming the
/// engine.
fn refuse_image_carrier(carrier: Option<String>, vision: bool) -> Result<(), ApiError> {
    let Some(shape) = carrier else {
        return Ok(());
    };
    if vision {
        return Err(ApiError::invalid_request(
            format!(
                "this route does not read an image in this shape: it reads chat message \
                 image_url/input_image parts, Responses input_image parts and Anthropic image \
                 blocks (DECISIONS.md D59), and the request body carries an image at {shape}; the \
                 request is refused rather than answered from its text tokens alone"
            ),
            None,
            Some("unsupported_capability".to_owned()),
        ));
    }
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

/// Prepares a request's images and expands their placeholders in `summary` (DECISIONS.md D59).
async fn prepare_and_place_images(
    state: &AppState,
    images: Vec<ChatImage>,
    summary: &mut PromptSummary,
) -> Result<Vec<VisionImage>, ApiError> {
    let codec = state
        .tokenizer()
        .and_then(|seam| seam.vision_codec())
        .ok_or_else(|| {
            ApiError::internal(
                "the engine advertises vision but the prompt codec has no vision front end",
            )
        })?;
    let prepared = prepare_images(images, image_fetch_enabled(), Arc::clone(&codec)).await?;
    let (ids, placed) = place_images(&summary.token_ids, prepared, codec.as_ref())?;
    summary.token_ids = ids;
    Ok(placed)
}

/// Refuses an image the protocol bridge does not render.
///
/// The bridge turns only the fields it knows (`input` for Responses, `messages`/`system` for
/// Anthropic) into chat parts, but [find_image_carrier] scans the whole raw body. An image
/// anywhere else would be dropped from the prompt, so it is refused by name instead of answered
/// from the text alone (D27's backstop, at the route's own boundary).
///
/// # Errors
///
/// The same unsupported_capability 400 [refuse_image_carrier] builds.
fn refuse_unbridged_image(body: &Value, rendered: &[&str]) -> Result<(), ApiError> {
    let Some(object) = body.as_object() else {
        return Ok(());
    };
    let mut remainder = object.clone();
    for key in rendered {
        remainder.remove(*key);
    }
    let shape = find_image_carrier(&Value::Object(remainder));
    refuse_image_carrier(shape, true)
}

/// Refuses an image on `/v1/completions`, which takes a prompt rather than messages.
///
/// The route has no image part to read, so two spellings are refused by name: a marker inside
/// the prompt (an `<image>...</image>` tag, a placeholder in any spelling, or the image token in
/// pre-tokenized ids) and an image carrier in the body. Either would reach the model with no
/// canvas behind it, and the request would be answered from its text alone (DECISIONS.md D59);
/// the conversation routes read images.
///
/// The check runs before the device-token fast path: a claimed envelope replays a recorded token
/// window on the ranks, so an image it carried would never be read on the host either.
///
/// # Errors
///
/// The unsupported_capability 400 the other carrier refusals build.
fn refuse_completion_image(
    state: &AppState,
    prompt: Option<&PromptSpec>,
    carrier: Option<String>,
) -> Result<(), ApiError> {
    let image_token = state
        .tokenizer()
        .and_then(|seam| seam.vision_codec())
        .map(|codec| codec.image_token_id());
    let marked = match prompt {
        None => None,
        Some(PromptSpec::Text(text)) => {
            completion_prompt_marker(text, None, image_token).map(|marker| (1, marker))
        }
        Some(PromptSpec::Many(items)) => items.iter().enumerate().find_map(|(index, text)| {
            completion_prompt_marker(text, None, image_token).map(|marker| (index + 1, marker))
        }),
        Some(PromptSpec::Tokens(ids)) => {
            completion_prompt_marker("", Some(ids), image_token).map(|marker| (1, marker))
        }
    };
    let Some(shape) = marked
        .map(|(index, marker)| format!("the {marker} in prompt {index}"))
        .or_else(|| carrier.map(|shape| format!("an image at {shape}")))
    else {
        return Ok(());
    };
    Err(ApiError::invalid_request(
        format!(
            "/v1/completions takes a prompt, not messages, so it reads no image, and {shape} would \
             reach the model with no canvas behind it. Send the image on /v1/chat/completions, \
             /v1/responses or /v1/messages (DECISIONS.md D59), or remove it; this route refuses \
             it rather than answering from the text alone"
        ),
        None,
        Some("unsupported_capability".to_owned()),
    ))
}

/// The image marker in one completion prompt, named for a refusal message.
///
/// A pre-tokenized prompt is checked by id - the placeholder's spelling need not survive a decode
/// - and a text prompt by its spelling.
fn completion_prompt_marker(
    text: &str,
    tokens: Option<&[u32]>,
    image_token: Option<u32>,
) -> Option<&'static str> {
    if tokens.is_some_and(|ids| image_token.is_some_and(|id| ids.contains(&id))) {
        return Some("image token id");
    }
    crate::multimodal::image_marker_in(text)
}

/// The chat-shape image path: take the body's image parts, refuse a carrier left behind, then
/// prepare and place them. The protocol bridges render their images as chat parts, so the
/// Responses and Anthropic routes take this same path.
async fn apply_chat_images(
    state: &AppState,
    chat: &Value,
    summary: &mut PromptSummary,
) -> Result<Vec<VisionImage>, ApiError> {
    let (images, stripped) = take_chat_images(chat)?;
    refuse_unread_image(&stripped)?;
    prepare_and_place_images(state, images, summary).await
}

/// The number of independent choices a request asks for, validated before generation.
///
/// vLLM's sampling validation is `n >= 1`; a shared-sampling engine path (the TP=4 group) cannot
/// promise independent draws, so it is refused typed instead of returning n copies of one sample.
fn resolve_choice_count(n: Option<usize>, engine: &dyn InferenceEngine) -> Result<usize, ApiError> {
    let n = n.unwrap_or(1);
    if n < 1 {
        return Err(ApiError::invalid_request(
            format!("n must be at least 1, got {n}."),
            Some("n".to_owned()),
            Some("invalid_n".to_owned()),
        ));
    }
    if n > 1 && !engine.independent_choices() {
        return Err(ApiError::invalid_request(
            "n > 1 requires an engine that draws each choice independently, and this engine path shares its sampling state across the ranks (the TP=4 host sampler is temperature + seed only); the request is refused rather than returning choices that were not independently sampled",
            Some("n".to_owned()),
            Some("unsupported_parameter".to_owned()),
        ));
    }
    Ok(n)
}

/// The tool policy a chat request asked for, resolved before generation.
struct ToolPolicy {
    /// Whether the tools are rendered into the prompt and a DSML block in the answer is lifted into
    /// `tool_calls`. False when the client sent `tool_choice: "none"`.
    tools_offered: bool,
    /// Whether one response may carry more than one parsed call (`parallel_tool_calls`, default
    /// true).
    parallel: bool,
}

/// Validates OpenAI's `tool_choice`/`parallel_tool_calls` pair.
///
/// `"none"` and `"auto"` are honoured. `"required"` and a named function selector need a template
/// policy that forces the call, which the V4.1 seam does not render, so they are refused with a
/// typed 400 (the same treatment `response_format` gets) rather than accepted as a no-op. The
/// selector is validated even when no tools are offered, because a malformed value is a client
/// error either way.
fn resolve_tool_policy(
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    parallel_tool_calls: Option<bool>,
) -> Result<ToolPolicy, ApiError> {
    let offered = tools.is_some_and(|tools| !tools.is_null());
    let parallel = parallel_tool_calls.unwrap_or(true);
    let forced = |what: &str| {
        ApiError::invalid_request(
            format!(
                "tool_choice {what} needs a prompt policy that forces a tool call, and this server \
                 renders the tool list without one; the request is refused rather than answered \
                 with a call the model was never required to make"
            ),
            Some("tool_choice".to_owned()),
            Some("unsupported_parameter".to_owned()),
        )
    };
    let malformed = || {
        ApiError::invalid_request(
            "tool_choice must be \"none\", \"auto\", \"required\" or a function selector",
            Some("tool_choice".to_owned()),
            Some("invalid_tool_choice".to_owned()),
        )
    };
    let mode = match tool_choice {
        None | Some(Value::Null) => "auto",
        Some(Value::String(choice)) => match choice.as_str() {
            "auto" => "auto",
            "none" => "none",
            "required" => return Err(forced("\"required\"")),
            _ => return Err(malformed()),
        },
        Some(Value::Object(selector)) => {
            let is_function = selector.get("type").and_then(Value::as_str) == Some("function");
            if !is_function {
                return Err(malformed());
            }
            return Err(forced("with a named function"));
        }
        Some(_) => return Err(malformed()),
    };
    Ok(ToolPolicy {
        tools_offered: offered && mode != "none",
        parallel,
    })
}

/// Validates OpenAI's `logit_bias`/`min_p` and refuses the values this engine cannot apply.
///
/// Both need a logits processor inside the sampling path: vLLM applies `logit_bias` to the raw
/// logits before the penalties and `min_p` as a candidate filter after `top_p`. The engine seam this
/// server owns has no such hook, so a request that would change generation is refused typed instead
/// of being silently dropped. The disabled values (`logit_bias: {}` and `min_p: 0.0`) are accepted
/// as the no-ops they are, and the malformed forms get vLLM's own conversion errors first.
fn check_sampling_extensions(
    logit_bias: Option<&Map<String, Value>>,
    min_p: Option<f32>,
    vocab_size: Option<usize>,
) -> Result<(), ApiError> {
    if let Some(bias) = logit_bias.filter(|bias| !bias.is_empty()) {
        let mut bad_keys: Vec<&str> = Vec::new();
        for (key, value) in bias {
            let Ok(id) = key.parse::<u32>() else {
                bad_keys.push(key);
                continue;
            };
            let Some(raw) = value.as_f64() else {
                return Err(ApiError::invalid_request(
                    format!("logit_bias value for token {id} must be a number"),
                    Some("logit_bias".to_owned()),
                    Some("invalid_logit_bias".to_owned()),
                ));
            };
            if !(-100.0..=100.0).contains(&raw) {
                return Err(ApiError::invalid_request(
                    format!(
                        "logit_bias must be a float between -100 and 100, got {raw} for token {id}"
                    ),
                    Some("logit_bias".to_owned()),
                    Some("invalid_logit_bias".to_owned()),
                ));
            }
            if let Some(vocab) = vocab_size
                && id as usize >= vocab
            {
                return Err(ApiError::invalid_request(
                    format!(
                        "logit_bias names token id {id}, outside the model vocabulary ({vocab} ids)"
                    ),
                    Some("logit_bias".to_owned()),
                    Some("invalid_logit_bias".to_owned()),
                ));
            }
        }
        if !bad_keys.is_empty() {
            let listed = bad_keys
                .iter()
                .map(|key| format!("'{key}'"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ApiError::invalid_request(
                format!(
                    "logit_bias contains key(s) that cannot be converted to integer token IDs: [{listed}] (parameter=logit_bias, value=[{listed}])"
                ),
                Some("logit_bias".to_owned()),
                Some("invalid_logit_bias".to_owned()),
            ));
        }
        return Err(ApiError::invalid_request(
            "logit_bias needs a logits processor in the sampling path, and this engine seam has none;              it is refused instead of being silently ignored. The reference deployment refuses it too,              reporting min_p and logit_bias as unsupported for its speculative decoder",
            Some("logit_bias".to_owned()),
            Some("unsupported_parameter".to_owned()),
        ));
    }
    match min_p {
        Some(value) if !(0.0..=1.0).contains(&value) => Err(ApiError::invalid_request(
            format!("min_p must be between 0 and 1, got {value}"),
            Some("min_p".to_owned()),
            Some("invalid_min_p".to_owned()),
        )),
        Some(value) if value > 0.0 => Err(ApiError::invalid_request(
            "min_p needs a candidate filter in the sampling path, and this engine seam has none; it              is refused instead of being silently ignored. The reference deployment refuses it too,              reporting min_p and logit_bias as unsupported for its speculative decoder",
            Some("min_p".to_owned()),
            Some("unsupported_parameter".to_owned()),
        )),
        _ => Ok(()),
    }
}

/// Refuses a structured-output mode this server cannot enforce.
///
/// The engine seam has no decoder logits processor, so vLLM's xgrammar guided-decoding path does
/// not exist here. The released V4.1 template can render a schema into the prompt as an
/// instruction, but an instruction is not a guarantee, so `json_object` and `json_schema` are
/// refused before generation instead of returning text that may not be valid JSON or match the
/// schema. `{"type":"text"}` is the OpenAI default and is accepted as a no-op.
fn refuse_unenforceable_response_format(format: Option<&ResponseFormat>) -> Result<(), ApiError> {
    match format {
        None | Some(ResponseFormat::Text) => Ok(()),
        Some(ResponseFormat::JsonObject) | Some(ResponseFormat::JsonSchema { .. }) => {
            Err(ApiError::invalid_request(
                "response_format json_object/json_schema requires constrained decoding (vLLM's xgrammar path); this server has no decoder logits processor, and rendering the schema into the prompt is only an instruction, so the request is refused instead of returning unconstrained JSON",
                Some("response_format".to_owned()),
                Some("unsupported_parameter".to_owned()),
            ))
        }
    }
}

/// The device-chain scaffold's plain-completion eligibility check.
///
/// The scaffold replays one recorded nine-token greedy window, so it can only serve a request whose
/// OpenAI fields do not change the assembly: any sampling parameter, penalty, seed, stop sequence,
/// logprob, echo, reasoning or response_format field makes the request ineligible, and it continues
/// to the host twin exactly as before. A pre-tokenized prompt is required because a claimed
/// envelope is tied to the token window the device recorded; replaying a text prompt needs the
/// tokenizer seam plus the prompt-match check, which the scaffold does not wire yet.
fn device_completion_eligibility(request: &CompletionRequest) -> Option<DeviceTokenRequest> {
    if request.n.unwrap_or(1) != 1
        || request.echo.unwrap_or(false)
        || request.stop.is_some()
        || request.logprobs.is_some()
        || request.prompt_logprobs.is_some()
        || request.logprob_token_ids.is_some()
        || request.temperature.is_some()
        || request.top_p.is_some()
        || request.top_k.is_some()
        || request.repetition_penalty.is_some()
        || request.presence_penalty.is_some()
        || request.frequency_penalty.is_some()
        || request.skip_special_tokens.is_some()
        || request.seed.is_some()
        || request.reasoning_effort.is_some()
        || request.enable_thinking.is_some()
        || request.chat_template_kwargs.is_some()
    {
        return None;
    }
    let Some(PromptSpec::Tokens(ids)) = request.prompt.as_ref() else {
        return None;
    };
    Some(DeviceTokenRequest {
        endpoint: DeviceEndpoint::Completions,
        stream: request.stream,
        include_usage: request
            .stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false),
        model: None,
        max_tokens: request.max_tokens,
        prompt_token_ids: Some(ids.clone()),
    })
}

/// The device-chain scaffold's plain-chat eligibility check.
///
/// The same plainness rule as the completions check, plus the chat-only fields: tools, echo, a
/// non-text response_format and every sampling/reasoning field fall back to the host twin. An image
/// carrier also falls back because the scaffold has no multimodal device evidence.
fn device_chat_eligibility(
    request: &ChatCompletionRequest,
    image_absent: bool,
    prompt_token_ids: Option<Vec<u32>>,
) -> Option<DeviceTokenRequest> {
    if !image_absent
        || request.n.unwrap_or(1) != 1
        || request.tools.as_ref().is_some_and(|tools| !tools.is_null())
        || request.stop.is_some()
        || request.echo.unwrap_or(false)
        || request.logprobs.unwrap_or(false)
        || request.top_logprobs.is_some()
        || request.prompt_logprobs.is_some()
        || request.logprob_token_ids.is_some()
        || request.temperature.is_some()
        || request.top_p.is_some()
        || request.top_k.is_some()
        || request.repetition_penalty.is_some()
        || request.presence_penalty.is_some()
        || request.frequency_penalty.is_some()
        || request.skip_special_tokens.is_some()
        || request.seed.is_some()
        || request.reasoning_effort.is_some()
        || request.enable_thinking.is_some()
        || request.chat_template_kwargs.is_some()
        || matches!(
            request.response_format.as_ref(),
            Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
        )
    {
        return None;
    }
    Some(DeviceTokenRequest {
        endpoint: DeviceEndpoint::ChatCompletions,
        stream: request.stream,
        include_usage: request
            .stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false),
        model: None,
        max_tokens: request.max_completion_tokens.or(request.max_tokens),
        prompt_token_ids,
    })
}

/// Claims the installed device token source, or returns None so the host twin serves the request.
///
/// The fallback reason is deliberately not logged or surfaced here: the scaffold's contract is that
/// the host path is the answer, and the reason already lives in the typed refusal record that a
/// future live source will log. docs/archive/build/serving-s7-s8-plan.md records that wiring point.
fn claim_device_token(state: &AppState, request: DeviceTokenRequest) -> Option<DeviceTokenStream> {
    let source = state.device_token_source()?;
    match source.claim(&request) {
        DeviceTokenClaim::Claimed(stream) => Some(*stream),
        DeviceTokenClaim::Fallback { reason: _ } => None,
    }
}

/// The response header naming the device-chain envelope provenance for a claimed body.
pub const DEVICE_TOKEN_SOURCE_HEADER: &str = "x-sparkflash-token-source";

/// The header value for the one-shot envelope source; UNMEASURED is part of the provenance, not a
/// decoration.
pub const DEVICE_TOKEN_SOURCE_UNMEASURED: &str = "device-chain-envelope (UNMEASURED)";

/// The value [`device_token_response`] actually sends: the source's own name plus the
/// unmeasured marker, so a live window follower and a static envelope are distinguishable on the
/// wire instead of both claiming to be `device-chain-envelope`.
#[must_use]
pub fn device_token_source_label(source: &str) -> String {
    format!("{source} (UNMEASURED)")
}

/// The header naming how the claim was established: explicit device assertion or derived evidence.
pub const DEVICE_TOKEN_CLAIM_HEADER: &str = "x-sparkflash-token-claim";

/// Serves a claimed device token stream as the OpenAI payload for one endpoint.
///
/// The assembly is the pure shape from crate::device_token: the buffered body carries the claimed
/// text with finish_reason length, and the SSE form carries the same chunks the host path emits
/// (chat role chunk, one content delta per token, finish chunk, optional usage chunk, then [DONE]).
/// The response is labelled with the envelope provenance header; no stats, tok/s or TTFT claim
/// rides along because none was measured.
#[allow(clippy::too_many_arguments)]
fn device_token_response(
    stream: &DeviceTokenStream,
    endpoint: DeviceEndpoint,
    state: &AppState,
    model: &str,
    stream_requested: bool,
    include_usage: bool,
    trace: Option<TraceDirective>,
) -> Response {
    let id = state.next_id(match endpoint {
        DeviceEndpoint::ChatCompletions => "chatcmpl-dsv41-",
        DeviceEndpoint::Completions => "cmpl-dsv41-",
    });
    let created = state.now_unix();
    let context = DeviceResponseContext {
        id,
        created,
        model: model.to_owned(),
        include_usage,
    };
    let mut response = if stream_requested {
        let frames = match endpoint {
            DeviceEndpoint::ChatCompletions => device_chat_frames(stream, &context),
            DeviceEndpoint::Completions => device_completions_frames(stream, &context),
        };
        let events: Vec<Result<Bytes, Infallible>> = frames
            .iter()
            .map(|frame| Ok(sse_frame(frame)))
            .chain(std::iter::once(Ok(sse_done())))
            .collect();
        sse_response(futures::stream::iter(events), trace, state.sse_keep_alive())
    } else {
        let body = match endpoint {
            DeviceEndpoint::ChatCompletions => device_chat_body(stream, &context),
            DeviceEndpoint::Completions => device_completions_body(stream, &context),
        };
        json_response(body, trace)
    };
    let label = device_token_source_label(stream.source);
    response.headers_mut().insert(
        HeaderName::from_static(DEVICE_TOKEN_SOURCE_HEADER),
        HeaderValue::from_str(&label)
            .unwrap_or_else(|_| HeaderValue::from_static(DEVICE_TOKEN_SOURCE_UNMEASURED)),
    );
    response.headers_mut().insert(
        HeaderName::from_static(DEVICE_TOKEN_CLAIM_HEADER),
        HeaderValue::from_static(stream.claim_label()),
    );
    response
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
        Some(requested) if model_matches(requested, engine_id, state.model_aliases()) => {
            Ok(engine_id.to_owned())
        }
        Some(requested) => Err(ApiError::model_not_found(requested)),
    }
}

/// The bare model id a client named, without a protocol suffix.
///
/// Anthropic spells a 1M-context request as `model[1m]`, and Claude Code sends that spelling
/// through to the server; the suffix decorates the id rather than replacing it.
fn without_model_suffix(requested: &str) -> &str {
    match requested.strip_suffix(']') {
        Some(stem) => match stem.rfind('[') {
            Some(open) => stem[..open].trim_end(),
            None => requested,
        },
        None => requested,
    }
}

fn model_matches(requested: &str, engine_id: &str, aliases: &[String]) -> bool {
    let requested = without_model_suffix(requested.trim());
    if requested.eq_ignore_ascii_case(engine_id) {
        return true;
    }
    let requested_leaf = requested.rsplit('/').next().unwrap_or(requested);
    let engine_leaf = engine_id.rsplit('/').next().unwrap_or(engine_id);
    requested_leaf.eq_ignore_ascii_case(engine_leaf)
        || aliases
            .iter()
            .any(|alias| alias.eq_ignore_ascii_case(requested))
}

/// Resolves the requested output-token budget against the V4.1 ceiling.
///
/// This is the compatibility wrapper for tests and callers that have not selected a model. A
/// serving path uses [`resolve_max_tokens_with_ceiling`] with the descriptor's ceiling.
#[allow(dead_code)]
fn resolve_max_tokens(
    primary: Option<usize>,
    primary_name: &'static str,
    secondary: Option<usize>,
    secondary_name: &'static str,
    spec: ReasoningSpec,
    prompt_tokens: usize,
    context_limit: usize,
) -> Result<usize, ApiError> {
    resolve_max_tokens_with_ceiling(
        primary,
        primary_name,
        secondary,
        secondary_name,
        spec,
        prompt_tokens,
        context_limit,
        MODEL_MAX_OUTPUT_TOKENS,
    )
}

/// Resolves the requested output-token budget.
///
/// An absent budget means [`DEFAULT_NON_THINKING_MAX_TOKENS`],
/// [`DEFAULT_THINKING_MAX_TOKENS`] or [`DEFAULT_MAX_EFFORT_MAX_TOKENS`] by mode; `0` and a
/// value above `max_output_tokens` are client errors; an otherwise legal value is lowered to the
/// context left after the prompt, and raised to at least 1.
#[allow(clippy::too_many_arguments)]
fn resolve_max_tokens_with_ceiling(
    primary: Option<usize>,
    primary_name: &'static str,
    secondary: Option<usize>,
    secondary_name: &'static str,
    spec: ReasoningSpec,
    prompt_tokens: usize,
    context_limit: usize,
    max_output_tokens: usize,
) -> Result<usize, ApiError> {
    resolve_max_tokens_impl(
        primary,
        primary_name,
        secondary,
        secondary_name,
        spec,
        prompt_tokens,
        context_limit,
        max_output_tokens,
        true,
    )
}

/// Anthropic Messages lowers an over-cap `max_tokens` to the protocol upper bound instead of
/// rejecting it, while Chat Completions and Responses reject it (recipe 0.2 semantics).
#[allow(clippy::too_many_arguments)]
fn resolve_max_tokens_for_messages_with_ceiling(
    primary: Option<usize>,
    primary_name: &'static str,
    secondary: Option<usize>,
    secondary_name: &'static str,
    spec: ReasoningSpec,
    prompt_tokens: usize,
    context_limit: usize,
    max_output_tokens: usize,
) -> Result<usize, ApiError> {
    resolve_max_tokens_impl(
        primary,
        primary_name,
        secondary,
        secondary_name,
        spec,
        prompt_tokens,
        context_limit,
        max_output_tokens,
        false,
    )
}

/// Anthropic Messages with the V4.1 ceiling: the compatibility wrapper.
#[allow(dead_code)]
fn resolve_max_tokens_for_messages(
    primary: Option<usize>,
    primary_name: &'static str,
    secondary: Option<usize>,
    secondary_name: &'static str,
    spec: ReasoningSpec,
    prompt_tokens: usize,
    context_limit: usize,
) -> Result<usize, ApiError> {
    resolve_max_tokens_for_messages_with_ceiling(
        primary,
        primary_name,
        secondary,
        secondary_name,
        spec,
        prompt_tokens,
        context_limit,
        MODEL_MAX_OUTPUT_TOKENS,
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_max_tokens_impl(
    primary: Option<usize>,
    primary_name: &'static str,
    secondary: Option<usize>,
    secondary_name: &'static str,
    spec: ReasoningSpec,
    prompt_tokens: usize,
    context_limit: usize,
    max_output_tokens: usize,
    reject_above_upper_bound: bool,
) -> Result<usize, ApiError> {
    let (field, mut requested) = match (primary, secondary) {
        (Some(value), _) => (primary_name, Some(value)),
        (None, Some(value)) => (secondary_name, Some(value)),
        (None, None) => (primary_name, None),
    };
    if requested == Some(0) {
        return Err(ApiError::invalid_request(
            format!("{field} must be at least 1"),
            Some(field.to_owned()),
            Some("invalid_max_tokens".to_owned()),
        ));
    }
    if let Some(value) = requested
        && value > max_output_tokens
    {
        if reject_above_upper_bound {
            return Err(ApiError::invalid_request(
                format!("{field} must not exceed {max_output_tokens}"),
                Some(field.to_owned()),
                Some("invalid_max_tokens".to_owned()),
            ));
        }
        requested = Some(max_output_tokens);
    }
    if prompt_tokens > context_limit {
        return Err(ApiError::invalid_request(
            format!(
                "This model's maximum context length is {context_limit} tokens. However, your \
                 prompt is {prompt_tokens} tokens. Please reduce the length of the messages."
            ),
            Some(field.to_owned()),
            Some("context_length_exceeded".to_owned()),
        ));
    }
    let default = if !spec.thinking {
        DEFAULT_NON_THINKING_MAX_TOKENS
    } else if spec.effort == Some(100) {
        DEFAULT_MAX_EFFORT_MAX_TOKENS
    } else {
        DEFAULT_THINKING_MAX_TOKENS
    };
    let remaining = context_limit - prompt_tokens;
    Ok(requested.unwrap_or(default).min(remaining).max(1))
}

/// Normalises the OpenAI `stop` field to a list, with vLLM's validation.
///
/// `stop` is a string or an array of strings; an empty string is refused with vLLM's message
/// (`SamplingParams._verify_args`: "stop cannot contain an empty string.") because a matcher on
/// "" would end every generation before its first token.
fn normalize_stop(stop: Option<&StopSpec>) -> Result<Vec<String>, ApiError> {
    let sequences = match stop {
        None => Vec::new(),
        Some(StopSpec::One(sequence)) => vec![sequence.clone()],
        Some(StopSpec::Many(sequences)) => sequences.clone(),
    };
    if sequences.iter().any(String::is_empty) {
        return Err(ApiError::invalid_request(
            "stop cannot contain an empty string.",
            Some("stop".to_owned()),
            Some("invalid_stop".to_owned()),
        ));
    }
    Ok(sequences)
}

struct RequestBuilder {
    id: String,
    model: String,
    summary: PromptSummary,
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    repetition_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    skip_special_tokens: Option<bool>,
    spec: ReasoningSpec,
    stop: Vec<String>,
    seed: Option<u64>,
    logprobs: LogprobsRequest,
    trace: Option<TraceDirective>,
    /// The prepared images, whose spans are already expanded in `summary.token_ids`.
    vision: Vec<VisionImage>,
}

/// V4.1 thinking-mode sampling policy, adopted from deepseek-recipe 0.2.
///
/// This is a deliberately labelled API-surface/sampling change (task-63), not a
/// prompt change: thinking mode fixes `temperature` at 1.0 and raises `top_p` to at
/// least 0.95; outside thinking mode the caller's values are passed through.
fn resolve_sampling(temperature: Option<f32>, top_p: Option<f32>, thinking: bool) -> (f32, f32) {
    if thinking {
        (1.0, top_p.unwrap_or(1.0).max(0.95))
    } else {
        (temperature.unwrap_or(1.0), top_p.unwrap_or(1.0))
    }
}

impl RequestBuilder {
    /// Builds the seam request. Unprepared image parts are passed through - never turned into a
    /// fabricated visual-token count - so the engine's `validate_request` can refuse them (see
    /// [`crate::engine::MissingCapability`]); prepared images ride in `vision`, their spans already
    /// in the prompt. The handler calls `validate_request` before generation.
    ///
    /// The sampling parameters are validated here, once, for every endpoint that can carry them;
    /// the messages are vLLM's own (with one documented extension: `top_p` 0.0 is accepted as
    /// "keep the single most likely token" where vLLM's range refuses it).
    fn build(self) -> Result<EngineRequest, ApiError> {
        let (temperature, top_p) =
            resolve_sampling(self.temperature, self.top_p, self.spec.thinking);
        Ok(EngineRequest {
            request_id: self.id,
            model: self.model,
            prompt_tokens: self.summary.token_ids,
            images: self.summary.images,
            vision: self.vision,
            max_tokens: self.max_tokens,
            temperature,
            top_p: validate_top_p(top_p)?,
            top_k: validate_top_k(self.top_k.unwrap_or(0))?,
            repetition_penalty: validate_repetition_penalty(
                self.repetition_penalty.unwrap_or(1.0),
            )?,
            presence_penalty: validate_penalty(self.presence_penalty, "presence_penalty")?,
            frequency_penalty: validate_penalty(self.frequency_penalty, "frequency_penalty")?,
            skip_special_tokens: self.skip_special_tokens.unwrap_or(true),
            thinking: self.spec.thinking,
            reasoning_effort: self.spec.effort,
            stop: self.stop,
            seed: self.seed,
            logprobs: self.logprobs,
            trace_scopes: self.trace.map_or(0, |directive| directive.scopes),
            trace_steps: self.trace.map_or(0, |directive| directive.steps),
        })
    }
}

/// One sampling-validation refusal with vLLM's message text and parameter name.
fn invalid_sampling(message: String, parameter: &str) -> ApiError {
    ApiError::invalid_request(
        message,
        Some(parameter.to_owned()),
        Some(format!("invalid_{parameter}")),
    )
}

/// `top_p` is vLLM's range opened at the bottom: 0.0 means "keep the single most likely token"
/// here, our documented extension (vLLM refuses 0.0 because its range is (0, 1]).
fn validate_top_p(value: f32) -> Result<f32, ApiError> {
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(invalid_sampling(
            format!("top_p must be in [0, 1], got {value:?}."),
            "top_p",
        ))
    }
}

/// vLLM: 0 (and -1) disable the filter, anything above zero keeps that many most likely tokens.
fn validate_top_k(value: i32) -> Result<i32, ApiError> {
    if value >= -1 {
        Ok(value)
    } else {
        Err(invalid_sampling(
            format!("top_k must be 0 (disable), or at least 1, got {value}."),
            "top_k",
        ))
    }
}

/// vLLM: the penalty must be finite and greater than zero.
fn validate_repetition_penalty(value: f32) -> Result<f32, ApiError> {
    if !value.is_finite() {
        return Err(invalid_sampling(
            format!("repetition_penalty must be a finite number, got {value:?}."),
            "repetition_penalty",
        ));
    }
    if value <= 0.0 {
        return Err(invalid_sampling(
            format!("repetition_penalty must be greater than zero, got {value:?}."),
            "repetition_penalty",
        ));
    }
    Ok(value)
}

/// vLLM: presence and frequency penalties live in [-2, 2].
fn validate_penalty(value: Option<f32>, parameter: &str) -> Result<f32, ApiError> {
    let value = value.unwrap_or(0.0);
    if (-2.0..=2.0).contains(&value) {
        Ok(value)
    } else {
        Err(invalid_sampling(
            format!("{parameter} must be in [-2, 2], got {value:?}."),
            parameter,
        ))
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

/// The comment frame written while an SSE stream has no engine event to send.
///
/// An SSE comment starts with a colon, is ignored by clients, and keeps an otherwise idle
/// connection open; it is not a JSON event and never appears after the terminator.
pub const SSE_KEEP_ALIVE_COMMENT: &[u8] = b": ping\n\n";

/// Wraps an SSE byte stream with idle comment frames.
///
/// The timer restarts after every item, so a busy stream is byte-identical to the unwrapped one;
/// only a gap longer than `interval` between engine events adds a comment. This is the same slot
/// the reference server's SSE keep-alive fills, and it never moves or duplicates the data frames.
fn sse_with_keep_alive<S>(
    stream: S,
    interval: Duration,
) -> impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static
where
    S: futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
{
    futures::stream::unfold(Box::pin(stream), move |mut stream| async move {
        match tokio::time::timeout(interval, stream.next()).await {
            Ok(Some(item)) => Some((item, stream)),
            Ok(None) => None,
            Err(_) => Some((Ok(Bytes::from_static(SSE_KEEP_ALIVE_COMMENT)), stream)),
        }
    })
}

fn sse_response(
    stream: impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
    trace: Option<TraceDirective>,
    keep_alive: Option<Duration>,
) -> Response {
    let stream: futures::stream::BoxStream<'static, Result<Bytes, Infallible>> = match keep_alive {
        Some(interval) => Box::pin(sse_with_keep_alive(stream, interval)),
        None => Box::pin(stream),
    };
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

/// Maps a prompt-codec refusal to the OpenAI-shaped 400, or 500 for a tokenizer failure.
///
/// The message and the optional parameter/code come from the codec, so the V4.1 pack keeps the
/// exactly matching refusal strings while the router stays model-agnostic.
fn prompt_codec_error(error: PromptError) -> ApiError {
    if error.is_internal() {
        return ApiError::internal(error.to_string());
    }
    ApiError::invalid_request(
        error.to_string(),
        error.param().map(str::to_owned),
        error.code().map(str::to_owned),
    )
}

/// A prompt summary carrying the checkpoint's real ids.
fn tokenizer_prompt_summary(ids: Vec<u32>, text_bytes: usize) -> PromptSummary {
    PromptSummary {
        token_ids: ids,
        text_bytes,
        images: Vec::new(),
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let value = parse_json(&body)?;
    let spec = resolve_reasoning(&value)?;
    let image_carrier = find_image_carrier(&value);
    // The chat template reads the raw body (chat_template_kwargs and the top-level
    // reasoning_effort decide thinking and the rendered budget), so keep a copy only when a
    // tokenizer seam is attached; the placeholder path does not need it.
    let seam = state.tokenizer().cloned();
    let mut raw_for_seam = seam.as_ref().map(|_| value.clone());
    let request: ChatCompletionRequest = serde_json::from_value(value).map_err(invalid_body)?;
    // vLLM's auto tool choice is enabled by `--enable-auto-tool-choice --tool-call-parser
    // deepseek_v41`; the request side of that contract is a non-null `tools` list, and the
    // answer side is parsed below before the OpenAI tool_calls array is filled.
    let tool_policy = resolve_tool_policy(
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
    )?;
    let tools_offered = tool_policy.tools_offered;
    if !tools_offered
        && let Some(raw) = raw_for_seam.as_mut()
        && let Some(map) = raw.as_object_mut()
    {
        // `tool_choice: "none"` removes the tools from the rendered template instead of merely
        // ignoring the parsed call: vLLM's chat utils do not add a tool list the model is told not
        // to use, and a prompt carrying tools changes what the model writes even when the caller
        // forbade calls.
        map.remove("tools");
        map.remove("tool_choice");
    }
    if request.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "\"messages\" must be a non-empty array",
            Some("messages".to_owned()),
            Some("invalid_messages".to_owned()),
        ));
    }
    let n = resolve_choice_count(request.n, state.engine.as_ref())?;
    refuse_unenforceable_response_format(request.response_format.as_ref())?;
    // Logprobs are resolved and validated before generation, exactly where vLLM validates them:
    // an impossible request (too many alternatives, out-of-vocabulary ids, streaming prompt
    // logprobs) is a 400, never a truncated answer.
    let vocab_size = state.engine.vocab_size();
    // logit_bias/min_p are validated against the stated vocabulary, then refused typed: the engine
    // seam has no logits processor (see the function's own comment).
    check_sampling_extensions(request.logit_bias.as_ref(), request.min_p, vocab_size)?;
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
    // vLLM's chat `echo` implication is deliberately not implemented: the template now renders
    // (see encoding/seam.rs), but echoing the final message changes the generation assembly, not
    // the prompt seam, so it stays deferred with the logprob protocol work. Explicit
    // prompt_logprobs are supported, and are refused for streams below.
    let prompt_logprobs = resolve_logprob_selection(
        request.prompt_logprobs,
        "prompt_logprobs",
        "prompt",
        state.max_logprobs(),
        vocab_size,
    )?;
    reject_streaming_prompt_logprobs(request.stream, request.prompt_logprobs)?;
    // Cloned rather than moved: the device-chain eligibility check below still reads the request.
    let token_ids =
        resolve_logprob_token_ids(request.logprob_token_ids.clone(), sample, vocab_size)?;
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
    // With the seam attached, the prompt is the released server's template rendered by the
    // official encoder port and tokenized with the checkpoint tokenizer. Without it (the
    // weight-free router tests), the Milestone 1 heuristic keeps the placeholder path.
    let (mut summary, prompt_token_ids) = match (seam.as_ref(), raw_for_seam.as_ref()) {
        (Some(seam), Some(raw)) => {
            let ids = seam.encode_chat(raw).map_err(prompt_codec_error)?;
            (tokenizer_prompt_summary(ids.clone(), 0), Some(ids))
        }
        _ => (summarize_chat_messages(&request.messages)?, None),
    };
    // Device-chain scaffold: a plain chat request may be served from a claimed four-rank token
    // envelope; an unclaimed envelope, a missing source or any advanced field continues to the host
    // twin below, so the existing serve path stays byte-identical.
    if let Some(claim) =
        device_chat_eligibility(&request, image_carrier.is_none(), prompt_token_ids)
            .and_then(|token_request| claim_device_token(&state, token_request))
    {
        return Ok(device_token_response(
            &claim,
            DeviceEndpoint::ChatCompletions,
            &state,
            &model,
            request.stream,
            request
                .stream_options
                .as_ref()
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
            trace,
        ));
    }
    // The vision path (D59): for an engine with the vision tower, each image part is prepared and
    // its placeholder (the template renders one per part) expanded to the image's span, so the
    // prompt length below counts the visual tokens. A carrier left once the parts are taken is a
    // shape this route does not read, refused rather than answered without its image.
    let mut vision = Vec::new();
    let capabilities = state.engine.capabilities();
    match raw_for_seam
        .as_ref()
        .filter(|_| image_carrier.is_some() && capabilities.vision)
    {
        Some(raw) => vision = apply_chat_images(&state, raw, &mut summary).await?,
        None => refuse_image_carrier(image_carrier, capabilities.vision)?,
    }
    let id = state.next_id("chatcmpl-dsv41-");
    let created = state.now_unix();
    let trace_id = trace_request_id(&headers, &id);
    let prompt_tokens = summary.token_ids.len();
    let max_tokens = resolve_max_tokens_with_ceiling(
        request.max_completion_tokens,
        "max_completion_tokens",
        request.max_tokens,
        "max_tokens",
        spec,
        prompt_tokens,
        state.max_seq_len(),
        state.max_output_tokens(),
    )?;
    let engine_request = RequestBuilder {
        id: id.clone(),
        model: model.clone(),
        summary,
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        repetition_penalty: request.repetition_penalty,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        skip_special_tokens: request.skip_special_tokens,
        spec,
        stop: normalize_stop(request.stop.as_ref())?,
        seed: request.seed,
        logprobs,
        trace,
        vision,
    }
    .build()?;
    // One engine request per choice. The reference engine serves one sequence per request, so n
    // choices are n sequential generate calls on the same prompt, each with vLLM's per-choice seed
    // derivation. n == 1 keeps the original request id, so single-choice responses are unchanged.
    let requests: Vec<EngineRequest> = (0..n)
        .map(|choice| {
            let mut engine_request = engine_request.clone();
            if n > 1 {
                engine_request.request_id = format!("{id}-{choice}");
            }
            engine_request.seed = state.choice_seed(request.seed, choice, n);
            engine_request
        })
        .collect();
    // The capability boundary: an engine that cannot serve a channel the request needs refuses
    // here, with a typed error naming it, before any token is generated or streamed.
    for engine_request in &requests {
        state
            .engine
            .validate_request(engine_request)
            .map_err(ApiError::from)?;
    }

    if request.stream {
        let include_usage = request
            .stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false);
        // vLLM only reports continuous usage when include_usage is on as well; a request that asks
        // for it alone gets a stream with no usage block at all, which is what the reference does.
        let continuous_usage_stats = include_usage
            && request
                .stream_options
                .and_then(|options| options.continuous_usage_stats)
                .unwrap_or(false);
        Ok(chat_stream_response(
            &state,
            requests,
            StreamContext {
                id,
                created,
                model,
                include_usage,
                trace_id,
                tools_offered,
                parallel_tool_calls: tool_policy.parallel,
                continuous_usage_stats,
            },
            logprob_info,
            trace,
        ))
    } else {
        let _binding = RequestTraceBinding::new(trace_id);
        let mut outputs = Vec::with_capacity(requests.len());
        for engine_request in requests {
            outputs.push(drive_engine(state.engine.as_ref(), engine_request).await?);
        }
        for output in &mut outputs {
            if tools_offered {
                apply_deepseek_v41_tool_parser(output, &state);
            }
            if !tool_policy.parallel && output.tool_calls.len() > 1 {
                // parallel_tool_calls = false: report the first call only, instead of handing back
                // a parallel set the caller said it cannot accept. The block is already generated,
                // so this is the parser-boundary enforcement point.
                output.tool_calls.truncate(1);
                output.finish_reason = FinishReason::ToolCalls;
            }
        }
        let usage = aggregate_usage_shared_prompt(&outputs);
        let stats = aggregate_stats(&outputs);
        state.metrics.record_request(&usage, &stats.kv_cache);
        state.metrics.record_request_stats(&stats);
        Ok(json_response(
            chat_completions_body(&outputs, &id, created, &model, logprob_info),
            trace,
        ))
    }
}

/// Lifts vLLM's `deepseek_v41` DSML tool calls out of the buffered content.
///
/// The streaming path applies the incremental [`tool_parser::ToolCallStreamParser`] while the
/// content deltas arrive instead, so both shapes lift the same block; see
/// [`chat_stream_response`].
fn apply_deepseek_v41_tool_parser(output: &mut EngineOutput, state: &AppState) {
    if !output.tool_calls.is_empty() {
        return;
    }
    let Some(parsed) = tool_parser::parse_completion(&output.content) else {
        return;
    };
    if parsed.tool_calls.is_empty() {
        return;
    }
    output.content = parsed.content;
    output.tool_calls = parsed
        .tool_calls
        .into_iter()
        .map(|call| BufferedToolCall {
            id: state.next_id("chatcmpl-tool-"),
            name: call.name,
            arguments: call.arguments,
        })
        .collect();
    output.finish_reason = FinishReason::ToolCalls;
}

fn chat_completions_body(
    outputs: &[EngineOutput],
    id: &str,
    created: u64,
    model: &str,
    logprob_info: LogprobInfo,
) -> Value {
    let choices: Vec<Value> = outputs
        .iter()
        .enumerate()
        .map(|(index, output)| chat_choice(output, index, logprob_info))
        .collect();
    let mut body = Map::new();
    body.insert("id".to_owned(), Value::String(id.to_owned()));
    body.insert(
        "object".to_owned(),
        Value::String("chat.completion".to_owned()),
    );
    body.insert("created".to_owned(), json!(created));
    body.insert("model".to_owned(), Value::String(model.to_owned()));
    body.insert("choices".to_owned(), Value::Array(choices));
    // Prompt logprobs describe the prompt every choice shares, so the first output's payload is
    // the response's; the extension is emitted only when the request asked for it.
    if let Some(prompt_logprobs) = outputs
        .first()
        .and_then(|output| output.prompt_logprobs.as_ref())
    {
        body.insert(
            "prompt_logprobs".to_owned(),
            prompt_logprobs_value(prompt_logprobs),
        );
    }
    body.insert(
        "usage".to_owned(),
        serde_json::to_value(aggregate_usage_shared_prompt(outputs)).unwrap_or(Value::Null),
    );
    body.insert(
        "x_deepseek_spark_stats".to_owned(),
        serde_json::to_value(aggregate_stats(outputs).as_ref()).unwrap_or(Value::Null),
    );
    Value::Object(body)
}

/// Builds one `choices[]` entry of a buffered chat response.
///
/// `logprobs` is always present (null when not requested, like the OpenAI schema and vLLM);
/// `prompt_logprobs` is a response-level vLLM extension, emitted by the caller.
fn chat_choice(output: &EngineOutput, index: usize, logprob_info: LogprobInfo) -> Value {
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
            .map(|(call_index, call)| {
                json!({
                    "index": call_index,
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect();
        message.insert("tool_calls".to_owned(), Value::Array(calls));
    }
    let logprobs = if logprob_info.wants_sample() {
        chat_logprobs_value(output, logprob_info)
    } else {
        Value::Null
    };
    let mut choice = Map::new();
    choice.insert("index".to_owned(), json!(index));
    choice.insert("message".to_owned(), Value::Object(message));
    choice.insert("logprobs".to_owned(), logprobs);
    choice.insert(
        "finish_reason".to_owned(),
        serde_json::to_value(output.finish_reason).unwrap_or(Value::Null),
    );
    Value::Object(choice)
}

#[allow(clippy::too_many_arguments)] // the wire chunk's fields, one argument each
fn chat_chunk(
    id: &str,
    model: &str,
    created: u64,
    choice_index: usize,
    delta: Value,
    finish_reason: Option<FinishReason>,
    logprobs: Option<Value>,
    stats: Option<&DeepSeekSparkRequestStats>,
) -> Value {
    let mut choice = json!({"index": choice_index, "delta": delta, "finish_reason": finish_reason});
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

/// Formats the id of one streamed tool call from the shared response-id counter.
///
/// The buffered path formats the same `chatcmpl-tool-` ids through [`AppState::next_id`]; both
/// paths draw from one counter, so the ids are identical for identical request orders.
fn next_tool_call_id(counter: &std::sync::atomic::AtomicU64) -> String {
    format!(
        "chatcmpl-tool-{:08x}",
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

/// The pending streamed chunks of one choice: parser events and logprob entries.
///
/// When a request offers tools *and* asks for sample logprobs, vLLM reports both at once: the DSML
/// block becomes delta.tool_calls while each chunk still carries the logprobs entries of every
/// token generated since the previous chunk - including the marker tokens the parser removed from
/// the content. This queue reproduces that shape: parser events and logprob entries accumulate, and
/// a drain emits one chunk per event with the pending logprob list attached to the first chunk of
/// the drain.
#[derive(Default)]
struct StreamAssembler {
    events: Vec<tool_parser::StreamToolEvent>,
    logprobs: Vec<Value>,
    calls: usize,
}

impl StreamAssembler {
    /// Queues what the incremental parser produced for one content delta.
    fn push(&mut self, events: Vec<tool_parser::StreamToolEvent>) {
        self.events.extend(events);
    }

    /// How many tool calls this choice has started streaming.
    fn calls(&self) -> usize {
        self.calls
    }

    /// Turns the queued events into (delta, logprobs) pairs.
    ///
    /// A request that disabled parallel calls keeps only call index 0: later calls and their
    /// argument fragments are dropped before they can reach the wire.
    fn drain(
        &mut self,
        next_id: &mut impl FnMut() -> String,
        parallel: bool,
    ) -> Vec<(Value, Option<Value>)> {
        // A logprob with no delta to ride on stays queued for the next chunk (or the terminal
        // chunk): vLLM never drops a token's logprob just because its delta was held by the
        // tool-call parser.
        if self.events.is_empty() {
            return Vec::new();
        }
        let mut logprobs = (!self.logprobs.is_empty())
            .then(|| json!({"content": std::mem::take(&mut self.logprobs)}));
        let mut chunks: Vec<(Value, Option<Value>)> = Vec::new();
        for event in std::mem::take(&mut self.events) {
            let delta = match event {
                tool_parser::StreamToolEvent::Content(content) => json!({"content": content}),
                tool_parser::StreamToolEvent::CallStart { index, name } => {
                    if !parallel && index > 0 {
                        continue;
                    }
                    self.calls += 1;
                    tool_call_delta(index, Some(next_id()), Some(name), String::new())
                }
                tool_parser::StreamToolEvent::ArgumentsDelta {
                    index,
                    arguments_delta,
                } => {
                    if !parallel && index > 0 {
                        continue;
                    }
                    tool_call_delta(index, None, None, arguments_delta)
                }
            };
            chunks.push((delta, logprobs.take()));
        }
        chunks
    }

    /// The logprob entries still pending, for the terminal chunk to carry.
    fn take_logprobs(&mut self) -> Option<Value> {
        (!self.logprobs.is_empty()).then(|| json!({"content": std::mem::take(&mut self.logprobs)}))
    }
}

/// Adds the running usage block to a chunk when stream_options.continuous_usage_stats is on.
///
/// The reference server puts usage on every chunk of such a stream and keeps the empty-choices
/// final chunk; the counts grow with generation and the last per-chunk value is the authoritative
/// one because the engine's own completion_tokens replaces the delta count when a choice finishes.
fn attach_continuous_usage(chunk: &mut Value, usage: &OpenAiUsageWithCache) {
    if let Some(map) = chunk.as_object_mut() {
        map.insert(
            "usage".to_owned(),
            serde_json::to_value(usage).unwrap_or(Value::Null),
        );
    }
}

fn chat_stream_response(
    state: &AppState,
    requests: Vec<EngineRequest>,
    context: StreamContext,
    logprob_info: LogprobInfo,
    trace: Option<TraceDirective>,
) -> Response {
    let engine = Arc::clone(&state.engine);
    let metrics = Arc::clone(&state.metrics);
    let id_counter = state.id_counter();
    let StreamContext {
        id,
        created,
        model,
        include_usage,
        trace_id,
        tools_offered,
        parallel_tool_calls,
        continuous_usage_stats,
    } = context;
    // The parser runs whenever the client offered tools. It used to be switched off for
    // streaming-logprob requests, which left the raw DSML block in the content; vLLM parses that
    // combination too and puts the logprob entries on the same chunks as the tool-call deltas, so
    // the assembler above carries both.
    let parse_tool_calls = tools_offered;
    let logprob_mode = logprob_info.wants_sample();
    let choice_count = requests.len();
    let prompt_tokens = requests
        .first()
        .map_or(0, |request| request.prompt_tokens.len());
    let stream = async_stream::stream! {
        // The spans are emitted while this generator is polled, so the binding has to be
        // taken here rather than on the handler's stack.
        let _binding = RequestTraceBinding::new(trace_id);
        // The running usage block of a continuous-usage stream.
        let usage_now = |finished_completion: usize, current_completion: usize, reasoning: usize| {
            let mut usage = OpenAiUsageWithCache::default();
            usage.prompt_tokens = prompt_tokens;
            usage.completion_tokens = finished_completion + current_completion;
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
            usage.completion_tokens_details.reasoning_tokens = reasoning;
            usage
        };
        let mut finished_completion = 0usize;
        let mut current_completion = 0usize;
        let mut reasoning_tokens = 0usize;
        // One assistant role chunk opens the response; later choices identify themselves with
        // their choice index instead of repeating the role, which is the OpenAI stream convention.
        let mut role_chunk = chat_chunk(
            &id,
            &model,
            created,
            0,
            json!({"role": "assistant"}),
            None,
            None,
            None,
        );
        if continuous_usage_stats {
            attach_continuous_usage(
                &mut role_chunk,
                &usage_now(finished_completion, current_completion, reasoning_tokens),
            );
        }
        yield Ok::<Bytes, Infallible>(sse_frame(&role_chunk));
        let mut next_tool_id = {
            let counter = Arc::clone(&id_counter);
            move || next_tool_call_id(&counter)
        };
        // With n > 1 the per-choice usage and stats are aggregated after every choice has
        // finished; a single choice keeps the engine's own objects and is recorded immediately.
        let mut usage_parts: Vec<OpenAiUsageWithCache> = Vec::with_capacity(choice_count);
        let mut stats_parts: Vec<Box<DeepSeekSparkRequestStats>> = Vec::with_capacity(choice_count);
        let mut terminal = false;
        'choices: for (choice_index, request) in requests.into_iter().enumerate() {
            // When logprobs are requested without a tool parser, the text has to travel with the
            // logprob entry of the token it belongs to, so content deltas are held here until its
            // TokenLogprobs event arrives.
            let mut buffered = String::new();
            let mut assembler = StreamAssembler::default();
            let mut tool_parser = parse_tool_calls.then(tool_parser::ToolCallStreamParser::new);
            // A malformed or unterminated DSML block is terminal for the stream: the deltas already
            // sent cannot be taken back, so the error frame ends the response and later events are
            // dropped instead of interleaving content with a broken call.
            let mut parser_refusal: Option<String> = None;
            let mut events = engine.generate(request);
            while let Some(event) = events.next().await {
                match event {
                    EngineEvent::Started { .. } => {}
                    // Explicit prompt_logprobs are refused for streams, so this cannot arrive here.
                    EngineEvent::PromptLogprobs(_) => {}
                    EngineEvent::ReasoningDelta(text) => {
                        current_completion += 1;
                        reasoning_tokens += 1;
                        let mut chunk = chat_chunk(
                            &id,
                            &model,
                            created,
                            choice_index,
                            json!({"reasoning_content": text}),
                            None,
                            None,
                            None,
                        );
                        if continuous_usage_stats {
                            attach_continuous_usage(
                                &mut chunk,
                                &usage_now(
                                    finished_completion,
                                    current_completion,
                                    reasoning_tokens,
                                ),
                            );
                        }
                        yield Ok(sse_frame(&chunk));
                    }
                    EngineEvent::ContentDelta(text) => {
                        current_completion += 1;
                        if let Some(parser) = tool_parser.as_mut() {
                            if parser_refusal.is_none() {
                                match parser.feed(&text) {
                                    Ok(chunks) => assembler.push(chunks),
                                    Err(refusal) => parser_refusal = Some(refusal.message),
                                }
                            }
                        } else if logprob_mode {
                            buffered.push_str(&text);
                        } else {
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                json!({"content": text}),
                                None,
                                None,
                                None,
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                        }
                    }
                    EngineEvent::TokenLogprobs(logprob) => {
                        if logprob_mode {
                            if parse_tool_calls {
                                // Queued with the parser deltas: vLLM reports the token's logprob on
                                // whichever chunk carries the delta the token produced, including
                                // the DSML marker tokens that never appear as content.
                                assembler
                                    .logprobs
                                    .push(chat_logprobs_entry(&logprob, logprob_info));
                            } else {
                                let text = std::mem::take(&mut buffered);
                                let logprobs =
                                    json!({"content": [chat_logprobs_entry(&logprob, logprob_info)]});
                                let mut chunk = chat_chunk(
                                    &id,
                                    &model,
                                    created,
                                    choice_index,
                                    json!({"content": text}),
                                    None,
                                    Some(logprobs),
                                    None,
                                );
                                if continuous_usage_stats {
                                    attach_continuous_usage(
                                        &mut chunk,
                                        &usage_now(
                                            finished_completion,
                                            current_completion,
                                            reasoning_tokens,
                                        ),
                                    );
                                }
                                yield Ok(sse_frame(&chunk));
                            }
                        }
                    }
                    EngineEvent::ToolCallDelta { index, id: call_id, name, arguments_delta } => {
                        current_completion += 1;
                        if parallel_tool_calls || index == 0 {
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                tool_call_delta(index, call_id, name, arguments_delta),
                                None,
                                None,
                                None,
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                        }
                    }
                    EngineEvent::Finished { finish_reason, usage, stats } => {
                        usage_parts.push(usage);
                        if choice_count == 1 {
                            metrics.record_request(&usage, &stats.kv_cache);
                            metrics.record_request_stats(&stats);
                        }
                        finished_completion += usage.completion_tokens;
                        current_completion = 0;
                        stats_parts.push(stats);
                        if let Some(parser) = tool_parser.as_mut()
                            && parser_refusal.is_none()
                        {
                            match parser.finish() {
                                Ok(chunks) => assembler.push(chunks),
                                Err(refusal) => parser_refusal = Some(refusal.message),
                            }
                        }
                        if let Some(message) = parser_refusal.take() {
                            yield Ok(sse_frame(&json!({
                                "error": {
                                    "message": message,
                                    "type": "server_error",
                                    "param": null,
                                    "code": tool_parser::TOOL_STREAM_REFUSAL_CODE,
                                },
                            })));
                            // The error frame is terminal for the response: no finish chunk and no
                            // usage chunk follow it, and the trailing [DONE] still closes the SSE.
                            terminal = true;
                            break 'choices;
                        }
                        // An engine that was asked for logprobs but emitted none for its last token
                        // has its text flushed here, without a logprob entry rather than with a
                        // fabricated one.
                        if logprob_mode && !parse_tool_calls && !buffered.is_empty() {
                            let text = std::mem::take(&mut buffered);
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                json!({"content": text}),
                                None,
                                None,
                                None,
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                        }
                        // Whatever the parser held back is emitted before the finish chunk, and the
                        // logprob entries still pending ride on the first of those chunks.
                        let pending = assembler.drain(&mut next_tool_id, parallel_tool_calls);
                        for (delta, logprobs) in pending {
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                delta,
                                None,
                                logprobs,
                                None,
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                        }
                        let parser_calls = assembler.calls();
                        // A streamed DSML block is a tool call exactly like the buffered one: the
                        // choice ends with tool_calls, not the engine's stop.
                        let finish_reason = if parser_calls > 0 {
                            FinishReason::ToolCalls
                        } else {
                            finish_reason
                        };
                        if choice_count == 1 {
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                json!({}),
                                Some(finish_reason),
                                assembler.take_logprobs(),
                                Some(stats_parts.last().expect("the choice's stats are recorded").as_ref()),
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                            if include_usage {
                                yield Ok(sse_frame(&chat_usage_chunk(&id, &model, created, &usage)));
                            }
                        } else if choice_index + 1 == choice_count {
                            let parts: Vec<(&DeepSeekSparkRequestStats, &OpenAiUsageWithCache)> =
                                stats_parts
                                    .iter()
                                    .zip(&usage_parts)
                                    .map(|(stats, usage)| (stats.as_ref(), usage))
                                    .collect();
                            let aggregate = aggregate_stats_parts(&parts);
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                json!({}),
                                Some(finish_reason),
                                assembler.take_logprobs(),
                                Some(aggregate.as_ref()),
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                        } else {
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                json!({}),
                                Some(finish_reason),
                                assembler.take_logprobs(),
                                None,
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                        }
                    }
                    EngineEvent::Error { message, code } => {
                        yield Ok(sse_frame(&json!({
                            "error": {"message": message, "type": "server_error", "param": null, "code": code},
                        })));
                    }
                }
                // Parser output for this event goes out before the next engine event. A choice
                // that is only producing reasoning logprobs still delivers them, on a delta:{}
                // chunk (the shape vLLM uses for a chunk with no delta).
                if parse_tool_calls {
                    let pending = assembler.drain(&mut next_tool_id, parallel_tool_calls);
                    if pending.is_empty() {
                        if let Some(logprobs) = assembler.take_logprobs() {
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                json!({}),
                                None,
                                Some(logprobs),
                                None,
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                        }
                    } else {
                        for (delta, logprobs) in pending {
                            let mut chunk = chat_chunk(
                                &id,
                                &model,
                                created,
                                choice_index,
                                delta,
                                None,
                                logprobs,
                                None,
                            );
                            if continuous_usage_stats {
                                attach_continuous_usage(
                                    &mut chunk,
                                    &usage_now(
                                        finished_completion,
                                        current_completion,
                                        reasoning_tokens,
                                    ),
                                );
                            }
                            yield Ok(sse_frame(&chunk));
                        }
                    }
                }
            }
        }
        // n > 1 aggregates the choices into the one usage block vLLM returns, and records the
        // response's metrics once. A terminal parser refusal skips the usage chunk, like n == 1.
        if choice_count > 1 && !usage_parts.is_empty() {
            let parts: Vec<(&DeepSeekSparkRequestStats, &OpenAiUsageWithCache)> = stats_parts
                .iter()
                .zip(&usage_parts)
                .map(|(stats, usage)| (stats.as_ref(), usage))
                .collect();
            let stats = aggregate_stats_parts(&parts);
            let usage = aggregate_usage_shared_prompt_parts(&usage_parts);
            metrics.record_request(&usage, &stats.kv_cache);
            metrics.record_request_stats(&stats);
            if include_usage && !terminal {
                yield Ok(sse_frame(&chat_usage_chunk(&id, &model, created, &usage)));
            }
        }
        yield Ok(sse_done());
    };
    sse_response(stream, trace, state.sse_keep_alive())
}
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
    // vLLM's list-prompt contract plus OpenAI's n: each prompt expands into n independent
    // choices, and the response choice index is prompt_index * n + choice_index.
    let n = resolve_choice_count(request.n, state.engine.as_ref())?;
    // Before the device-token fast path: an envelope claim replays a recorded window on the ranks
    // without the host reading the prompt, so the image refusal must not depend on which path
    // serves the request.
    refuse_completion_image(&state, request.prompt.as_ref(), image_carrier)?;
    // Device-chain scaffold: only a plain, single-choice, pre-tokenized request is served from a
    // claimed four-rank token envelope; everything else continues to the host twin below.
    if let Some(claim) = device_completion_eligibility(&request)
        .and_then(|token_request| claim_device_token(&state, token_request))
    {
        return Ok(device_token_response(
            &claim,
            DeviceEndpoint::Completions,
            &state,
            &model,
            request.stream,
            request
                .stream_options
                .as_ref()
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
            trace,
        ));
    }
    // vLLM's completion batch contract: a list prompt is a list of independent prompts, each
    // producing its own choice in one response (vllm/entrypoints/openai/completion/serving.py
    // builds one choice per prompt and sums usage over the batch). `tokens` carries the pre-tokenized form straight through;
    // text is tokenized with the checkpoint seam when attached, and the Milestone 1 heuristic keeps
    // the weight-free placeholder path.
    let prompts = match request.prompt.as_ref() {
        Some(PromptSpec::Text(text)) => vec![CompletionPrompt {
            text: text.clone(),
            tokens: None,
        }],
        Some(PromptSpec::Many(items)) if items.is_empty() => {
            return Err(ApiError::invalid_request(
                "\"prompt\" must contain at least one prompt",
                Some("prompt".to_owned()),
                Some("invalid_prompt".to_owned()),
            ));
        }
        Some(PromptSpec::Many(items)) => items
            .iter()
            .map(|text| CompletionPrompt {
                text: text.clone(),
                tokens: None,
            })
            .collect(),
        Some(PromptSpec::Tokens(ids)) => {
            let text = state
                .tokenizer()
                .and_then(|seam| seam.decode(ids))
                .unwrap_or_default();
            vec![CompletionPrompt {
                text,
                tokens: Some(ids.clone()),
            }]
        }
        None => {
            return Err(ApiError::invalid_request(
                "\"prompt\" is required",
                Some("prompt".to_owned()),
                Some("invalid_prompt".to_owned()),
            ));
        }
    };
    let id = state.next_id("cmpl-dsv41-");
    let created = state.now_unix();
    let trace_id = trace_request_id(&headers, &id);
    // vLLM's echo rules: `echo` with no explicit `prompt_logprobs` computes prompt logprobs with
    // `logprobs`'s value, and `echo` with `max_tokens == 0` is the echo-without-generation
    // request (text is the prompt, no tokens are generated, finish_reason is "length").
    let echo = request.echo.unwrap_or(false);
    let vocab_size = state.engine.vocab_size();
    // S2: the completions schema now carries response_format with the chat refusal rule, and S5's
    // best_of is accepted as vLLM's documented no-op. logit_bias/min_p follow the chat validation.
    refuse_unenforceable_response_format(request.response_format.as_ref())?;
    check_sampling_extensions(request.logit_bias.as_ref(), request.min_p, vocab_size)?;
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
    let stop = normalize_stop(request.stop.as_ref())?;
    let mut engine_requests = Vec::with_capacity(prompts.len() * n);
    // The engine has one sequence per generate call, so (prompt, choice) pairs become requests;
    // request_plan keeps the wire mapping back to the response choice index.
    let mut request_plan: Vec<(usize, usize)> = Vec::with_capacity(prompts.len() * n);
    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let summary = match &prompt.tokens {
            Some(ids) => tokenizer_prompt_summary(ids.clone(), prompt.text.len()),
            None => match state.tokenizer() {
                Some(seam) => {
                    let ids = seam
                        .encode_text(&prompt.text, false)
                        .map_err(prompt_codec_error)?;
                    tokenizer_prompt_summary(ids, prompt.text.len())
                }
                None => summarize_prompt_text(&prompt.text),
            },
        };
        let max_tokens = if echo && request.max_tokens == Some(0) {
            0
        } else {
            resolve_max_tokens_with_ceiling(
                None,
                "max_tokens",
                request.max_tokens,
                "max_tokens",
                spec,
                summary.token_ids.len(),
                state.max_seq_len(),
                state.max_output_tokens(),
            )?
        };
        for choice in 0..n {
            // vLLM gives every prompt its own engine request ('{request_id}-{i}'); a single-prompt
            // request keeps the response id, and an n-choice request names the choice too so a
            // trace or a log can tell the sequences apart.
            let request_id = if prompts.len() == 1 && n == 1 {
                id.clone()
            } else if n == 1 {
                format!("{id}-{prompt_index}")
            } else {
                format!("{id}-{prompt_index}-{choice}")
            };
            engine_requests.push(
                RequestBuilder {
                    id: request_id,
                    model: model.clone(),
                    summary: summary.clone(),
                    max_tokens,
                    temperature: request.temperature,
                    top_p: request.top_p,
                    top_k: request.top_k,
                    repetition_penalty: request.repetition_penalty,
                    presence_penalty: request.presence_penalty,
                    frequency_penalty: request.frequency_penalty,
                    skip_special_tokens: request.skip_special_tokens,
                    spec,
                    stop: stop.clone(),
                    seed: state.choice_seed(request.seed, choice, n),
                    logprobs: logprobs.clone(),
                    trace,
                    vision: Vec::new(),
                }
                .build()?,
            );
            request_plan.push((prompt_index, choice));
        }
    }
    for engine_request in &engine_requests {
        state
            .engine
            .validate_request(engine_request)
            .map_err(ApiError::from)?;
    }

    if request.stream {
        let include_usage = request
            .stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false);
        Ok(completions_stream_response(
            &state,
            engine_requests,
            &prompts,
            &request_plan,
            n,
            StreamContext {
                id,
                created,
                model,
                include_usage,
                trace_id,
                tools_offered: false,
                parallel_tool_calls: true,
                continuous_usage_stats: false,
            },
            logprob_info,
            echo,
            trace,
        ))
    } else {
        let _binding = RequestTraceBinding::new(trace_id);
        let mut outputs = Vec::with_capacity(engine_requests.len());
        for engine_request in engine_requests {
            outputs.push(drive_engine(state.engine.as_ref(), engine_request).await?);
        }
        let usage = aggregate_completion_usage(&outputs, &request_plan);
        let stats = aggregate_stats(&outputs);
        state.metrics.record_request(&usage, &stats.kv_cache);
        state.metrics.record_request_stats(&stats);
        let body = completion_body(
            &outputs,
            &prompts,
            &request_plan,
            n,
            &id,
            created,
            &model,
            logprob_info,
            echo,
        );
        Ok(json_response(body, trace))
    }
}

/// One prompt of a `/v1/completions` batch: the text used for echo and summary, and, for the
/// vLLM `int[]` form, the caller's token ids.
struct CompletionPrompt {
    text: String,
    tokens: Option<Vec<u32>>,
}

/// Builds the OpenAI /v1/completions body for a batch of prompts, applying vLLM's echo and
/// logprob rules.
///
/// vLLM's contract for a list prompt is one response whose `choices` carries one entry per prompt
/// (`index` = choice position, echo text and prompt logprobs per prompt) and one `usage` object
/// summed over the batch (`serving_completion.py::request_output_to_completion_response`). The
/// `x_deepseek_spark_stats` extension is aggregated the same way, with its rates recomputed.
#[allow(clippy::too_many_arguments)] // the response assembly's inputs, one argument each
fn completion_body(
    outputs: &[EngineOutput],
    prompts: &[CompletionPrompt],
    request_plan: &[(usize, usize)],
    n: usize,
    id: &str,
    created: u64,
    model: &str,
    logprob_info: LogprobInfo,
    echo: bool,
) -> Value {
    debug_assert_eq!(outputs.len(), request_plan.len());
    let choices: Vec<Value> = outputs
        .iter()
        .zip(request_plan)
        .map(|(output, (prompt_index, choice_index))| {
            let prompt = &prompts[*prompt_index];
            let index = *prompt_index * n + *choice_index;
            // A completion has no reasoning field: an engine's reasoning channel is part of the
            // text, as it is in the raw output.
            let text = if echo {
                format!("{}{}{}", prompt.text, output.reasoning, output.content)
            } else {
                format!("{}{}", output.reasoning, output.content)
            };
            let logprobs = if logprob_info.wants_sample() {
                let mut positions = Vec::new();
                if echo && let Some(prompt_logprobs) = output.prompt_logprobs.as_ref() {
                    positions.extend(prompt_positions(prompt_logprobs));
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
                "index": index,
                "text": text,
                "logprobs": logprobs,
                "finish_reason": output.finish_reason,
                // The reference engine has no stop-sequence matching (its finish reason is the
                // token cap or a step with no token), so vLLM's stop_reason is null for every
                // choice until the device engine can report one.
                "stop_reason": Value::Null,
                "prompt_logprobs": prompt_logprobs,
            })
        })
        .collect();
    let usage = aggregate_completion_usage(outputs, request_plan);
    let stats = aggregate_stats(outputs);
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
        "x_deepseek_spark_stats": stats,
    })
}

/// Builds the completion `usage` block: prompts counted once, choices summed.
///
/// OpenAI/vLLM count a prompt once even when n choices were generated from it, so the prompt side
/// comes from each prompt's first choice (choice 0) while every choice's completion side is added.
fn aggregate_completion_usage(
    outputs: &[EngineOutput],
    request_plan: &[(usize, usize)],
) -> OpenAiUsageWithCache {
    debug_assert_eq!(outputs.len(), request_plan.len());
    let mut usage = OpenAiUsageWithCache::default();
    for (output, (_, choice_index)) in outputs.iter().zip(request_plan) {
        if *choice_index == 0 {
            add_prompt_usage(&mut usage, &output.usage);
        }
        add_completion_usage(&mut usage, &output.usage);
    }
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    usage.prompt_cache_hit_tokens = usage.prompt_tokens_details.cached_tokens;
    usage.prompt_cache_miss_tokens = usage
        .prompt_tokens
        .saturating_sub(usage.prompt_tokens_details.cached_tokens);
    usage
}

/// Adds the prompt-side usage fields, which one prompt may share across several choices.
fn add_prompt_usage(total: &mut OpenAiUsageWithCache, usage: &OpenAiUsageWithCache) {
    total.prompt_tokens += usage.prompt_tokens;
    total.prompt_tokens_details.cached_tokens += usage.prompt_tokens_details.cached_tokens;
    total.prompt_tokens_details.uma_cached_tokens += usage.prompt_tokens_details.uma_cached_tokens;
    total.prompt_tokens_details.nvme_cached_tokens +=
        usage.prompt_tokens_details.nvme_cached_tokens;
    total.prompt_tokens_details.image_tokens += usage.prompt_tokens_details.image_tokens;
}

/// Adds the completion-side usage fields; one per generated choice.
fn add_completion_usage(total: &mut OpenAiUsageWithCache, usage: &OpenAiUsageWithCache) {
    total.completion_tokens += usage.completion_tokens;
    total.completion_tokens_details.reasoning_tokens +=
        usage.completion_tokens_details.reasoning_tokens;
    total.completion_tokens_details.accepted_prediction_tokens +=
        usage.completion_tokens_details.accepted_prediction_tokens;
    total.completion_tokens_details.rejected_prediction_tokens +=
        usage.completion_tokens_details.rejected_prediction_tokens;
}

/// Builds the chat `usage` block for one request with one or more choices.
///
/// The choices share the prompt, so the prompt side is counted once (from the first output) while
/// every choice's completion side is summed; `total_tokens` and the DeepSeek cache hit/miss pair
/// are then derived from the sums. A single-choice request therefore reproduces the engine's own
/// usage object exactly.
fn aggregate_usage_shared_prompt(outputs: &[EngineOutput]) -> OpenAiUsageWithCache {
    let usages: Vec<OpenAiUsageWithCache> = outputs.iter().map(|output| output.usage).collect();
    aggregate_usage_shared_prompt_parts(&usages)
}

/// The parts form of [aggregate_usage_shared_prompt], shared with the merged streaming response.
fn aggregate_usage_shared_prompt_parts(usages: &[OpenAiUsageWithCache]) -> OpenAiUsageWithCache {
    let mut usage = OpenAiUsageWithCache::default();
    if let Some(first) = usages.first() {
        add_prompt_usage(&mut usage, first);
    }
    for part in usages {
        add_completion_usage(&mut usage, part);
    }
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    usage.prompt_cache_hit_tokens = usage.prompt_tokens_details.cached_tokens;
    usage.prompt_cache_miss_tokens = usage
        .prompt_tokens
        .saturating_sub(usage.prompt_tokens_details.cached_tokens);
    usage
}

/// Aggregates the per-request `x_deepseek_spark_stats` extension over a batch.
///
/// A single-prompt request returns the engine's own stats unchanged, so the single-prompt wire
/// bytes stay exactly what the engine measured. For several prompts the counters sum, the rates
/// are recomputed from the sums (never averaged), and the wall-clock figures take the batch's
/// worst case: TTFT is the slowest prompt's, throughput is total tokens over summed seconds.
fn aggregate_stats(outputs: &[EngineOutput]) -> Box<DeepSeekSparkRequestStats> {
    let parts: Vec<(&DeepSeekSparkRequestStats, &OpenAiUsageWithCache)> = outputs
        .iter()
        .map(|output| (&*output.stats, &output.usage))
        .collect();
    aggregate_stats_parts(&parts)
}

/// The parts form of [aggregate_stats], shared with the merged streaming response.
fn aggregate_stats_parts(
    parts: &[(&DeepSeekSparkRequestStats, &OpenAiUsageWithCache)],
) -> Box<DeepSeekSparkRequestStats> {
    if let [only] = parts {
        return Box::new((*only.0).clone());
    }
    let mut kv_cache = PrefixMatchPlan {
        total_prompt_tokens: 0,
        global_kv_cached_tokens: 0,
        uma_hit_tokens: 0,
        nvme_restored_tokens: 0,
        swa_tail_hit: false,
        encoder_swa_replay_tokens: 0,
        uncached_suffix_tokens: 0,
        decoder_swa_replay_tokens: 0,
    };
    let mut engram_layers = [
        parts[0].0.engram_layers[0].clone(),
        parts[0].0.engram_layers[1].clone(),
    ];
    for layer in &mut engram_layers {
        layer.lookups_total = 0;
        layer.tier1_drm_hits = 0;
        layer.tier2_uma_hits = 0;
        layer.tier3_nvme_misses = 0;
        layer.hit_rate = 0.0;
        layer.mean_prefetch_stall_us = 0.0;
        layer.mean_gate_alpha = 0.0;
    }
    let mut stall_weighted = [0.0f64; 2];
    let mut gate_weighted = [0.0f64; 2];
    let mut transport = parts[0].0.transport.clone();
    transport.rocenante_oneshot_calls = 0;
    transport.nccl_ring_calls = 0;
    transport.rail0_rocep1s0f0_mb = 0.0;
    transport.rail1_rocep2p1s0f0_mb = 0.0;
    transport.l2_weight_prefetch_overlaps = 0;
    let (mut ttft_ms, mut min_uma_free_gib) = (0.0f64, f64::INFINITY);
    let (mut prefill_seconds, mut decode_seconds) = (0.0f64, 0.0f64);
    for (stats, usage) in parts {
        kv_cache.total_prompt_tokens += stats.kv_cache.total_prompt_tokens;
        kv_cache.global_kv_cached_tokens += stats.kv_cache.global_kv_cached_tokens;
        kv_cache.uma_hit_tokens += stats.kv_cache.uma_hit_tokens;
        kv_cache.nvme_restored_tokens += stats.kv_cache.nvme_restored_tokens;
        kv_cache.swa_tail_hit |= stats.kv_cache.swa_tail_hit;
        kv_cache.encoder_swa_replay_tokens += stats.kv_cache.encoder_swa_replay_tokens;
        kv_cache.uncached_suffix_tokens += stats.kv_cache.uncached_suffix_tokens;
        kv_cache.decoder_swa_replay_tokens += stats.kv_cache.decoder_swa_replay_tokens;
        for (index, layer) in stats.engram_layers.iter().enumerate() {
            let target = &mut engram_layers[index];
            target.lookups_total += layer.lookups_total;
            target.tier1_drm_hits += layer.tier1_drm_hits;
            target.tier2_uma_hits += layer.tier2_uma_hits;
            target.tier3_nvme_misses += layer.tier3_nvme_misses;
            let lookups = layer.lookups_total as f64;
            stall_weighted[index] += layer.mean_prefetch_stall_us * lookups;
            gate_weighted[index] += layer.mean_gate_alpha * lookups;
        }
        transport.rocenante_oneshot_calls += stats.transport.rocenante_oneshot_calls;
        transport.nccl_ring_calls += stats.transport.nccl_ring_calls;
        transport.rail0_rocep1s0f0_mb += stats.transport.rail0_rocep1s0f0_mb;
        transport.rail1_rocep2p1s0f0_mb += stats.transport.rail1_rocep2p1s0f0_mb;
        transport.l2_weight_prefetch_overlaps += stats.transport.l2_weight_prefetch_overlaps;
        ttft_ms = ttft_ms.max(stats.timing.ttft_ms);
        min_uma_free_gib = min_uma_free_gib.min(stats.timing.min_rank_uma_free_gib);
        let prompt_tokens = usage.prompt_tokens as f64;
        let completion_tokens = usage.completion_tokens as f64;
        if stats.timing.prefill_tok_per_s > 0.0 {
            prefill_seconds += prompt_tokens / stats.timing.prefill_tok_per_s;
        }
        if stats.timing.decode_tok_per_s > 0.0 {
            decode_seconds += completion_tokens / stats.timing.decode_tok_per_s;
        }
    }
    for (index, layer) in engram_layers.iter_mut().enumerate() {
        layer.hit_rate = safe_ratio(
            (layer.tier1_drm_hits + layer.tier2_uma_hits) as usize,
            layer.lookups_total as usize,
        );
        if layer.lookups_total > 0 {
            let lookups = layer.lookups_total as f64;
            layer.mean_prefetch_stall_us = stall_weighted[index] / lookups;
            layer.mean_gate_alpha = gate_weighted[index] / lookups;
        }
    }
    let dspark_speculative = DSparkStepStats::from_counts(
        parts
            .iter()
            .map(|(stats, _)| stats.dspark_speculative.target_verify_steps)
            .sum(),
        parts
            .iter()
            .map(|(stats, _)| stats.dspark_speculative.draft_tokens_proposed)
            .sum(),
        parts
            .iter()
            .map(|(stats, _)| stats.dspark_speculative.draft_tokens_confidence_gated)
            .sum(),
        parts
            .iter()
            .map(|(stats, _)| stats.dspark_speculative.draft_tokens_accepted)
            .sum(),
    );
    let kv_cache_hit_rate = safe_ratio(
        kv_cache.global_kv_cached_tokens,
        kv_cache.total_prompt_tokens,
    );
    Box::new(DeepSeekSparkRequestStats {
        kv_cache,
        kv_cache_hit_rate,
        engram_layers,
        dspark_speculative,
        transport,
        timing: crate::telemetry_types::RequestTimingStats {
            ttft_ms,
            prefill_tok_per_s: if prefill_seconds > 0.0 {
                (parts
                    .iter()
                    .map(|(_, usage)| usage.prompt_tokens)
                    .sum::<usize>() as f64)
                    / prefill_seconds
            } else {
                0.0
            },
            decode_tok_per_s: if decode_seconds > 0.0 {
                (parts
                    .iter()
                    .map(|(_, usage)| usage.completion_tokens)
                    .sum::<usize>() as f64)
                    / decode_seconds
            } else {
                0.0
            },
            min_rank_uma_free_gib: if min_uma_free_gib.is_finite() {
                min_uma_free_gib
            } else {
                0.0
            },
        },
    })
}

fn completion_chunk(
    id: &str,
    model: &str,
    created: u64,
    index: usize,
    text: &str,
    finish_reason: Option<FinishReason>,
    logprobs: Option<Value>,
) -> Value {
    let mut choice = json!({"index": index, "text": text, "finish_reason": finish_reason});
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

#[allow(clippy::too_many_arguments)] // the stream assembly's inputs, one argument each
fn completions_stream_response(
    state: &AppState,
    requests: Vec<EngineRequest>,
    prompts: &[CompletionPrompt],
    request_plan: &[(usize, usize)],
    n: usize,
    context: StreamContext,
    logprob_info: LogprobInfo,
    echo: bool,
    trace: Option<TraceDirective>,
) -> Response {
    let engine = Arc::clone(&state.engine);
    let metrics = Arc::clone(&state.metrics);
    let prompt_texts: Vec<String> = prompts.iter().map(|prompt| prompt.text.clone()).collect();
    let request_plan: Vec<(usize, usize)> = request_plan.to_vec();
    let StreamContext {
        id,
        created,
        model,
        include_usage,
        trace_id,
        ..
    } = context;
    // One generator per engine request, tagged with its request index; request_plan maps it back
    // to (prompt, choice), so the wire choice index is prompt_index * n + choice_index. That is
    // vLLM's shape for a batch: merge_async_iterators feeds one result generator per prompt, and
    // every one of the prompt's n choices is a separate streamed choice.
    let streams: Vec<futures::stream::BoxStream<'static, (usize, EngineEvent)>> = requests
        .into_iter()
        .enumerate()
        .map(|(index, request)| {
            let engine = Arc::clone(&engine);
            let stream = async_stream::stream! {
                let mut events = engine.generate(request);
                while let Some(event) = events.next().await {
                    yield (index, event);
                }
            };
            Box::pin(stream) as futures::stream::BoxStream<'static, (usize, EngineEvent)>
        })
        .collect();
    let mut events = futures::stream::select_all(streams);
    let stream = async_stream::stream! {
        // The spans are emitted while this generator is polled, so the binding has to be
        // taken here rather than on the handler's stack.
        let _binding = RequestTraceBinding::new(trace_id);
        let request_count = request_plan.len();
        // With logprobs requested the chunk has to carry the text the logprob belongs to, so
        // content deltas are held until their TokenLogprobs event arrives. All per-sequence stream
        // state is indexed by the engine request, not by the wire choice index.
        let mut buffered: Vec<String> = (0..request_count).map(|_| String::new()).collect();
        let mut prompt_logprobs: Vec<Option<PromptLogprobs>> =
            (0..request_count).map(|_| None).collect();
        // vLLM starts a streamed chunk's text_offset at the length of the text it has already
        // sent, not at zero for every chunk.
        let mut emitted_chars = vec![0usize; request_count];
        let mut echo_sent = vec![false; request_count];
        let mut usage_parts: Vec<(usize, usize, OpenAiUsageWithCache)> =
            Vec::with_capacity(request_count);
        let mut stats_parts: Vec<(DeepSeekSparkRequestStats, OpenAiUsageWithCache)> =
            Vec::with_capacity(request_count);
        while let Some((request_index, event)) = events.next().await {
            let (prompt_index, choice_index) = request_plan[request_index];
            let chunk_index = prompt_index * n + choice_index;
            match event {
                EngineEvent::Started { .. } => {}
                EngineEvent::PromptLogprobs(payload) => {
                    prompt_logprobs[request_index] = Some(*payload)
                }
                // A completion has no reasoning field: the reasoning channel is text here.
                EngineEvent::ReasoningDelta(text) | EngineEvent::ContentDelta(text) => {
                    if logprob_info.wants_sample() {
                        buffered[request_index].push_str(&text);
                    } else {
                        emitted_chars[request_index] += text.chars().count();
                        yield Ok(sse_frame(&completion_chunk(
                            &id,
                            &model,
                            created,
                            chunk_index,
                            &text,
                            None,
                            None,
                        )));
                    }
                }
                EngineEvent::TokenLogprobs(logprob) => {
                    if logprob_info.wants_sample() {
                        let mut positions = Vec::new();
                        let text = if echo && !echo_sent[request_index] {
                            echo_sent[request_index] = true;
                            if let Some(prompt) = prompt_logprobs[request_index].as_ref() {
                                positions.extend(prompt_positions(prompt));
                            }
                            format!(
                                "{}{}",
                                prompt_texts[request_index],
                                std::mem::take(&mut buffered[request_index])
                            )
                        } else {
                            std::mem::take(&mut buffered[request_index])
                        };
                        positions.push(CompletionPosition {
                            token: &logprob.token,
                            logprob: Some(&logprob),
                        });
                        let logprobs =
                            completion_logprobs_value(&positions, emitted_chars[request_index]);
                        emitted_chars[request_index] += text.chars().count();
                        yield Ok(sse_frame(&completion_chunk(
                            &id,
                            &model,
                            created,
                            chunk_index,
                            &text,
                            None,
                            Some(logprobs),
                        )));
                    }
                }
                EngineEvent::ToolCallDelta { .. } => {}
                EngineEvent::Finished {
                    finish_reason,
                    usage: part_usage,
                    stats,
                } => {
                    // The last text a logprobs request has (a token the engine did not logprob,
                    // or the echo prompt of an echo-without-generation stream) goes out once here.
                    if logprob_info.wants_sample() {
                        let mut positions = Vec::new();
                        let mut text = std::mem::take(&mut buffered[request_index]);
                        if echo && !echo_sent[request_index] {
                            echo_sent[request_index] = true;
                            if let Some(prompt) = prompt_logprobs[request_index].as_ref() {
                                positions.extend(prompt_positions(prompt));
                            }
                            text = format!("{}{text}", prompt_texts[request_index]);
                        }
                        if !text.is_empty() || !positions.is_empty() {
                            let logprobs = if positions.is_empty() {
                                None
                            } else {
                                Some(completion_logprobs_value(
                                    &positions,
                                    emitted_chars[request_index],
                                ))
                            };
                            yield Ok(sse_frame(&completion_chunk(
                                &id,
                                &model,
                                created,
                                chunk_index,
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
                        chunk_index,
                        "",
                        Some(finish_reason),
                        None,
                    )));
                    usage_parts.push((prompt_index, choice_index, part_usage));
                    stats_parts.push(((*stats).clone(), part_usage));
                }
                EngineEvent::Error { message, code } => {
                    yield Ok(sse_frame(&json!({
                        "error": {"message": message, "type": "server_error", "param": null, "code": code},
                    })));
                }
            }
        }
        // n choices share a prompt, so the prompt side is counted once (from each prompt's choice
        // 0) and the completion side is summed over every sequence. The hit/miss extension keeps
        // the pre-existing streaming behaviour, where those two fields stay zero until the buffered
        // body derives them.
        let mut usage = OpenAiUsageWithCache::default();
        for (_, choice_index, part_usage) in &usage_parts {
            if *choice_index == 0 {
                add_prompt_usage(&mut usage, part_usage);
            }
            add_completion_usage(&mut usage, part_usage);
        }
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
        // S6: the buffered body derives the DeepSeek hit/miss pair from the same cached-token
        // total, so the streaming usage chunk reports them instead of two zeros.
        usage.prompt_cache_hit_tokens = usage.prompt_tokens_details.cached_tokens;
        usage.prompt_cache_miss_tokens = usage
            .prompt_tokens
            .saturating_sub(usage.prompt_tokens_details.cached_tokens);
        if !stats_parts.is_empty() {
            let parts: Vec<(&DeepSeekSparkRequestStats, &OpenAiUsageWithCache)> = stats_parts
                .iter()
                .map(|(stats, part_usage)| (stats, part_usage))
                .collect();
            let stats = aggregate_stats_parts(&parts);
            metrics.record_request(&usage, &stats.kv_cache);
            metrics.record_request_stats(&stats);
        }
        if include_usage {
            // vLLM emits one final usage chunk after every choice has finished, with the usage
            // summed over the batch. A batched stream also carries the aggregate extension stats
            // here; a single-sequence stream keeps the pre-task-40 usage-only chunk byte for byte.
            let mut chunk = json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            });
            if request_count > 1 && !stats_parts.is_empty() {
                let parts: Vec<(&DeepSeekSparkRequestStats, &OpenAiUsageWithCache)> = stats_parts
                    .iter()
                    .map(|(stats, part_usage)| (stats, part_usage))
                    .collect();
                if let Some(map) = chunk.as_object_mut() {
                    map.insert(
                        "x_deepseek_spark_stats".to_owned(),
                        json!(aggregate_stats_parts(&parts)),
                    );
                }
            }
            yield Ok(sse_frame(&chunk));
        }
        yield Ok(sse_done());
    };
    sse_response(stream, trace, state.sse_keep_alive())
}

// ---------------------------------------------------------------------------------------
// Protocol adapters: Responses and Anthropic requests onto the chat template
// ---------------------------------------------------------------------------------------

/// A protocol request normalized onto the chat-completion shape the prompt seam renders.
struct ChatBridge {
    /// The raw chat body handed to PromptCodec::encode_chat.
    chat: Value,
    /// The same messages as typed chat messages, for the tokenizer-free router path.
    messages: Vec<ChatMessage>,
    /// Whether a non-empty tool list is rendered and a DSML answer may be lifted into calls.
    tools_offered: bool,
    /// Whether one answer may carry more than one tool call.
    parallel_tool_calls: bool,
}

fn chat_bridge(
    chat: Value,
    tools_offered: bool,
    parallel_tool_calls: bool,
) -> Result<ChatBridge, ApiError> {
    let messages = chat
        .get("messages")
        .cloned()
        .unwrap_or(Value::Array(Vec::new()));
    let messages: Vec<ChatMessage> = serde_json::from_value(messages).map_err(invalid_body)?;
    Ok(ChatBridge {
        chat,
        messages,
        tools_offered,
        parallel_tool_calls,
    })
}

fn unsupported_input_item(what: &str) -> ApiError {
    ApiError::invalid_request(
        format!(
            "{what} is not supported on this endpoint; this server renders chat messages, tool \
             calls and tool outputs, and refuses an item it cannot turn into a prompt rather than \
             silently dropping it"
        ),
        Some("input".to_owned()),
        Some("unsupported_capability".to_owned()),
    )
}

fn object_member<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    object.get(key).filter(|value| !value.is_null())
}

fn required_string(object: &Map<String, Value>, key: &str, item: &str) -> Result<String, ApiError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            ApiError::invalid_request(
                format!("{item} requires a non-empty string \"{key}\""),
                Some(key.to_owned()),
                Some("invalid_input".to_owned()),
            )
        })
}

/// Text of a Responses content-part array, for items that carry prose (instructions and reasoning).
fn responses_text_parts(value: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(items) = value.as_array() {
        for item in items {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                parts.push(text.to_owned());
            } else if let Some(text) = item.as_str() {
                parts.push(text.to_owned());
            }
        }
    } else if let Some(text) = value.as_str() {
        parts.push(text.to_owned());
    }
    parts.join("\n")
}

/// Converts one Responses content value (string or part array) to a chat content value.
fn responses_content_to_chat(value: &Value) -> Result<Value, ApiError> {
    match value {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Null => Ok(Value::Null),
        Value::Array(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for item in items {
                let Some(object) = item.as_object() else {
                    return Err(unsupported_input_item(
                        "a non-object Responses content part",
                    ));
                };
                let part_type = object.get("type").and_then(Value::as_str);
                match part_type {
                    Some("input_text" | "output_text" | "text") => {
                        let text = object
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        parts.push(json!({"type": "text", "text": text}));
                    }
                    Some("input_image" | "image_url") => {
                        let image = object
                            .get("image_url")
                            .cloned()
                            .or_else(|| object.get("url").cloned())
                            .ok_or_else(|| {
                                ApiError::invalid_request(
                                    "an image content part requires \"image_url\"",
                                    Some("input".to_owned()),
                                    Some("invalid_input".to_owned()),
                                )
                            })?;
                        // The Responses schema puts `detail` beside `image_url`, while the chat
                        // part carries it inside the image URL object; dropping it here would
                        // prepare a `low` image at full size.
                        let detail = object.get("detail").cloned();
                        let image = match (image, detail) {
                            (Value::String(url), Some(detail)) => {
                                json!({"url": url, "detail": detail})
                            }
                            (Value::Object(mut map), Some(detail)) => {
                                map.insert("detail".to_owned(), detail);
                                Value::Object(map)
                            }
                            (image, _) => image,
                        };
                        parts.push(json!({"type": "image_url", "image_url": image}));
                    }
                    Some("refusal") => {
                        let text = object
                            .get("refusal")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        parts.push(json!({"type": "text", "text": text}));
                    }
                    Some(other) => {
                        return Err(unsupported_input_item(&format!(
                            "the Responses content part type {other:?}"
                        )));
                    }
                    None => {
                        return Err(ApiError::invalid_request(
                            "a Responses content part requires a string \"type\"",
                            Some("input".to_owned()),
                            Some("invalid_input".to_owned()),
                        ));
                    }
                }
            }
            Ok(Value::Array(parts))
        }
        _ => Err(ApiError::invalid_request(
            "Responses content must be a string or an array of content parts",
            Some("input".to_owned()),
            Some("invalid_input".to_owned()),
        )),
    }
}

/// Appends one assistant function call, merging contiguous calls into one assistant message.
fn push_responses_function_call(
    messages: &mut Vec<Value>,
    object: &Map<String, Value>,
) -> Result<(), ApiError> {
    let call_id = required_string(object, "call_id", "a function_call item")?;
    let name = required_string(object, "name", "a function_call item")?;
    let arguments = match object.get("arguments") {
        Some(Value::String(text)) => text.clone(),
        Some(value @ (Value::Object(_) | Value::Array(_))) => value.to_string(),
        None | Some(Value::Null) => String::new(),
        Some(_) => {
            return Err(ApiError::invalid_request(
                "function_call arguments must be a JSON string or object",
                Some("arguments".to_owned()),
                Some("invalid_input".to_owned()),
            ));
        }
    };
    let call = json!({
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    });
    if let Some(last) = messages.last_mut().and_then(Value::as_object_mut)
        && last.get("role").and_then(Value::as_str) == Some("assistant")
        && last.get("tool_calls").is_some()
        && last.get("tool_call_id").is_none()
        && let Some(Value::Array(calls)) = last.get_mut("tool_calls")
    {
        calls.push(call);
        return Ok(());
    }
    messages.push(json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": [call],
    }));
    Ok(())
}

fn push_responses_reasoning(
    messages: &mut Vec<Value>,
    object: &Map<String, Value>,
) -> Result<(), ApiError> {
    let text = object
        .get("summary")
        .map(responses_text_parts)
        .filter(|text| !text.is_empty())
        .or_else(|| object.get("content").map(responses_text_parts))
        .unwrap_or_default();
    if text.is_empty() {
        return Ok(());
    }
    if let Some(last) = messages.last_mut().and_then(Value::as_object_mut)
        && last.get("role").and_then(Value::as_str) == Some("assistant")
        && last.get("tool_call_id").is_none()
        && last.get("tool_calls").is_none()
    {
        let existing = last
            .get("reasoning_content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let joined = if existing.is_empty() {
            text
        } else {
            format!("{existing}\n{text}")
        };
        last.insert("reasoning_content".to_owned(), Value::String(joined));
        return Ok(());
    }
    messages.push(json!({
        "role": "assistant",
        "content": Value::Null,
        "reasoning_content": text,
    }));
    Ok(())
}

fn responses_message_item(object: &Map<String, Value>) -> Result<Value, ApiError> {
    let role = required_string(object, "role", "a Responses message item")?;
    if !matches!(
        role.as_str(),
        "system" | "developer" | "user" | "assistant" | "tool"
    ) {
        return Err(ApiError::invalid_request(
            format!("invalid Responses message role {role:?}"),
            Some("role".to_owned()),
            Some("invalid_input".to_owned()),
        ));
    }
    // The released V4.1 encoder drops a developer message in thinking mode (the default) and
    // refuses it in chat mode. Fold it onto the system role, which the renderer supports at any
    // position, so the instruction reaches the model instead of disappearing.
    let role = if role == "developer" {
        "system".to_owned()
    } else {
        role
    };
    let content = match object.get("content") {
        None => Value::Null,
        Some(content) => responses_content_to_chat(content)?,
    };
    let mut message = Map::new();
    message.insert("role".to_owned(), Value::String(role));
    message.insert("content".to_owned(), content);
    for key in ["name", "tool_call_id", "tool_calls"] {
        if let Some(value) = object_member(object, key) {
            message.insert(key.to_owned(), value.clone());
        }
    }
    Ok(Value::Object(message))
}

fn responses_function_output(object: &Map<String, Value>) -> Result<Value, ApiError> {
    let call_id = required_string(object, "call_id", "a function_call_output item")?;
    let output = object.get("output").ok_or_else(|| {
        ApiError::invalid_request(
            "a function_call_output item requires \"output\"",
            Some("output".to_owned()),
            Some("invalid_input".to_owned()),
        )
    })?;
    let content = responses_content_to_chat(output)?;
    Ok(json!({"role": "tool", "tool_call_id": call_id, "content": content}))
}

/// Converts the Responses input member into chat messages.
fn responses_input_messages(input: Option<&Value>) -> Result<Vec<Value>, ApiError> {
    let mut messages = Vec::new();
    let Some(input) = input else {
        return Ok(messages);
    };
    match input {
        Value::Null => {}
        Value::String(text) => messages.push(json!({"role": "user", "content": text})),
        Value::Array(items) => {
            let mut pending_parts = Vec::new();
            let flush = |messages: &mut Vec<Value>, pending_parts: &mut Vec<Value>| {
                if !pending_parts.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": Value::Array(std::mem::take(pending_parts)),
                    }));
                }
            };
            for item in items {
                match item {
                    Value::String(text) => {
                        flush(&mut messages, &mut pending_parts);
                        messages.push(json!({"role": "user", "content": text}));
                    }
                    Value::Object(object) => match object.get("type").and_then(Value::as_str) {
                        Some("message") | None if object.contains_key("role") => {
                            flush(&mut messages, &mut pending_parts);
                            messages.push(responses_message_item(object)?);
                        }
                        Some("input_text" | "input_image" | "text" | "image_url") => {
                            let part =
                                responses_content_to_chat(&Value::Array(vec![item.clone()]))?;
                            if let Value::Array(mut parts) = part {
                                pending_parts.append(&mut parts);
                            }
                        }
                        Some("function_call") => {
                            flush(&mut messages, &mut pending_parts);
                            push_responses_function_call(&mut messages, object)?;
                        }
                        Some("function_call_output") => {
                            flush(&mut messages, &mut pending_parts);
                            messages.push(responses_function_output(object)?);
                        }
                        Some("reasoning") => {
                            flush(&mut messages, &mut pending_parts);
                            push_responses_reasoning(&mut messages, object)?;
                        }
                        Some("item_reference") => {
                            return Err(ApiError::invalid_request(
                                "item_reference input needs a response store, and this server keeps none; send the referenced content inline instead",
                                Some("input".to_owned()),
                                Some("unsupported_parameter".to_owned()),
                            ));
                        }
                        Some(other) => {
                            return Err(unsupported_input_item(&format!(
                                "the Responses input item type {other:?}"
                            )));
                        }
                        None => {
                            return Err(ApiError::invalid_request(
                                "a Responses input object requires \"type\" or \"role\"",
                                Some("input".to_owned()),
                                Some("invalid_input".to_owned()),
                            ));
                        }
                    },
                    _ => {
                        return Err(unsupported_input_item(
                            "a non-string, non-object input item",
                        ));
                    }
                }
            }
            flush(&mut messages, &mut pending_parts);
        }
        Value::Object(object) => match object.get("type").and_then(Value::as_str) {
            Some("input_text" | "input_image" | "text" | "image_url") => {
                let content = responses_content_to_chat(&Value::Array(vec![input.clone()]))?;
                messages.push(json!({"role": "user", "content": content}));
            }
            Some("message") | None if object.contains_key("role") => {
                messages.push(responses_message_item(object)?);
            }
            Some("function_call") => push_responses_function_call(&mut messages, object)?,
            Some("function_call_output") => {
                messages.push(responses_function_output(object)?);
            }
            Some("reasoning") => push_responses_reasoning(&mut messages, object)?,
            Some("item_reference") => {
                return Err(ApiError::invalid_request(
                    "item_reference input needs a response store, and this server keeps none; send the referenced content inline instead",
                    Some("input".to_owned()),
                    Some("unsupported_parameter".to_owned()),
                ));
            }
            Some(other) => {
                return Err(unsupported_input_item(&format!(
                    "the Responses input item type {other:?}"
                )));
            }
            None => {
                return Err(ApiError::invalid_request(
                    "a Responses input object requires \"type\" or \"role\"",
                    Some("input".to_owned()),
                    Some("invalid_input".to_owned()),
                ));
            }
        },
        _ => {
            return Err(unsupported_input_item(
                "a Responses input of this JSON type",
            ));
        }
    }
    Ok(messages)
}

fn responses_tools_to_chat(tools: Option<&Value>) -> Result<Option<Value>, ApiError> {
    let Some(tools) = tools.filter(|tools| !tools.is_null()) else {
        return Ok(None);
    };
    let Some(items) = tools.as_array() else {
        return Err(ApiError::invalid_request(
            "\"tools\" must be an array",
            Some("tools".to_owned()),
            Some("invalid_tools".to_owned()),
        ));
    };
    let mut converted = Vec::with_capacity(items.len());
    for item in items {
        let Some(object) = item.as_object() else {
            return Err(ApiError::invalid_request(
                "each tool must be an object",
                Some("tools".to_owned()),
                Some("invalid_tools".to_owned()),
            ));
        };
        if object.get("function").is_some() {
            converted.push(item.clone());
            continue;
        }
        let tool_type = object.get("type").and_then(Value::as_str);
        if !matches!(tool_type, Some("function") | None) {
            return Err(ApiError::invalid_request(
                format!(
                    "the Responses tool type {:?} is not supported: this server renders function \
                     tools only, so custom, hosted and built-in tools are refused rather than \
                     advertised without an implementation",
                    tool_type
                ),
                Some("tools".to_owned()),
                Some("unsupported_capability".to_owned()),
            ));
        }
        let name = required_string(object, "name", "a function tool")?;
        let mut function = Map::new();
        function.insert("name".to_owned(), Value::String(name));
        for key in ["description", "parameters", "strict"] {
            if let Some(value) = object_member(object, key) {
                function.insert(key.to_owned(), value.clone());
            }
        }
        converted.push(json!({"type": "function", "function": function}));
    }
    Ok(Some(Value::Array(converted)))
}

fn validate_responses_text_format(text: Option<&Value>) -> Result<(), ApiError> {
    let Some(text) = text.filter(|text| !text.is_null()) else {
        return Ok(());
    };
    let Some(object) = text.as_object() else {
        return Err(ApiError::invalid_request(
            "\"text\" must be an object",
            Some("text".to_owned()),
            Some("invalid_request".to_owned()),
        ));
    };
    let Some(format) = object.get("format").filter(|format| !format.is_null()) else {
        return Ok(());
    };
    let Some(format) = format.as_object() else {
        return Err(ApiError::invalid_request(
            "\"text.format\" must be an object",
            Some("text.format".to_owned()),
            Some("invalid_request".to_owned()),
        ));
    };
    match format.get("type").and_then(Value::as_str) {
        Some("text") | None => Ok(()),
        Some("json_object" | "json_schema") => Err(ApiError::invalid_request(
            "text.format json_object/json_schema needs constrained decoding, and this server has \
             no decoder logits processor; the request is refused rather than returning JSON that \
             may violate the caller's guarantee",
            Some("text.format".to_owned()),
            Some("unsupported_parameter".to_owned()),
        )),
        Some(other) => Err(ApiError::invalid_request(
            format!("unknown text.format type {other:?}"),
            Some("text.format".to_owned()),
            Some("invalid_request".to_owned()),
        )),
    }
}

/// The template spelling of a client's reasoning effort.
///
/// The released template understands low/high/xhigh/max/none. OpenAI also spells minimal and
/// medium; map them onto the template's nearest documented aliases.
fn reasoning_effort_alias(effort: &Value) -> Value {
    match effort.as_str() {
        Some("minimal") => json!("low"),
        Some("medium") => json!("high"),
        _ => effort.clone(),
    }
}

fn responses_bridge(value: &Value, request: &ResponsesRequest) -> Result<ChatBridge, ApiError> {
    if request.previous_response_id.is_some() {
        return Err(ApiError::invalid_request(
            "previous_response_id needs a response store, and this server keeps none; send the \
             full conversation in \"input\" instead",
            Some("previous_response_id".to_owned()),
            Some("unsupported_parameter".to_owned()),
        ));
    }
    if let Some(reasoning) = request.reasoning.as_ref().filter(|value| !value.is_null())
        && !reasoning.is_object()
    {
        return Err(ApiError::invalid_request(
            "\"reasoning\" must be an object with an optional effort member",
            Some("reasoning".to_owned()),
            Some("invalid_request".to_owned()),
        ));
    }
    validate_responses_text_format(request.text.as_ref())?;
    let policy = resolve_tool_policy(
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
    )?;
    let tools = responses_tools_to_chat(request.tools.as_ref())?;
    let mut messages = Vec::new();
    if let Some(instructions) = request.instructions.as_deref() {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    messages.extend(responses_input_messages(request.input.as_ref())?);
    let mut chat = Map::new();
    chat.insert("messages".to_owned(), Value::Array(messages));
    if policy.tools_offered
        && let Some(tools) = tools
    {
        chat.insert("tools".to_owned(), tools);
    }
    if policy.tools_offered
        && let Some(tool_choice) = request.tool_choice.clone()
    {
        chat.insert("tool_choice".to_owned(), tool_choice);
    }
    if let Some(parallel) = request.parallel_tool_calls {
        chat.insert("parallel_tool_calls".to_owned(), json!(parallel));
    }
    if let Some(effort) = request
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.get("effort"))
    {
        chat.insert(
            "reasoning_effort".to_owned(),
            reasoning_effort_alias(effort),
        );
    }
    for key in ["enable_thinking", "chat_template_kwargs"] {
        if let Some(value) = value.get(key).filter(|value| !value.is_null()) {
            chat.insert(key.to_owned(), value.clone());
        }
    }
    chat_bridge(Value::Object(chat), policy.tools_offered, policy.parallel)
}

/// Converts an Anthropic image source to a chat image URL.
fn anthropic_image_url(source: &Value) -> Result<String, ApiError> {
    let Some(source) = source.as_object() else {
        return Err(ApiError::invalid_request(
            "an Anthropic image source must be an object",
            Some("source".to_owned()),
            Some("invalid_input".to_owned()),
        ));
    };
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = required_string(source, "media_type", "an Anthropic image source")?;
            let data = required_string(source, "data", "an Anthropic image source")?;
            Ok(crate::multimodal::anthropic_source_to_data_url(
                &media_type,
                &data,
            ))
        }
        Some("url") => required_string(source, "url", "an Anthropic image source"),
        Some(other) => Err(unsupported_input_item(&format!(
            "the Anthropic image source type {other:?}"
        ))),
        None => Err(ApiError::invalid_request(
            "an Anthropic image source requires a string \"type\"",
            Some("source".to_owned()),
            Some("invalid_input".to_owned()),
        )),
    }
}

/// Converts an Anthropic content block array to chat parts, returning tool calls separately.
fn anthropic_content_to_chat(
    role: &str,
    content: Option<&Value>,
    messages: &mut Vec<Value>,
) -> Result<(), ApiError> {
    let Some(content) = content else {
        messages.push(json!({"role": role, "content": Value::Null}));
        return Ok(());
    };
    match content {
        Value::String(text) => messages.push(json!({"role": role, "content": text})),
        Value::Null => messages.push(json!({"role": role, "content": Value::Null})),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut reasoning = String::new();
            let flush_user = |messages: &mut Vec<Value>, parts: &mut Vec<Value>| {
                if !parts.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": Value::Array(std::mem::take(parts)),
                    }));
                }
            };
            for block in blocks {
                let Some(object) = block.as_object() else {
                    return Err(unsupported_input_item(
                        "a non-object Anthropic content block",
                    ));
                };
                let block_type = object.get("type").and_then(Value::as_str);
                match block_type {
                    Some("text") => {
                        let text = object
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        parts.push(json!({"type": "text", "text": text}));
                    }
                    Some("image") => {
                        let source = object.get("source").ok_or_else(|| {
                            ApiError::invalid_request(
                                "an Anthropic image block requires \"source\"",
                                Some("source".to_owned()),
                                Some("invalid_input".to_owned()),
                            )
                        })?;
                        let url = anthropic_image_url(source)?;
                        parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    }
                    Some("thinking") => {
                        if let Some(text) = object.get("thinking").and_then(Value::as_str) {
                            reasoning.push_str(text);
                        }
                    }
                    Some("redacted_thinking") => {}
                    Some("tool_use") => {
                        let id = required_string(object, "id", "an Anthropic tool_use block")?;
                        let name = required_string(object, "name", "an Anthropic tool_use block")?;
                        let input = object.get("input").cloned().unwrap_or_else(|| json!({}));
                        let arguments = input.to_string();
                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }));
                    }
                    Some("tool_result") => {
                        if role != "user" {
                            return Err(ApiError::invalid_request(
                                "tool_result blocks are only valid in user messages",
                                Some("messages".to_owned()),
                                Some("invalid_input".to_owned()),
                            ));
                        }
                        flush_user(messages, &mut parts);
                        let tool_use_id = required_string(
                            object,
                            "tool_use_id",
                            "an Anthropic tool_result block",
                        )?;
                        let result = object.get("content").cloned().unwrap_or(Value::Null);
                        let result = match result {
                            Value::Array(blocks) => {
                                let mut text: Vec<String> = Vec::new();
                                for block in blocks {
                                    let Some(block) = block.as_object() else {
                                        return Err(unsupported_input_item(
                                            "a non-object Anthropic tool_result content block",
                                        ));
                                    };
                                    match block.get("type").and_then(Value::as_str) {
                                        Some("text") => {
                                            let Some(text_value) =
                                                block.get("text").and_then(Value::as_str)
                                            else {
                                                return Err(ApiError::invalid_request(
                                                    "an Anthropic tool_result text block requires a string \"text\"",
                                                    Some("messages".to_owned()),
                                                    Some("invalid_input".to_owned()),
                                                ));
                                            };
                                            text.push(text_value.to_owned());
                                        }
                                        Some("image") => {
                                            let source = block.get("source").ok_or_else(|| {
                                                ApiError::invalid_request(
                                                    "an Anthropic tool_result image block requires \"source\"",
                                                    Some("source".to_owned()),
                                                    Some("invalid_input".to_owned()),
                                                )
                                            })?;
                                            text.push(anthropic_image_url(source)?);
                                        }
                                        Some(other) => {
                                            return Err(unsupported_input_item(&format!(
                                                "the Anthropic tool_result block type {other:?}"
                                            )));
                                        }
                                        None => {
                                            return Err(ApiError::invalid_request(
                                                "an Anthropic tool_result content block requires a string \"type\"",
                                                Some("messages".to_owned()),
                                                Some("invalid_input".to_owned()),
                                            ));
                                        }
                                    }
                                }
                                Value::String(text.join("\n"))
                            }
                            other => other,
                        };
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": result,
                        }));
                    }
                    Some(other) => {
                        return Err(unsupported_input_item(&format!(
                            "the Anthropic content block type {other:?}"
                        )));
                    }
                    None => {
                        return Err(ApiError::invalid_request(
                            "an Anthropic content block requires a string \"type\"",
                            Some("messages".to_owned()),
                            Some("invalid_input".to_owned()),
                        ));
                    }
                }
            }
            if role == "assistant" {
                if !parts.is_empty() || !tool_calls.is_empty() || !reasoning.is_empty() {
                    let content = if parts.is_empty() {
                        Value::Null
                    } else {
                        Value::Array(parts)
                    };
                    let mut message = Map::new();
                    message.insert("role".to_owned(), Value::String("assistant".to_owned()));
                    message.insert("content".to_owned(), content);
                    if !tool_calls.is_empty() {
                        message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
                    }
                    if !reasoning.is_empty() {
                        message.insert("reasoning_content".to_owned(), Value::String(reasoning));
                    }
                    messages.push(Value::Object(message));
                }
            } else {
                flush_user(messages, &mut parts);
            }
        }
        _ => {
            return Err(ApiError::invalid_request(
                "Anthropic message content must be a string or an array of blocks",
                Some("messages".to_owned()),
                Some("invalid_input".to_owned()),
            ));
        }
    }
    Ok(())
}

fn anthropic_system_text(system: Option<&Value>) -> Result<Option<String>, ApiError> {
    let Some(system) = system.filter(|system| !system.is_null()) else {
        return Ok(None);
    };
    match system {
        Value::String(text) => Ok(Some(text.clone())),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                let Some(object) = block.as_object() else {
                    return Err(ApiError::invalid_request(
                        "each Anthropic system block must be an object",
                        Some("system".to_owned()),
                        Some("invalid_input".to_owned()),
                    ));
                };
                match object.get("type").and_then(Value::as_str) {
                    Some("text") | None => {
                        if let Some(text) = object.get("text").and_then(Value::as_str) {
                            parts.push(text.to_owned());
                        }
                    }
                    Some(other) => {
                        return Err(unsupported_input_item(&format!(
                            "the Anthropic system block type {other:?}"
                        )));
                    }
                }
            }
            Ok(Some(parts.join("\n\n")))
        }
        _ => Err(ApiError::invalid_request(
            "Anthropic system must be a string or an array of text blocks",
            Some("system".to_owned()),
            Some("invalid_input".to_owned()),
        )),
    }
}

fn anthropic_tools_to_chat(tools: Option<&Value>) -> Result<Option<Value>, ApiError> {
    let Some(tools) = tools.filter(|tools| !tools.is_null()) else {
        return Ok(None);
    };
    let Some(items) = tools.as_array() else {
        return Err(ApiError::invalid_request(
            "\"tools\" must be an array",
            Some("tools".to_owned()),
            Some("invalid_tools".to_owned()),
        ));
    };
    let mut converted = Vec::with_capacity(items.len());
    for item in items {
        let Some(object) = item.as_object() else {
            return Err(ApiError::invalid_request(
                "each tool must be an object",
                Some("tools".to_owned()),
                Some("invalid_tools".to_owned()),
            ));
        };
        if object.get("function").is_some() {
            converted.push(item.clone());
            continue;
        }
        let name = required_string(object, "name", "an Anthropic tool")?;
        let mut function = Map::new();
        function.insert("name".to_owned(), Value::String(name));
        if let Some(description) = object_member(object, "description") {
            function.insert("description".to_owned(), description.clone());
        }
        if let Some(schema) = object_member(object, "input_schema") {
            function.insert("parameters".to_owned(), schema.clone());
        }
        converted.push(json!({"type": "function", "function": function}));
    }
    Ok(Some(Value::Array(converted)))
}

fn anthropic_tool_choice_to_chat(
    choice: Option<&Value>,
) -> Result<(Option<Value>, bool), ApiError> {
    let Some(choice) = choice.filter(|choice| !choice.is_null()) else {
        return Ok((None, true));
    };
    let Some(object) = choice.as_object() else {
        return Err(ApiError::invalid_request(
            "tool_choice must be an object",
            Some("tool_choice".to_owned()),
            Some("invalid_tool_choice".to_owned()),
        ));
    };
    let choice_type = object.get("type").and_then(Value::as_str);
    let parallel = !object
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let converted = match choice_type {
        Some("auto") => Value::String("auto".to_owned()),
        Some("none") => Value::String("none".to_owned()),
        Some("any") => Value::String("required".to_owned()),
        Some("tool") => {
            let name = required_string(object, "name", "an Anthropic tool_choice")?;
            json!({"type": "function", "function": {"name": name}})
        }
        _ => {
            return Err(ApiError::invalid_request(
                "tool_choice must be {\"type\":\"auto\"}, {\"type\":\"any\"}, {\"type\":\"tool\",\"name\":...} or {\"type\":\"none\"}",
                Some("tool_choice".to_owned()),
                Some("invalid_tool_choice".to_owned()),
            ));
        }
    };
    Ok((Some(converted), parallel))
}

fn anthropic_bridge(request: &MessagesRequest) -> Result<ChatBridge, ApiError> {
    let (tool_choice, parallel) = anthropic_tool_choice_to_chat(request.tool_choice.as_ref())?;
    let policy = resolve_tool_policy(request.tools.as_ref(), tool_choice.as_ref(), Some(parallel))?;
    let tools = anthropic_tools_to_chat(request.tools.as_ref())?;
    // The Messages API has no "system" role inside `messages`, but clients send one (Claude Code
    // among them). Its text joins the system prompt rather than failing the request, so a
    // conversation that carries instruction turns still renders.
    let mut system_parts = Vec::new();
    if let Some(system) = anthropic_system_text(request.system.as_ref())? {
        system_parts.push(system);
    }
    for message in request
        .messages
        .iter()
        .filter(|message| message.role == "system")
    {
        if let Some(text) = anthropic_system_text(message.content.as_ref())?
            && !text.is_empty()
        {
            system_parts.push(text);
        }
    }
    let mut messages = Vec::new();
    if !system_parts.is_empty() {
        messages.push(json!({"role": "system", "content": system_parts.join("\n\n")}));
    }
    for message in &request.messages {
        match message.role.as_str() {
            "user" | "assistant" => {
                anthropic_content_to_chat(&message.role, message.content.as_ref(), &mut messages)?;
            }
            "system" => {}
            other => {
                return Err(ApiError::invalid_request(
                    format!("invalid Anthropic message role {other:?}"),
                    Some("messages".to_owned()),
                    Some("invalid_input".to_owned()),
                ));
            }
        }
    }
    let mut chat = Map::new();
    chat.insert("messages".to_owned(), Value::Array(messages));
    if policy.tools_offered
        && let Some(tools) = tools
    {
        chat.insert("tools".to_owned(), tools);
    }
    if policy.tools_offered
        && let Some(tool_choice) = tool_choice
    {
        chat.insert("tool_choice".to_owned(), tool_choice);
    }
    chat.insert("parallel_tool_calls".to_owned(), json!(policy.parallel));
    if let Some(thinking) = request
        .thinking
        .as_ref()
        .filter(|thinking| !thinking.is_null())
    {
        let Some(thinking) = thinking.as_object() else {
            return Err(ApiError::invalid_request(
                "thinking must be an object",
                Some("thinking".to_owned()),
                Some("invalid_thinking".to_owned()),
            ));
        };
        match thinking.get("type").and_then(Value::as_str) {
            // "adaptive" is the current Anthropic mode: the model decides when and how much to
            // think, and the client asks for an amount through output_config.effort (carried
            // below). The V4.1 template has one switch, so adaptive means thinking on.
            Some("enabled" | "adaptive") => {
                chat.insert("enable_thinking".to_owned(), Value::Bool(true));
            }
            Some("disabled") => {
                chat.insert("enable_thinking".to_owned(), Value::Bool(false));
            }
            other => {
                return Err(ApiError::invalid_request(
                    format!(
                        "thinking.type must be \"enabled\", \"disabled\" or \"adaptive\", got {}",
                        other.map_or_else(
                            || "a non-string value".to_owned(),
                            |value| format!("{value:?}")
                        )
                    ),
                    Some("thinking".to_owned()),
                    Some("invalid_thinking".to_owned()),
                ));
            }
        }
    }
    if let Some(effort) = request
        .output_config
        .as_ref()
        .and_then(|config| config.get("effort"))
        .filter(|effort| !effort.is_null())
    {
        chat.insert(
            "reasoning_effort".to_owned(),
            reasoning_effort_alias(effort),
        );
    }
    chat_bridge(Value::Object(chat), policy.tools_offered, policy.parallel)
}

/// Sizes a bridged request with the checkpoint tokenizer, or the Milestone 1 estimator for the
/// weight-free router. Mirrors the chat handler's seam choice exactly.
fn bridge_prompt_summary(
    state: &AppState,
    bridge: &ChatBridge,
) -> Result<(PromptSummary, Option<Vec<u32>>), ApiError> {
    match state.tokenizer() {
        Some(seam) => {
            let ids = seam.encode_chat(&bridge.chat).map_err(prompt_codec_error)?;
            Ok((tokenizer_prompt_summary(ids.clone(), 0), Some(ids)))
        }
        None => Ok((summarize_chat_messages(&bridge.messages)?, None)),
    }
}

/// Response-appropriate ids for the tool calls of a buffered output.
fn response_call_ids(state: &AppState, calls: usize) -> Vec<(String, String)> {
    (0..calls)
        .map(|_| (state.next_id("fc_dsv41_"), state.next_id("call_dsv41_")))
        .collect()
}

fn responses_usage_value(usage: &OpenAiUsageWithCache) -> Value {
    json!({
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {"cached_tokens": usage.prompt_tokens_details.cached_tokens},
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.completion_tokens_details.reasoning_tokens},
        "total_tokens": usage.total_tokens,
    })
}

fn responses_message_item_value(message_id: &str, content: &str, status: &str) -> Value {
    json!({
        "type": "message",
        "id": message_id,
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    })
}

fn responses_reasoning_item_value(reasoning_id: &str, text: &str, status: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": reasoning_id,
        "status": status,
        "summary": [{"type": "summary_text", "text": text}],
    })
}

fn responses_function_call_item_value(
    item_id: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
    status: &str,
) -> Value {
    json!({
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    })
}

fn responses_output_items(
    output: &EngineOutput,
    reasoning_id: &str,
    message_id: &str,
    call_ids: &[(String, String)],
) -> Vec<Value> {
    let mut items = Vec::new();
    if !output.reasoning.is_empty() {
        items.push(responses_reasoning_item_value(
            reasoning_id,
            &output.reasoning,
            "completed",
        ));
    }
    if !output.content.is_empty() || output.tool_calls.is_empty() {
        items.push(responses_message_item_value(
            message_id,
            &output.content,
            "completed",
        ));
    }
    for (index, call) in output.tool_calls.iter().enumerate() {
        let (item_id, call_id) = call_ids.get(index).cloned().unwrap_or_else(|| {
            (
                format!("fc_dsv41_{index:08x}"),
                format!("call_dsv41_{index:08x}"),
            )
        });
        items.push(responses_function_call_item_value(
            &item_id,
            &call_id,
            &call.name,
            &call.arguments,
            "completed",
        ));
    }
    items
}

/// The Responses status/incomplete-details pair for a terminal finish reason.
fn responses_terminal_status(reason: FinishReason) -> (&'static str, Option<Value>) {
    match reason {
        FinishReason::Length => ("incomplete", Some(json!({"reason": "max_output_tokens"}))),
        FinishReason::Stop | FinishReason::ToolCalls | FinishReason::LoopAbort => {
            ("completed", None)
        }
    }
}

fn responses_response_body(
    output: &EngineOutput,
    id: &str,
    created: u64,
    model: &str,
    reasoning_id: &str,
    message_id: &str,
    call_ids: &[(String, String)],
) -> Value {
    let items = responses_output_items(output, reasoning_id, message_id, call_ids);
    let (status, incomplete_details) = responses_terminal_status(output.finish_reason);
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": Value::Null,
        "incomplete_details": incomplete_details,
        "model": model,
        "output": items,
        "output_text": output.content,
        "usage": responses_usage_value(&output.usage),
        "x_deepseek_spark_stats": output.stats,
    })
}

fn responses_snapshot(id: &str, created: u64, model: &str, status: &str, items: &[Value]) -> Value {
    let output_text = items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .flat_map(|item| {
            item.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<String>();
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": Value::Null,
        "incomplete_details": Value::Null,
        "model": model,
        "output": items,
        "output_text": output_text,
        "usage": Value::Null,
    })
}

fn responses_sse_event(kind: &str, mut data: Value, sequence: &mut u64) -> Bytes {
    if let Some(map) = data.as_object_mut() {
        map.insert("type".to_owned(), Value::String(kind.to_owned()));
        map.insert("sequence_number".to_owned(), json!(*sequence));
    }
    *sequence += 1;
    let mut text = format!("event: {kind}\ndata: ");
    text.push_str(&serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_owned()));
    text.push_str("\n\n");
    Bytes::from(text)
}

/// One open tool-call item in a Responses stream.
struct ResponsesCall {
    output_index: usize,
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
}

/// Streaming state shared by the content, reasoning and tool-call paths.
struct ResponsesStreamItems {
    next_output_index: usize,
    reasoning_index: Option<usize>,
    reasoning_text: String,
    message_index: Option<usize>,
    message_text: String,
    calls: Vec<ResponsesCall>,
    native_calls: BTreeMap<usize, usize>,
    parsed_calls: BTreeMap<usize, usize>,
}

impl ResponsesStreamItems {
    fn new() -> Self {
        Self {
            next_output_index: 0,
            reasoning_index: None,
            reasoning_text: String::new(),
            message_index: None,
            message_text: String::new(),
            calls: Vec::new(),
            native_calls: BTreeMap::new(),
            parsed_calls: BTreeMap::new(),
        }
    }

    fn output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn output(&self, reasoning_id: &str, message_id: &str) -> Vec<Value> {
        let mut items: Vec<(usize, Value)> = Vec::new();
        if let Some(index) = self.reasoning_index {
            items.push((
                index,
                responses_reasoning_item_value(reasoning_id, &self.reasoning_text, "completed"),
            ));
        }
        if let Some(index) = self.message_index {
            items.push((
                index,
                responses_message_item_value(message_id, &self.message_text, "completed"),
            ));
        }
        for call in &self.calls {
            items.push((
                call.output_index,
                responses_function_call_item_value(
                    &call.item_id,
                    &call.call_id,
                    &call.name,
                    &call.arguments,
                    "completed",
                ),
            ));
        }
        items.sort_by_key(|(index, _)| *index);
        items.into_iter().map(|(_, item)| item).collect()
    }
}

fn responses_open_message(
    state: &mut ResponsesStreamItems,
    message_id: &str,
    sequence: &mut u64,
) -> Vec<Bytes> {
    if state.message_index.is_some() {
        return Vec::new();
    }
    let index = state.output_index();
    state.message_index = Some(index);
    vec![
        responses_sse_event(
            "response.output_item.added",
            json!({
                "output_index": index,
                "item": {"type": "message", "id": message_id, "status": "in_progress",
                         "role": "assistant", "content": []},
            }),
            sequence,
        ),
        responses_sse_event(
            "response.content_part.added",
            json!({
                "item_id": message_id,
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }),
            sequence,
        ),
    ]
}

fn responses_content_frames(
    state: &mut ResponsesStreamItems,
    message_id: &str,
    text: &str,
    sequence: &mut u64,
) -> Vec<Bytes> {
    let mut frames = responses_open_message(state, message_id, sequence);
    state.message_text.push_str(text);
    let index = state.message_index.expect("the message item is open");
    frames.push(responses_sse_event(
        "response.output_text.delta",
        json!({
            "item_id": message_id,
            "output_index": index,
            "content_index": 0,
            "delta": text,
        }),
        sequence,
    ));
    frames
}

fn responses_reasoning_frames(
    state: &mut ResponsesStreamItems,
    reasoning_id: &str,
    text: &str,
    sequence: &mut u64,
) -> Vec<Bytes> {
    let mut frames = Vec::new();
    if state.reasoning_index.is_none() {
        let index = state.output_index();
        state.reasoning_index = Some(index);
        frames.push(responses_sse_event(
            "response.output_item.added",
            json!({
                "output_index": index,
                "item": {"type": "reasoning", "id": reasoning_id, "status": "in_progress",
                         "summary": []},
            }),
            sequence,
        ));
        frames.push(responses_sse_event(
            "response.reasoning_summary_part.added",
            json!({
                "item_id": reasoning_id,
                "output_index": index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": ""},
            }),
            sequence,
        ));
    }
    state.reasoning_text.push_str(text);
    let index = state.reasoning_index.expect("the reasoning item is open");
    frames.push(responses_sse_event(
        "response.reasoning_summary_text.delta",
        json!({
            "item_id": reasoning_id,
            "output_index": index,
            "summary_index": 0,
            "delta": text,
        }),
        sequence,
    ));
    frames
}

fn responses_open_parsed_call(
    state: &mut ResponsesStreamItems,
    next_id: &mut impl FnMut(&str) -> String,
    parser_index: usize,
    name: &str,
    parallel: bool,
    sequence: &mut u64,
) -> Vec<Bytes> {
    if !parallel && parser_index > 0 {
        return Vec::new();
    }
    if let Some(existing) = state.parsed_calls.get(&parser_index) {
        let call = &state.calls[*existing];
        return vec![responses_sse_event(
            "response.output_item.added",
            json!({
                "output_index": call.output_index,
                "item": {"type": "function_call", "id": call.item_id, "call_id": call.call_id,
                         "name": call.name, "arguments": "", "status": "in_progress"},
            }),
            sequence,
        )];
    }
    let output_index = state.output_index();
    let item_id = next_id("fc_dsv41_");
    let call_id = next_id("call_dsv41_");
    state.calls.push(ResponsesCall {
        output_index,
        item_id: item_id.clone(),
        call_id: call_id.clone(),
        name: name.to_owned(),
        arguments: String::new(),
    });
    state
        .parsed_calls
        .insert(parser_index, state.calls.len() - 1);
    vec![responses_sse_event(
        "response.output_item.added",
        json!({
            "output_index": output_index,
            "item": {"type": "function_call", "id": item_id, "call_id": call_id,
                     "name": name, "arguments": "", "status": "in_progress"},
        }),
        sequence,
    )]
}

fn responses_arguments_frame(
    state: &mut ResponsesStreamItems,
    call_index: usize,
    arguments_delta: &str,
    sequence: &mut u64,
) -> Option<Bytes> {
    let call = state.calls.get_mut(call_index)?;
    call.arguments.push_str(arguments_delta);
    Some(responses_sse_event(
        "response.function_call_arguments.delta",
        json!({
            "item_id": call.item_id,
            "output_index": call.output_index,
            "delta": arguments_delta,
        }),
        sequence,
    ))
}

fn responses_native_call_frames(
    state: &mut ResponsesStreamItems,
    next_id: &mut impl FnMut(&str) -> String,
    index: usize,
    name: Option<&str>,
    arguments_delta: &str,
    parallel: bool,
    sequence: &mut u64,
) -> Vec<Bytes> {
    if !parallel && index > 0 {
        return Vec::new();
    }
    let mut frames = Vec::new();
    let call_index = if let Some(existing) = state.native_calls.get(&index) {
        *existing
    } else {
        let output_index = state.output_index();
        let item_id = next_id("fc_dsv41_");
        let call_id = next_id("call_dsv41_");
        let name = name.unwrap_or_default().to_owned();
        state.calls.push(ResponsesCall {
            output_index,
            item_id: item_id.clone(),
            call_id: call_id.clone(),
            name: name.clone(),
            arguments: String::new(),
        });
        let call_index = state.calls.len() - 1;
        state.native_calls.insert(index, call_index);
        frames.push(responses_sse_event(
            "response.output_item.added",
            json!({
                "output_index": output_index,
                "item": {"type": "function_call", "id": item_id, "call_id": call_id,
                         "name": name, "arguments": "", "status": "in_progress"},
            }),
            sequence,
        ));
        call_index
    };
    if let Some(name) = name
        && let Some(call) = state.calls.get_mut(call_index)
        && call.name.is_empty()
    {
        call.name = name.to_owned();
    }
    if !arguments_delta.is_empty()
        && let Some(frame) = responses_arguments_frame(state, call_index, arguments_delta, sequence)
    {
        frames.push(frame);
    }
    frames
}

fn responses_finish_items(
    state: &ResponsesStreamItems,
    reasoning_id: &str,
    message_id: &str,
    sequence: &mut u64,
) -> Vec<Bytes> {
    let mut frames = Vec::new();
    if let Some(index) = state.reasoning_index {
        frames.push(responses_sse_event(
            "response.reasoning_summary_text.done",
            json!({
                "item_id": reasoning_id,
                "output_index": index,
                "summary_index": 0,
                "text": state.reasoning_text,
            }),
            sequence,
        ));
        frames.push(responses_sse_event(
            "response.reasoning_summary_part.done",
            json!({
                "item_id": reasoning_id,
                "output_index": index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": state.reasoning_text},
            }),
            sequence,
        ));
        frames.push(responses_sse_event(
            "response.output_item.done",
            json!({
                "output_index": index,
                "item": responses_reasoning_item_value(reasoning_id, &state.reasoning_text, "completed"),
            }),
            sequence,
        ));
    }
    if let Some(index) = state.message_index {
        frames.push(responses_sse_event(
            "response.output_text.done",
            json!({
                "item_id": message_id,
                "output_index": index,
                "content_index": 0,
                "text": state.message_text,
            }),
            sequence,
        ));
        frames.push(responses_sse_event(
            "response.content_part.done",
            json!({
                "item_id": message_id,
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": state.message_text, "annotations": []},
            }),
            sequence,
        ));
        frames.push(responses_sse_event(
            "response.output_item.done",
            json!({
                "output_index": index,
                "item": responses_message_item_value(message_id, &state.message_text, "completed"),
            }),
            sequence,
        ));
    }
    for call in &state.calls {
        frames.push(responses_sse_event(
            "response.function_call_arguments.done",
            json!({
                "item_id": call.item_id,
                "output_index": call.output_index,
                "arguments": call.arguments,
            }),
            sequence,
        ));
        frames.push(responses_sse_event(
            "response.output_item.done",
            json!({
                "output_index": call.output_index,
                "item": responses_function_call_item_value(
                    &call.item_id, &call.call_id, &call.name, &call.arguments, "completed"),
            }),
            sequence,
        ));
    }
    frames
}

fn responses_failed_event(
    id: &str,
    created: u64,
    model: &str,
    items: &[Value],
    code: &str,
    message: &str,
    sequence: &mut u64,
) -> Bytes {
    let mut response = responses_snapshot(id, created, model, "failed", items);
    if let Some(map) = response.as_object_mut() {
        map.insert(
            "error".to_owned(),
            json!({"code": code, "message": message}),
        );
        map.insert("usage".to_owned(), Value::Null);
    }
    responses_sse_event("response.failed", json!({"response": response}), sequence)
}

/// The context a Responses streaming answer needs.
struct ResponsesStreamContext {
    id: String,
    created: u64,
    model: String,
    reasoning_id: String,
    message_id: String,
    tools_offered: bool,
    parallel_tool_calls: bool,
    trace_id: Option<String>,
}

fn responses_stream_response(
    state: &AppState,
    request: EngineRequest,
    context: ResponsesStreamContext,
    trace: Option<TraceDirective>,
) -> Response {
    let engine = Arc::clone(&state.engine);
    let metrics = Arc::clone(&state.metrics);
    let counter = state.id_counter();
    let ResponsesStreamContext {
        id,
        created,
        model,
        reasoning_id,
        message_id,
        tools_offered,
        parallel_tool_calls,
        trace_id,
    } = context;
    let stream = async_stream::stream! {
        let _binding = RequestTraceBinding::new(trace_id);
        let mut sequence = 0u64;
        let mut next_id = {
            let counter = Arc::clone(&counter);
            move |prefix: &str| {
                format!(
                    "{prefix}{:08x}",
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                )
            }
        };
        let mut items = ResponsesStreamItems::new();
        let mut parser = tools_offered.then(tool_parser::ToolCallStreamParser::new);
        let mut parser_refusal: Option<String> = None;
        yield Ok::<Bytes, Infallible>(responses_sse_event(
            "response.created",
            json!({"response": responses_snapshot(&id, created, &model, "in_progress", &[])}),
            &mut sequence,
        ));
        yield Ok(responses_sse_event(
            "response.in_progress",
            json!({"response": responses_snapshot(&id, created, &model, "in_progress", &[])}),
            &mut sequence,
        ));
        let mut events = engine.generate(request);
        let mut finished = None;
        while let Some(event) = events.next().await {
            match event {
                EngineEvent::Started { .. }
                | EngineEvent::PromptLogprobs(_)
                | EngineEvent::TokenLogprobs(_) => {}
                EngineEvent::ReasoningDelta(text) => {
                    for frame in responses_reasoning_frames(
                        &mut items,
                        &reasoning_id,
                        &text,
                        &mut sequence,
                    ) {
                        yield Ok(frame);
                    }
                }
                EngineEvent::ContentDelta(text) => {
                    if let Some(parser) = parser.as_mut() {
                        if parser_refusal.is_none() {
                            match parser.feed(&text) {
                                Ok(events) => {
                                    for event in events {
                                        match event {
                                            tool_parser::StreamToolEvent::Content(text) => {
                                                for frame in responses_content_frames(
                                                    &mut items,
                                                    &message_id,
                                                    &text,
                                                    &mut sequence,
                                                ) {
                                                    yield Ok(frame);
                                                }
                                            }
                                            tool_parser::StreamToolEvent::CallStart { index, name } => {
                                                for frame in responses_open_parsed_call(
                                                    &mut items,
                                                    &mut next_id,
                                                    index,
                                                    &name,
                                                    parallel_tool_calls,
                                                    &mut sequence,
                                                ) {
                                                    yield Ok(frame);
                                                }
                                            }
                                            tool_parser::StreamToolEvent::ArgumentsDelta { index, arguments_delta } => {
                                                if let Some(call_index) = items.parsed_calls.get(&index).copied()
                                                    && let Some(frame) = responses_arguments_frame(
                                                        &mut items,
                                                        call_index,
                                                        &arguments_delta,
                                                        &mut sequence,
                                                    )
                                                {
                                                    yield Ok(frame);
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(refusal) => parser_refusal = Some(refusal.message),
                            }
                        }
                    } else {
                        for frame in responses_content_frames(
                            &mut items,
                            &message_id,
                            &text,
                            &mut sequence,
                        ) {
                            yield Ok(frame);
                        }
                    }
                }
                EngineEvent::ToolCallDelta {
                    index,
                    id: _,
                    name,
                    arguments_delta,
                } => {
                    for frame in responses_native_call_frames(
                        &mut items,
                        &mut next_id,
                        index,
                        name.as_deref(),
                        &arguments_delta,
                        parallel_tool_calls,
                        &mut sequence,
                    ) {
                        yield Ok(frame);
                    }
                }
                EngineEvent::Finished {
                    finish_reason,
                    usage,
                    stats,
                } => {
                    metrics.record_request(&usage, &stats.kv_cache);
                    metrics.record_request_stats(&stats);
                    finished = Some((finish_reason, usage, stats));
                }
                EngineEvent::Error { message, code } => {
                    let items = items.output(&reasoning_id, &message_id);
                    yield Ok(responses_failed_event(
                        &id,
                        created,
                        &model,
                        &items,
                        &code,
                        &message,
                        &mut sequence,
                    ));
                    return;
                }
            }
        }
        if let Some(parser) = parser.as_mut()
            && parser_refusal.is_none()
        {
            match parser.finish() {
                Ok(events) => {
                    for event in events {
                        match event {
                            tool_parser::StreamToolEvent::Content(text) => {
                                for frame in responses_content_frames(
                                    &mut items,
                                    &message_id,
                                    &text,
                                    &mut sequence,
                                ) {
                                    yield Ok(frame);
                                }
                            }
                            tool_parser::StreamToolEvent::CallStart { index, name } => {
                                for frame in responses_open_parsed_call(
                                    &mut items,
                                    &mut next_id,
                                    index,
                                    &name,
                                    parallel_tool_calls,
                                    &mut sequence,
                                ) {
                                    yield Ok(frame);
                                }
                            }
                            tool_parser::StreamToolEvent::ArgumentsDelta { index, arguments_delta } => {
                                if let Some(call_index) = items.parsed_calls.get(&index).copied()
                                    && let Some(frame) = responses_arguments_frame(
                                        &mut items,
                                        call_index,
                                        &arguments_delta,
                                        &mut sequence,
                                    )
                                {
                                    yield Ok(frame);
                                }
                            }
                        }
                    }
                }
                Err(refusal) => parser_refusal = Some(refusal.message),
            }
        }
        if let Some(message) = parser_refusal {
            let output = items.output(&reasoning_id, &message_id);
            yield Ok(responses_failed_event(
                &id,
                created,
                &model,
                &output,
                tool_parser::TOOL_STREAM_REFUSAL_CODE,
                &message,
                &mut sequence,
            ));
            return;
        }
        let Some((finish_reason, usage, stats)) = finished else {
            let output = items.output(&reasoning_id, &message_id);
            yield Ok(responses_failed_event(
                &id,
                created,
                &model,
                &output,
                "engine_stream_incomplete",
                "engine stream ended without a finish event",
                &mut sequence,
            ));
            return;
        };
        if items.message_index.is_none() && items.calls.is_empty() {
            for frame in responses_open_message(&mut items, &message_id, &mut sequence) {
                yield Ok(frame);
            }
        }
        for frame in responses_finish_items(&items, &reasoning_id, &message_id, &mut sequence) {
            yield Ok(frame);
        }
        let output = items.output(&reasoning_id, &message_id);
        let (status, incomplete_details) = responses_terminal_status(finish_reason);
        let mut response = responses_snapshot(&id, created, &model, status, &output);
        if let Some(map) = response.as_object_mut() {
            map.insert("usage".to_owned(), responses_usage_value(&usage));
            map.insert(
                "incomplete_details".to_owned(),
                incomplete_details.unwrap_or(Value::Null),
            );
            map.insert(
                "x_deepseek_spark_stats".to_owned(),
                serde_json::to_value(stats.as_ref()).unwrap_or(Value::Null),
            );
        }
        let event = if status == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        yield Ok(responses_sse_event(
            event,
            json!({"response": response}),
            &mut sequence,
        ));
    };
    sse_response(stream, trace, state.sse_keep_alive())
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
    let request: ResponsesRequest = serde_json::from_value(value.clone()).map_err(invalid_body)?;
    // A text-only engine cannot serve an image on any route, and the refusal must not depend on
    // whether the bridge understands the item that carries it, so it runs before the bridge.
    let capabilities = state.engine.capabilities();
    if image_carrier.is_some() && !capabilities.vision {
        refuse_image_carrier(image_carrier.clone(), false)?;
    }
    refuse_unbridged_image(&value, &["input"])?;
    let bridge = responses_bridge(&value, &request)?;
    if bridge.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "\"input\" must contain at least one message or item",
            Some("input".to_owned()),
            Some("invalid_messages".to_owned()),
        ));
    }
    let model = resolve_model(&state, request.model.as_deref())?;
    let trace = trace_context(&headers)?;
    let (mut summary, _prompt_ids) = bridge_prompt_summary(&state, &bridge)?;
    // The bridge renders every input_image as a chat image_url part, so a request with an image
    // takes the chat vision path (DECISIONS.md D59). A carrier the bridge left behind is a shape
    // this route does not read, refused by name.
    let mut vision = Vec::new();
    if image_carrier.is_some() && capabilities.vision {
        let (images, stripped) = take_chat_images(&bridge.chat)?;
        refuse_unread_image(&stripped)?;
        if images.is_empty() {
            refuse_image_carrier(image_carrier, capabilities.vision)?;
        }
        vision = prepare_and_place_images(&state, images, &mut summary).await?;
    } else {
        refuse_image_carrier(image_carrier, capabilities.vision)?;
    }
    let response_id = state.next_id("resp_dsv41_");
    let message_id = state.next_id("msg_dsv41_");
    let reasoning_id = state.next_id("rs_dsv41_");
    let created = state.now_unix();
    let trace_id = trace_request_id(&headers, &response_id);
    let prompt_tokens = summary.token_ids.len();
    let max_tokens = resolve_max_tokens_with_ceiling(
        None,
        "max_output_tokens",
        request.max_output_tokens,
        "max_output_tokens",
        spec,
        prompt_tokens,
        state.max_seq_len(),
        state.max_output_tokens(),
    )?;
    let engine_request = RequestBuilder {
        id: response_id.clone(),
        model: model.clone(),
        summary,
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: None,
        repetition_penalty: None,
        presence_penalty: None,
        frequency_penalty: None,
        skip_special_tokens: None,
        spec,
        stop: Vec::new(),
        seed: None,
        logprobs: LogprobsRequest::default(),
        trace,
        vision,
    }
    .build()?;
    state
        .engine
        .validate_request(&engine_request)
        .map_err(ApiError::from)?;
    if request.stream.unwrap_or(false) {
        return Ok(responses_stream_response(
            &state,
            engine_request,
            ResponsesStreamContext {
                id: response_id,
                created,
                model,
                reasoning_id,
                message_id,
                tools_offered: bridge.tools_offered,
                parallel_tool_calls: bridge.parallel_tool_calls,
                trace_id,
            },
            trace,
        ));
    }
    let _binding = RequestTraceBinding::new(trace_id);
    let mut output = drive_engine(state.engine.as_ref(), engine_request).await?;
    if bridge.tools_offered {
        apply_deepseek_v41_tool_parser(&mut output, &state);
    }
    if !bridge.parallel_tool_calls && output.tool_calls.len() > 1 {
        output.tool_calls.truncate(1);
        output.finish_reason = FinishReason::ToolCalls;
    }
    state
        .metrics
        .record_request(&output.usage, &output.stats.kv_cache);
    state.metrics.record_request_stats(&output.stats);
    let call_ids = response_call_ids(&state, output.tool_calls.len());
    Ok(json_response(
        responses_response_body(
            &output,
            &response_id,
            created,
            &model,
            &reasoning_id,
            &message_id,
            &call_ids,
        ),
        trace,
    ))
}

// ---------------------------------------------------------------------------------------
// Anthropic output assembly and streaming
// ---------------------------------------------------------------------------------------

fn anthropic_tool_input(arguments: &str) -> Value {
    // Anthropic's tool_use.input is always an object. A native engine can hand back an empty or
    // partial JSON fragment (for example a no-argument call before its closing brace), which must
    // not be serialized as a JSON string: clients decode this member as a map.
    match serde_json::from_str(arguments) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => Value::Object(Map::new()),
    }
}

fn anthropic_content_blocks(output: &EngineOutput, call_ids: &[String]) -> Vec<Value> {
    let mut content = Vec::new();
    if !output.reasoning.is_empty() {
        content.push(json!({
            "type": "thinking",
            "thinking": output.reasoning,
            "signature": "",
        }));
    }
    if !output.content.is_empty() || output.tool_calls.is_empty() {
        content.push(json!({"type": "text", "text": output.content}));
    }
    for (index, call) in output.tool_calls.iter().enumerate() {
        let id = call_ids
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("toolu_dsv41_{index:08x}"));
        content.push(json!({
            "type": "tool_use",
            "id": id,
            "name": call.name,
            "input": anthropic_tool_input(&call.arguments),
        }));
    }
    content
}

fn anthropic_response_body(
    output: &EngineOutput,
    id: &str,
    model: &str,
    call_ids: &[String],
) -> Value {
    let cache_read_input_tokens = output.usage.prompt_tokens_details.cached_tokens;
    let input_tokens = output
        .usage
        .prompt_tokens
        .saturating_sub(cache_read_input_tokens);
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": anthropic_content_blocks(output, call_ids),
        "stop_reason": if output.tool_calls.is_empty() {
            anthropic_stop_reason(output.finish_reason)
        } else {
            "tool_use"
        },
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output.usage.completion_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": 0,
        },
        "x_deepseek_spark_stats": output.stats,
    })
}

fn anthropic_sse_event(kind: &str, mut data: Value) -> Bytes {
    if let Some(map) = data.as_object_mut() {
        map.insert("type".to_owned(), Value::String(kind.to_owned()));
    }
    let mut text = format!("event: {kind}\ndata: ");
    text.push_str(&serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_owned()));
    text.push_str("\n\n");
    Bytes::from(text)
}

struct AnthropicCall {
    index: usize,
    id: String,
    name: String,
    input: String,
}

struct AnthropicStreamItems {
    next_index: usize,
    reasoning_index: Option<usize>,
    reasoning_text: String,
    message_index: Option<usize>,
    message_text: String,
    calls: Vec<AnthropicCall>,
    native_calls: BTreeMap<usize, usize>,
    parsed_calls: BTreeMap<usize, usize>,
}

impl AnthropicStreamItems {
    fn new() -> Self {
        Self {
            next_index: 0,
            reasoning_index: None,
            reasoning_text: String::new(),
            message_index: None,
            message_text: String::new(),
            calls: Vec::new(),
            native_calls: BTreeMap::new(),
            parsed_calls: BTreeMap::new(),
        }
    }

    fn index(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn stop_reason(&self, finish_reason: FinishReason) -> &'static str {
        if self.calls.is_empty() {
            anthropic_stop_reason(finish_reason)
        } else {
            "tool_use"
        }
    }
}

fn anthropic_reasoning_frames(state: &mut AnthropicStreamItems, text: &str) -> Vec<Bytes> {
    let mut frames = Vec::new();
    if state.reasoning_index.is_none() {
        let index = state.index();
        state.reasoning_index = Some(index);
        frames.push(anthropic_sse_event(
            "content_block_start",
            json!({
                "index": index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }),
        ));
    }
    state.reasoning_text.push_str(text);
    let index = state.reasoning_index.expect("the thinking block is open");
    frames.push(anthropic_sse_event(
        "content_block_delta",
        json!({"index": index, "delta": {"type": "thinking_delta", "thinking": text}}),
    ));
    frames
}

fn anthropic_text_frames(state: &mut AnthropicStreamItems, text: &str) -> Vec<Bytes> {
    let mut frames = Vec::new();
    if state.message_index.is_none() {
        let index = state.index();
        state.message_index = Some(index);
        frames.push(anthropic_sse_event(
            "content_block_start",
            json!({"index": index, "content_block": {"type": "text", "text": ""}}),
        ));
    }
    state.message_text.push_str(text);
    let index = state.message_index.expect("the text block is open");
    frames.push(anthropic_sse_event(
        "content_block_delta",
        json!({"index": index, "delta": {"type": "text_delta", "text": text}}),
    ));
    frames
}

fn anthropic_open_parsed_call(
    state: &mut AnthropicStreamItems,
    next_id: &mut impl FnMut(&str) -> String,
    parser_index: usize,
    name: &str,
    parallel: bool,
) -> Vec<Bytes> {
    if !parallel && parser_index > 0 {
        return Vec::new();
    }
    if let Some(existing) = state.parsed_calls.get(&parser_index) {
        let call = &state.calls[*existing];
        return vec![anthropic_sse_event(
            "content_block_start",
            json!({
                "index": call.index,
                "content_block": {"type": "tool_use", "id": call.id, "name": call.name, "input": {}},
            }),
        )];
    }
    let index = state.index();
    let id = next_id("toolu_dsv41_");
    state.calls.push(AnthropicCall {
        index,
        id: id.clone(),
        name: name.to_owned(),
        input: String::new(),
    });
    state
        .parsed_calls
        .insert(parser_index, state.calls.len() - 1);
    vec![anthropic_sse_event(
        "content_block_start",
        json!({
            "index": index,
            "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}},
        }),
    )]
}

fn anthropic_native_call_frames(
    state: &mut AnthropicStreamItems,
    next_id: &mut impl FnMut(&str) -> String,
    index: usize,
    name: Option<&str>,
    arguments_delta: &str,
    parallel: bool,
) -> Vec<Bytes> {
    if !parallel && index > 0 {
        return Vec::new();
    }
    let mut frames = Vec::new();
    let call_index = if let Some(existing) = state.native_calls.get(&index) {
        *existing
    } else {
        let block_index = state.index();
        let id = next_id("toolu_dsv41_");
        let name = name.unwrap_or_default().to_owned();
        state.calls.push(AnthropicCall {
            index: block_index,
            id: id.clone(),
            name: name.clone(),
            input: String::new(),
        });
        let call_index = state.calls.len() - 1;
        state.native_calls.insert(index, call_index);
        frames.push(anthropic_sse_event(
            "content_block_start",
            json!({
                "index": block_index,
                "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}},
            }),
        ));
        call_index
    };
    if let Some(name) = name
        && let Some(call) = state.calls.get_mut(call_index)
        && call.name.is_empty()
    {
        call.name = name.to_owned();
    }
    if !arguments_delta.is_empty() {
        let call = &mut state.calls[call_index];
        call.input.push_str(arguments_delta);
        frames.push(anthropic_sse_event(
            "content_block_delta",
            json!({
                "index": call.index,
                "delta": {"type": "input_json_delta", "partial_json": arguments_delta},
            }),
        ));
    }
    frames
}

fn anthropic_arguments_frame(
    state: &mut AnthropicStreamItems,
    call_index: usize,
    arguments_delta: &str,
) -> Option<Bytes> {
    let call = state.calls.get_mut(call_index)?;
    call.input.push_str(arguments_delta);
    Some(anthropic_sse_event(
        "content_block_delta",
        json!({
            "index": call.index,
            "delta": {"type": "input_json_delta", "partial_json": arguments_delta},
        }),
    ))
}

fn anthropic_finish_frames(
    state: &AnthropicStreamItems,
    finish_reason: FinishReason,
    usage: &OpenAiUsageWithCache,
    prompt_tokens: usize,
) -> Vec<Bytes> {
    let mut frames = Vec::new();
    if let Some(index) = state.reasoning_index {
        frames.push(anthropic_sse_event(
            "content_block_stop",
            json!({"index": index}),
        ));
    }
    if let Some(index) = state.message_index {
        frames.push(anthropic_sse_event(
            "content_block_stop",
            json!({"index": index}),
        ));
    }
    for call in &state.calls {
        frames.push(anthropic_sse_event(
            "content_block_stop",
            json!({"index": call.index}),
        ));
    }
    let cache_read = usage.prompt_tokens_details.cached_tokens;
    let input_tokens = prompt_tokens.saturating_sub(cache_read);
    frames.push(anthropic_sse_event(
        "message_delta",
        json!({
            "delta": {
                "stop_reason": state.stop_reason(finish_reason),
                "stop_sequence": null,
            },
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": usage.completion_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": 0,
            },
        }),
    ));
    frames.push(anthropic_sse_event("message_stop", json!({})));
    frames
}

struct AnthropicStreamContext {
    id: String,
    model: String,
    tools_offered: bool,
    parallel_tool_calls: bool,
    prompt_tokens: usize,
    trace_id: Option<String>,
}

fn messages_stream_response(
    state: &AppState,
    request: EngineRequest,
    context: AnthropicStreamContext,
    trace: Option<TraceDirective>,
) -> Response {
    let engine = Arc::clone(&state.engine);
    let metrics = Arc::clone(&state.metrics);
    let counter = state.id_counter();
    let AnthropicStreamContext {
        id,
        model,
        tools_offered,
        parallel_tool_calls,
        prompt_tokens,
        trace_id,
    } = context;
    let stream = async_stream::stream! {
        let _binding = RequestTraceBinding::new(trace_id);
        let mut next_id = {
            let counter = Arc::clone(&counter);
            move |prefix: &str| {
                format!(
                    "{prefix}{:08x}",
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                )
            }
        };
        let mut items = AnthropicStreamItems::new();
        let mut parser = tools_offered.then(tool_parser::ToolCallStreamParser::new);
        let mut parser_refusal: Option<String> = None;
        // The engine's first event carries the cache hit, which Anthropic's message_start usage
        // must partition the same way message_delta does. Read that event before starting the
        // message so the first usage block is not gross while the last one is net.
        let mut events = engine.generate(request);
        let mut cached_tokens = 0usize;
        let mut pending = None;
        if let Some(event) = events.next().await {
            match event {
                EngineEvent::Started {
                    cached_tokens: cached,
                    ..
                } => cached_tokens = cached,
                other => pending = Some(other),
            }
        }
        let input_tokens = prompt_tokens.saturating_sub(cached_tokens);
        yield Ok::<Bytes, Infallible>(anthropic_sse_event(
            "message_start",
            json!({
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                        "cache_read_input_tokens": cached_tokens,
                        "cache_creation_input_tokens": 0,
                    },
                }
            }),
        ));
        let mut finished = None;
        loop {
            let event = if let Some(event) = pending.take() {
                event
            } else {
                let Some(event) = events.next().await else {
                    break;
                };
                event
            };
            match event {
                EngineEvent::Started { .. }
                | EngineEvent::PromptLogprobs(_)
                | EngineEvent::TokenLogprobs(_) => {}
                EngineEvent::ReasoningDelta(text) => {
                    for frame in anthropic_reasoning_frames(&mut items, &text) {
                        yield Ok(frame);
                    }
                }
                EngineEvent::ContentDelta(text) => {
                    if let Some(parser) = parser.as_mut() {
                        if parser_refusal.is_none() {
                            match parser.feed(&text) {
                                Ok(events) => {
                                    for event in events {
                                        match event {
                                            tool_parser::StreamToolEvent::Content(text) => {
                                                for frame in anthropic_text_frames(&mut items, &text) {
                                                    yield Ok(frame);
                                                }
                                            }
                                            tool_parser::StreamToolEvent::CallStart { index, name } => {
                                                for frame in anthropic_open_parsed_call(
                                                    &mut items,
                                                    &mut next_id,
                                                    index,
                                                    &name,
                                                    parallel_tool_calls,
                                                ) {
                                                    yield Ok(frame);
                                                }
                                            }
                                            tool_parser::StreamToolEvent::ArgumentsDelta { index, arguments_delta } => {
                                                if let Some(call_index) = items.parsed_calls.get(&index).copied()
                                                    && let Some(frame) = anthropic_arguments_frame(
                                                        &mut items,
                                                        call_index,
                                                        &arguments_delta,
                                                    )
                                                {
                                                    yield Ok(frame);
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(refusal) => parser_refusal = Some(refusal.message),
                            }
                        }
                    } else {
                        for frame in anthropic_text_frames(&mut items, &text) {
                            yield Ok(frame);
                        }
                    }
                }
                EngineEvent::ToolCallDelta {
                    index,
                    id: _,
                    name,
                    arguments_delta,
                } => {
                    for frame in anthropic_native_call_frames(
                        &mut items,
                        &mut next_id,
                        index,
                        name.as_deref(),
                        &arguments_delta,
                        parallel_tool_calls,
                    ) {
                        yield Ok(frame);
                    }
                }
                EngineEvent::Finished {
                    finish_reason,
                    usage,
                    stats,
                } => {
                    metrics.record_request(&usage, &stats.kv_cache);
                    metrics.record_request_stats(&stats);
                    finished = Some((finish_reason, usage));
                }
                EngineEvent::Error { message, .. } => {
                    yield Ok(anthropic_sse_event(
                        "error",
                        json!({"error": {"type": "api_error", "message": message}}),
                    ));
                    return;
                }
            }
        }
        if let Some(parser) = parser.as_mut()
            && parser_refusal.is_none()
        {
            match parser.finish() {
                Ok(events) => {
                    for event in events {
                        match event {
                            tool_parser::StreamToolEvent::Content(text) => {
                                for frame in anthropic_text_frames(&mut items, &text) {
                                    yield Ok(frame);
                                }
                            }
                            tool_parser::StreamToolEvent::CallStart { index, name } => {
                                for frame in anthropic_open_parsed_call(
                                    &mut items,
                                    &mut next_id,
                                    index,
                                    &name,
                                    parallel_tool_calls,
                                ) {
                                    yield Ok(frame);
                                }
                            }
                            tool_parser::StreamToolEvent::ArgumentsDelta { index, arguments_delta } => {
                                if let Some(call_index) = items.parsed_calls.get(&index).copied()
                                    && let Some(frame) = anthropic_arguments_frame(
                                        &mut items,
                                        call_index,
                                        &arguments_delta,
                                    )
                                {
                                    yield Ok(frame);
                                }
                            }
                        }
                    }
                }
                Err(refusal) => parser_refusal = Some(refusal.message),
            }
        }
        if let Some(message) = parser_refusal {
            yield Ok(anthropic_sse_event(
                "error",
                json!({"error": {"type": "api_error", "message": message}}),
            ));
            return;
        }
        let Some((finish_reason, usage)) = finished else {
            yield Ok(anthropic_sse_event(
                "error",
                json!({"error": {"type": "api_error", "message": "engine stream ended without a finish event"}}),
            ));
            return;
        };
        if items.message_index.is_none()
            && items.calls.is_empty()
            && let Some(frame) = anthropic_text_frames(&mut items, "").into_iter().next()
        {
            yield Ok(frame);
        }
        for frame in anthropic_finish_frames(&items, finish_reason, &usage, prompt_tokens) {
            yield Ok(frame);
        }
    };
    sse_response(stream, trace, state.sse_keep_alive())
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
    let image_carrier = find_image_carrier(&value);
    let request: MessagesRequest = serde_json::from_value(value.clone()).map_err(invalid_body)?;
    // A text-only engine cannot serve an image on any route, and the refusal must not depend on
    // whether the bridge understands the block that carries it, so it runs before the bridge.
    let capabilities = state.engine.capabilities();
    if image_carrier.is_some() && !capabilities.vision {
        refuse_image_carrier(image_carrier.clone(), false)?;
    }
    if request.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "\"messages\" must be a non-empty array",
            Some("messages".to_owned()),
            Some("invalid_messages".to_owned()),
        ));
    }
    if request.max_tokens.is_none() {
        return Err(ApiError::invalid_request(
            "\"max_tokens\" is required by the Anthropic Messages API",
            Some("max_tokens".to_owned()),
            Some("invalid_max_tokens".to_owned()),
        ));
    }
    refuse_unbridged_image(&value, &["messages", "system"])?;
    let bridge = anthropic_bridge(&request)?;
    if bridge.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "\"messages\" must carry at least one non-empty user or assistant turn",
            Some("messages".to_owned()),
            Some("invalid_messages".to_owned()),
        ));
    }
    let spec = resolve_reasoning(&bridge.chat)?;
    let model = resolve_model(&state, request.model.as_deref())?;
    let trace = trace_context(&headers)?;
    let (mut summary, _prompt_ids) = bridge_prompt_summary(&state, &bridge)?;
    // The bridge renders every Anthropic image block as a chat image_url part, so a request with
    // an image takes the chat vision path (DECISIONS.md D59). A carrier the bridge left behind is
    // a shape this route does not read, refused by name.
    let mut vision = Vec::new();
    if image_carrier.is_some() && capabilities.vision {
        let (images, stripped) = take_chat_images(&bridge.chat)?;
        refuse_unread_image(&stripped)?;
        if images.is_empty() {
            refuse_image_carrier(image_carrier, capabilities.vision)?;
        }
        vision = prepare_and_place_images(&state, images, &mut summary).await?;
    } else {
        refuse_image_carrier(image_carrier, capabilities.vision)?;
    }
    let stop = request.stop_sequences.clone().unwrap_or_default();
    let id = state.next_id("msg_dsv41_");
    let trace_id = trace_request_id(&headers, &id);
    let prompt_tokens = summary.token_ids.len();
    let max_tokens = resolve_max_tokens_for_messages_with_ceiling(
        None,
        "max_tokens",
        request.max_tokens,
        "max_tokens",
        spec,
        prompt_tokens,
        state.max_seq_len(),
        state.max_output_tokens(),
    )?;
    let engine_request = RequestBuilder {
        id: id.clone(),
        model: model.clone(),
        summary,
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        repetition_penalty: None,
        presence_penalty: None,
        frequency_penalty: None,
        skip_special_tokens: None,
        spec,
        stop,
        seed: None,
        logprobs: LogprobsRequest::default(),
        trace,
        vision,
    }
    .build()?;
    state
        .engine
        .validate_request(&engine_request)
        .map_err(ApiError::from)?;
    if request.stream.unwrap_or(false) {
        return Ok(messages_stream_response(
            &state,
            engine_request,
            AnthropicStreamContext {
                id,
                model,
                tools_offered: bridge.tools_offered,
                parallel_tool_calls: bridge.parallel_tool_calls,
                prompt_tokens,
                trace_id,
            },
            trace,
        ));
    }
    let _binding = RequestTraceBinding::new(trace_id);
    let mut output = drive_engine(state.engine.as_ref(), engine_request).await?;
    if bridge.tools_offered {
        apply_deepseek_v41_tool_parser(&mut output, &state);
    }
    if !bridge.parallel_tool_calls && output.tool_calls.len() > 1 {
        output.tool_calls.truncate(1);
        output.finish_reason = FinishReason::ToolCalls;
    }
    state
        .metrics
        .record_request(&output.usage, &output.stats.kv_cache);
    state.metrics.record_request_stats(&output.stats);
    let call_ids: Vec<String> = (0..output.tool_calls.len())
        .map(|_| state.next_id("toolu_dsv41_"))
        .collect();
    Ok(json_response(
        anthropic_response_body(&output, &id, &model, &call_ids),
        trace,
    ))
}

fn anthropic_stop_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::LoopAbort => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ToolCalls => "tool_use",
    }
}

// ---------------------------------------------------------------------------------------
// GET /health, /readyz, /v1/models, /v1/cluster/stats, /v1/admin/trace/dump, /metrics
// ---------------------------------------------------------------------------------------

/// `GET /health`: vLLM's liveness probe, a 200 with an empty body while the engine can serve.
///
/// vLLM answers 200 when its engine client is healthy and 503 when it is dead
/// (`vllm/entrypoints/serve/instrumentator/health.py`). Here the engine says
/// ([`InferenceEngine::health`]): a 503 carries its reason as plain text (a loop that failed, or
/// one that has made no progress for too long).
async fn health(State(state): State<AppState>) -> Response {
    match state.engine.health() {
        Ok(()) => Response::new(Body::empty()),
        Err(reason) => unavailable(reason),
    }
}

/// `GET /readyz`: 200 while accepting requests, 503 once the drain signal was received or while
/// the engine cannot serve (with its reason, as `/health` gives it).
///
/// Readiness is separate from liveness on purpose: during a graceful drain the process is still
/// alive and finishes its in-flight requests (so `/health` stays 200), but an orchestrator or load
/// balancer should stop sending it new work as soon as `/readyz` turns 503. The route answers
/// from flags, so it also works while the engine is busy and never blocks on a request.
async fn readyz(State(state): State<AppState>) -> Response {
    if let Err(reason) = state.engine.health() {
        return unavailable(reason);
    }
    let mut response = Response::new(Body::empty());
    if state.is_draining() {
        *response.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
    }
    response
}

/// A probe's 503 with its reason as plain text.
fn unavailable(reason: String) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        reason,
    )
        .into_response()
}

async fn list_models(State(state): State<AppState>) -> Response {
    let card = model_card(&state, state.now_unix());
    Json(json!({"object": "list", "data": [card]})).into_response()
}

/// `GET /v1/models/{model}`: one model card, or the OpenAI 404 envelope.
///
/// The reference vLLM build answers this route with FastAPI's plain `{"detail":"Not Found"}` 404
/// (its `models/api_router.py` in that image mounts only the list route), while the OpenAI API
/// documents the retrieve route; this server implements the documented OpenAI behaviour, which is a
/// superset of what the reference serves. `owned_by` stays `sparkflash`, the same intentional
/// divergence the list route carries.
async fn retrieve_model(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Response, ApiError> {
    // Axum's rest-of-path capture keeps the separating slash.
    let model = model.trim_start_matches('/');
    if !model_matches(model, state.model_id(), state.model_aliases()) {
        return Err(ApiError::model_not_found(model));
    }
    Ok(Json(model_card(&state, state.now_unix())).into_response())
}

/// One `/v1/models` card: shared by the list and retrieve routes so the two cannot drift.
fn model_card(state: &AppState, created: u64) -> Value {
    // Most of the capability line is the engine's own answer, because a server must not advertise a
    // channel its engine cannot serve: `tools`, `thinking` and the reasoning-effort schema come from
    // `Capabilities`, `tool_choice` rides along with tools, and the schema is published only when
    // the engine honours it.
    //
    // `multimodal` and `image_url` are `Capabilities::vision`: true only for an engine with the
    // vision tower loaded, the same fact that decides whether the chat route prepares an image or
    // refuses it (DECISIONS.md D59).
    let capabilities = state.engine.capabilities();
    let endpoints = vec![
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/models/{model}",
        "/tokenize",
        "/detokenize",
        "/v1/cluster/stats",
        "/v1/admin/trace",
        "/health",
        "/readyz",
        "/metrics",
    ];
    let mut advertised = json!({
        "streaming": true,
        "tools": capabilities.tools,
        "tool_choice": capabilities.tools,
        "multimodal": capabilities.vision,
        "image_url": capabilities.vision,
        "thinking": capabilities.thinking,
        "chat_template_kwargs": if capabilities.thinking { json!(["thinking"]) } else { json!([]) },
        "endpoints": endpoints,
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
    json!({
        "id": state.model_id(),
        "object": "model",
        "created": created,
        "owned_by": MODEL_OWNED_BY,
        // vLLM's ModelCard also publishes these four members (root, parent, max_model_len and
        // an empty permission list); OpenAI clients read `max_model_len` for context, so the
        // card carries them beside this server's own context_length/max_output_tokens names.
        "root": state.model_id(),
        "parent": Value::Null,
        "max_model_len": state.max_seq_len(),
        "permission": json!([]),
        "context_length": state.max_seq_len(),
        "max_output_tokens": state.max_output_tokens(),
        "fabric_mode": state.engine.cluster_stats().fabric_mode,
        "capabilities": advertised,
    })
}

async fn cluster_stats(State(state): State<AppState>) -> Response {
    Json(state.engine.cluster_stats()).into_response()
}

/// The tokenizer seam a token-level route needs, or the typed 503 that says why it is absent.
///
/// `sparkflash serve` always attaches one; the weight-free router tests (and the synthetic tiny
/// plan's decode path) do not, so the route has to say that instead of answering from the
/// Milestone 1 placeholder estimate, which would report ids that are not the checkpoint's.
fn require_tokenizer(state: &AppState) -> Result<Arc<dyn PromptCodec>, ApiError> {
    state.tokenizer().cloned().ok_or_else(|| {
        ApiError::service_unavailable(
            "this server was started without a tokenizer, so it cannot report checkpoint token ids;              start `sparkflash serve` with a tokenizer attached (the default is              reference/tokenizer.json)",
        )
    })
}

/// `POST /tokenize`: the checkpoint tokenizer's ids for a prompt or a rendered chat request.
///
/// Wire shape is the reference server's: `{count, max_model_len, tokens, token_strs}`. vLLM's
/// request union is `{model, prompt: str}` or `{model, messages: [...]}`; a list prompt is a 400 in
/// that schema, and when both fields are present the chat form wins (the request model's
/// `messages` validator runs first). `add_special_tokens` defaults to true - vLLM's default for
/// this route - and `return_token_strs` selects the per-id string array.
async fn tokenize(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let value = parse_json(&body)?;
    let _model = resolve_model(&state, value.get("model").and_then(Value::as_str))?;
    let seam = require_tokenizer(&state)?;
    let add_special_tokens = value
        .get("add_special_tokens")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let return_token_strs = value
        .get("return_token_strs")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let tokens = if value
        .get("messages")
        .is_some_and(|messages| !messages.is_null())
    {
        // The chat form tokenizes the rendered V4.1 template, so the ids are what
        // /v1/chat/completions would prefill.
        seam.encode_chat(&value).map_err(prompt_codec_error)?
    } else if let Some(prompt) = value.get("prompt").filter(|prompt| !prompt.is_null()) {
        let text = prompt.as_str().ok_or_else(|| {
            ApiError::invalid_request(
                "`prompt` must be a string; this route accepts one prompt or a `messages` array",
                Some("prompt".to_owned()),
                Some("invalid_prompt".to_owned()),
            )
        })?;
        seam.encode_text(text, add_special_tokens)
            .map_err(prompt_codec_error)?
    } else {
        return Err(ApiError::invalid_request(
            "the request must carry `prompt` (a string) or `messages` (an array)",
            None,
            Some("invalid_request".to_owned()),
        ));
    };
    let count = tokens.len();
    let token_strs = return_token_strs.then(|| {
        tokens
            .iter()
            .map(|id| seam.token_str(*id).map_or(Value::Null, Value::String))
            .collect::<Vec<_>>()
    });
    Ok(Json(json!({
        "count": count,
        "max_model_len": state.max_seq_len(),
        "tokens": tokens,
        "token_strs": token_strs,
    }))
    .into_response())
}

/// `POST /detokenize`: the checkpoint tokenizer's text for a token-id array.
///
/// The reference server answers `{prompt: ""}` for ids it cannot map (its tokenizer drops them), so
/// an undecodable array is rendered as the empty string here too instead of a made-up 4xx.
async fn detokenize(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let value = parse_json(&body)?;
    let _model = resolve_model(&state, value.get("model").and_then(Value::as_str))?;
    let seam = require_tokenizer(&state)?;
    let Some(tokens_value) = value.get("tokens") else {
        return Err(ApiError::invalid_request(
            "the request must carry `tokens` (an array of token ids)",
            Some("tokens".to_owned()),
            Some("invalid_tokens".to_owned()),
        ));
    };
    let tokens: Vec<u32> = serde_json::from_value(tokens_value.clone()).map_err(|_| {
        ApiError::invalid_request(
            "`tokens` must be an array of token ids",
            Some("tokens".to_owned()),
            Some("invalid_tokens".to_owned()),
        )
    })?;
    let prompt = seam.decode(&tokens).unwrap_or_default();
    Ok(Json(json!({"prompt": prompt})).into_response())
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

    fn spec(thinking: bool, effort: Option<u8>) -> ReasoningSpec {
        ReasoningSpec { thinking, effort }
    }

    #[test]
    fn max_tokens_defaults_follow_the_recipe_contract() {
        let context = 1_048_576;
        assert_eq!(
            resolve_max_tokens(
                None,
                "max_tokens",
                None,
                "max_tokens",
                spec(false, None),
                10,
                context
            )
            .unwrap(),
            DEFAULT_NON_THINKING_MAX_TOKENS
        );
        assert_eq!(
            resolve_max_tokens(
                None,
                "max_tokens",
                None,
                "max_tokens",
                spec(true, Some(50)),
                10,
                context
            )
            .unwrap(),
            DEFAULT_THINKING_MAX_TOKENS
        );
        assert_eq!(
            resolve_max_tokens(
                None,
                "max_tokens",
                None,
                "max_tokens",
                spec(true, Some(100)),
                10,
                context
            )
            .unwrap(),
            DEFAULT_MAX_EFFORT_MAX_TOKENS
        );
        assert_eq!(
            resolve_max_tokens(
                None,
                "max_tokens",
                Some(4096),
                "max_tokens",
                spec(true, Some(50)),
                10,
                context
            )
            .unwrap(),
            4096
        );
    }

    #[test]
    fn max_tokens_is_lowered_to_the_context_left_after_the_prompt() {
        // Explicit value lowered to the context remainder, not silently emitted past the window.
        assert_eq!(
            resolve_max_tokens(
                None,
                "max_tokens",
                Some(300_000),
                "max_tokens",
                spec(true, Some(50)),
                1_000,
                200_000
            )
            .unwrap(),
            199_000
        );
        // Default lowered too.
        assert_eq!(
            resolve_max_tokens(
                None,
                "max_tokens",
                None,
                "max_tokens",
                spec(false, None),
                10,
                100
            )
            .unwrap(),
            90
        );
        // A prompt that exactly fills the window still gets the recipe's minimum of 1.
        assert_eq!(
            resolve_max_tokens(
                None,
                "max_tokens",
                None,
                "max_tokens",
                spec(true, Some(50)),
                100,
                100
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn max_tokens_validation_is_explicit() {
        assert!(
            resolve_max_tokens(
                None,
                "max_tokens",
                Some(0),
                "max_tokens",
                spec(true, Some(50)),
                1,
                100
            )
            .is_err()
        );
        assert!(
            resolve_max_tokens(
                None,
                "max_tokens",
                Some(MODEL_MAX_OUTPUT_TOKENS + 1),
                "max_tokens",
                spec(true, Some(50)),
                1,
                1_000_000
            )
            .is_err()
        );
        assert!(
            resolve_max_tokens(
                None,
                "max_tokens",
                None,
                "max_tokens",
                spec(true, Some(50)),
                101,
                100
            )
            .is_err()
        );
    }

    #[test]
    fn anthropic_messages_lower_an_over_cap_max_tokens_instead_of_rejecting() {
        assert_eq!(
            resolve_max_tokens_for_messages(
                None,
                "max_tokens",
                Some(MODEL_MAX_OUTPUT_TOKENS + 1),
                "max_tokens",
                spec(true, Some(50)),
                10,
                1_048_576,
            )
            .unwrap(),
            MODEL_MAX_OUTPUT_TOKENS
        );
        assert_eq!(
            resolve_max_tokens_for_messages(
                None,
                "max_tokens",
                Some(300_000),
                "max_tokens",
                spec(true, Some(50)),
                1_000,
                200_000,
            )
            .unwrap(),
            199_000
        );
    }

    #[test]
    fn thinking_sampling_follows_the_labelled_v41_policy() {
        assert_eq!(resolve_sampling(Some(0.6), Some(0.7), true), (1.0, 0.95));
        assert_eq!(resolve_sampling(None, None, true), (1.0, 1.0));
        assert_eq!(resolve_sampling(Some(0.6), Some(0.7), false), (0.6, 0.7));
    }
}
