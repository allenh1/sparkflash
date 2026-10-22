//! End-to-end router tests: no GPU, no checkpoint, no sockets.
//!
//! Every case drives the real axum Router through tower::ServiceExt::oneshot against
//! MockEngine, so response bodies (including exact SSE framing) are asserted directly.

use std::io::Read as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::{Mutex, MutexGuard};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use futures::StreamExt as _;
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::device_chain::ClaimMode;
use sparkflash_api::device_token::{
    DeviceEndpoint, DeviceToken, DeviceTokenClaim, DeviceTokenRequest, DeviceTokenSource,
    DeviceTokenStream,
};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, InferenceEngine,
    LogprobAlternative, MockEngine, PromptLogprobs, TokenLogprob, TokenString,
};
use sparkflash_api::server::{
    DEVICE_TOKEN_CLAIM_HEADER, DEVICE_TOKEN_SOURCE_HEADER, DEVICE_TOKEN_SOURCE_UNMEASURED,
    disarm_traces, router,
};
use sparkflash_api::state::{AppState, SequentialChoiceSeeds};
use tower::ServiceExt as _;

const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
const FIXED_TIME: u64 = 1_700_000_000;

/// Trace arming touches process-global atomics, so the tracing tests are serialised.
fn trace_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn engine_with(pieces: &[&str]) -> MockEngine {
    // The released model serves every optional channel, and the mock's own default is the
    // conservative one, so a test that describes the real surface says so explicitly.
    MockEngine::new(MODEL)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .with_capabilities(Capabilities::ALL)
        .content(pieces)
}

fn app() -> Router {
    router(AppState::new(Arc::new(engine_with(&["Hello", " world"]))).with_fixed_time(FIXED_TIME))
}

fn app_with(engine: MockEngine) -> Router {
    router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME))
}

/// An engine that counts how many times generation was entered.
///
/// Used to show a refused request never reaches the model: the image path is refused at the seam,
/// so the dummy-grid accounting some earlier revision would have run has no way in.
#[derive(Clone, Default)]
struct RecordingEngine {
    calls: Arc<AtomicUsize>,
}

impl InferenceEngine for RecordingEngine {
    fn model_id(&self) -> &str {
        MODEL
    }

    fn generate(
        &self,
        _request: EngineRequest,
    ) -> futures::stream::BoxStream<'static, EngineEvent> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(futures::stream::empty())
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        ClusterStatsSnapshot::default()
    }
}

/// A scripted engine that also records that generation was entered.
///
/// `RecordingEngine` above is deliberately empty (a refusal test only needs "was the engine
/// entered"), so a test that expects HTTP 200 needs a stream that finishes. This wraps
/// `MockEngine`'s script and counts the calls.
struct RecordingMockEngine {
    calls: Arc<AtomicUsize>,
    inner: MockEngine,
}

impl RecordingMockEngine {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            inner: MockEngine::new(MODEL),
        }
    }

    /// A scripted host-twin engine with a non-empty answer, for the device-source fallback tests.
    fn with_content(pieces: &[&str]) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            inner: MockEngine::new(MODEL)
                .with_capabilities(Capabilities::ALL)
                .content(pieces),
        }
    }
}

impl InferenceEngine for RecordingMockEngine {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> futures::stream::BoxStream<'static, EngineEvent> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.generate(request)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.inner.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
}

/// A scripted engine that records the normalized seam request it was handed.
///
/// The other recorders count entries or keep MockEngine's script; this one keeps the whole
/// [`EngineRequest`] so a test can assert the parsed sampling parameters crossed the seam unchanged.
struct RecordingRequestEngine {
    requests: Arc<std::sync::Mutex<Vec<EngineRequest>>>,
    inner: MockEngine,
}

impl RecordingRequestEngine {
    fn new() -> Self {
        Self {
            requests: Arc::new(std::sync::Mutex::new(Vec::new())),
            inner: MockEngine::new(MODEL),
        }
    }
}

impl InferenceEngine for RecordingRequestEngine {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> futures::stream::BoxStream<'static, EngineEvent> {
        self.requests
            .lock()
            .expect("the request recorder is not poisoned")
            .push(request.clone());
        self.inner.generate(request)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.inner.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
}

async fn send(app: Router, request: Request<Body>) -> (StatusCode, HeaderMap, Vec<u8>) {
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes()
        .to_vec();
    (status, headers, bytes)
}

fn json_request(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).expect("json body")))
        .expect("request")
}

fn get_request(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("request")
}

fn body_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("body is JSON")
}

fn sse_frames(bytes: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE frame is JSON"))
        .collect()
}

fn chat_body(content: &str) -> Value {
    json!({"model": MODEL, "messages": [{"role": "user", "content": content}]})
}

const EXPECTED_STREAM: &str = concat!(
    r#"data: {"id":"chatcmpl-dsv41-00000000","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-ai/DeepSeek-V4.1-Flash","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
    "\n\n",
    r#"data: {"id":"chatcmpl-dsv41-00000000","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-ai/DeepSeek-V4.1-Flash","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
    "\n\n",
    r#"data: {"id":"chatcmpl-dsv41-00000000","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-ai/DeepSeek-V4.1-Flash","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}"#,
    "\n\n",
    r#"data: {"id":"chatcmpl-dsv41-00000000","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-ai/DeepSeek-V4.1-Flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"x_deepseek_spark_stats":{"kv_cache":{"total_prompt_tokens":8,"global_kv_cached_tokens":0,"uma_hit_tokens":0,"nvme_restored_tokens":0,"swa_tail_hit":false,"encoder_swa_replay_tokens":0,"uncached_suffix_tokens":8,"decoder_swa_replay_tokens":8},"kv_cache_hit_rate":0.0,"engram_layers":[{"layer_id":1,"lookups_total":0,"tier1_drm_hits":0,"tier2_uma_hits":0,"tier3_nvme_misses":0,"hit_rate":0.0,"mean_prefetch_stall_us":0.0,"mean_gate_alpha":0.0},{"layer_id":14,"lookups_total":0,"tier1_drm_hits":0,"tier2_uma_hits":0,"tier3_nvme_misses":0,"hit_rate":0.0,"mean_prefetch_stall_us":0.0,"mean_gate_alpha":0.0}],"dspark_speculative":{"target_verify_steps":0,"draft_tokens_proposed":0,"draft_tokens_confidence_gated":0,"draft_tokens_accepted":0,"mean_tokens_per_verify_step":0.0,"acceptance_rate":0.0},"transport":{"fabric_mode":"","rocenante_oneshot_calls":0,"nccl_ring_calls":0,"rail0_rocep1s0f0_mb":0.0,"rail1_rocep2p1s0f0_mb":0.0,"l2_weight_prefetch_overlaps":0},"timing":{"ttft_ms":0.0,"prefill_tok_per_s":0.0,"decode_tok_per_s":0.0,"min_rank_uma_free_gib":0.0}}}"#,
    "\n\n",
    r#"data: {"id":"chatcmpl-dsv41-00000000","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-ai/DeepSeek-V4.1-Flash","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":0,"uma_cached_tokens":0,"nvme_cached_tokens":0,"image_tokens":0},"completion_tokens_details":{"reasoning_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":8}}"#,
    "\n\n",
    r#"data: [DONE]"#,
    "\n\n",
);

#[tokio::test]
async fn streaming_chat_completions_sse_is_byte_exact() {
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let (status, headers, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers.get("content-type").expect("content type"),
        "text/event-stream"
    );
    let text = String::from_utf8(bytes).expect("utf8");
    eprintln!("SSE_BODY_START>>>{text}<<<SSE_BODY_END");
    assert_eq!(text, EXPECTED_STREAM);
}

#[tokio::test]
async fn an_idle_stream_is_kept_alive_with_comment_frames() {
    // Every event after the first waits 20 ms; a 5 ms keep-alive interval therefore has to fill
    // each engine gap with a comment frame.
    let engine = engine_with(&["Hello", " world"]).with_delay(Duration::from_millis(20));
    let state = AppState::new(Arc::new(engine))
        .with_fixed_time(FIXED_TIME)
        .with_sse_keep_alive(Some(Duration::from_millis(5)));
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    let (status, _, bytes) = send(router(state), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(bytes.clone()).expect("utf8");
    let frames: Vec<&str> = text
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .collect();
    let comments = frames.iter().filter(|frame| **frame == ": ping").count();
    assert!(
        comments > 0,
        "an idle engine gap should carry keep-alive comments: {text}"
    );
    assert_ne!(
        frames.first().copied(),
        Some(": ping"),
        "no comment before the first event: {text}"
    );
    assert_eq!(
        frames.last().copied(),
        Some("data: [DONE]"),
        "comments never follow the terminator: {text}"
    );
    assert!(
        !sse_frames(&bytes).is_empty(),
        "the data frames are untouched"
    );
}

#[tokio::test]
async fn the_keep_alive_frames_can_be_turned_off() {
    let engine = engine_with(&["Hello", " world"]).with_delay(Duration::from_millis(20));
    let state = AppState::new(Arc::new(engine))
        .with_fixed_time(FIXED_TIME)
        .with_sse_keep_alive(None);
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    let (status, _, bytes) = send(router(state), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(bytes).expect("utf8");
    assert!(!text.contains(": ping"), "keep-alive is opt-out: {text}");
}

#[tokio::test]
async fn streaming_emits_role_reasoning_content_tool_and_done_frames() {
    let engine = MockEngine::new(MODEL)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .reasoning(&["let me think "])
        .content(&["Hello", " world"])
        .tool_call("call_1", "get_weather", &["{\"loc", "ation\": \"SF\"}"])
        .with_finish_reason(sparkflash_api::engine::FinishReason::ToolCalls);
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    let (status, _, bytes) = send(
        app_with(engine),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(bytes).expect("utf8");
    assert!(text.ends_with("data: [DONE]\n\n"));
    let frames = sse_frames(text.as_bytes());
    assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        frames[1]["choices"][0]["delta"]["reasoning_content"],
        "let me think "
    );
    assert_eq!(frames[2]["choices"][0]["delta"]["content"], "Hello");
    assert_eq!(frames[3]["choices"][0]["delta"]["content"], " world");
    assert_eq!(
        frames[4]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        frames[4]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(
        frames[5]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        "ation\": \"SF\"}"
    );
    assert_eq!(frames[6]["choices"][0]["finish_reason"], "tool_calls");
    // No usage chunk was requested, but the stats block still rides the final chunk.
    assert!(frames[6].get("x_deepseek_spark_stats").is_some());
}

#[tokio::test]
async fn non_streaming_chat_completion_round_trip() {
    let (status, _, bytes) = send(
        app(),
        json_request("/v1/chat/completions", &chat_body("hi")),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["id"], "chatcmpl-dsv41-00000000");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["created"], FIXED_TIME);
    assert_eq!(body["model"], MODEL);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 8);
    assert_eq!(body["usage"]["completion_tokens"], 2);
    assert_eq!(body["usage"]["total_tokens"], 10);
    assert_eq!(body["usage"]["prompt_tokens_details"]["image_tokens"], 0);
    let stats = &body["x_deepseek_spark_stats"];
    assert_eq!(stats["kv_cache"]["total_prompt_tokens"], 8);
    assert_eq!(stats["kv_cache"]["uncached_suffix_tokens"], 8);
    assert_eq!(stats["engram_layers"].as_array().expect("layers").len(), 2);
}

#[tokio::test]
async fn non_streaming_tool_call_response_is_assembled() {
    let engine = engine_with(&["Hello"])
        .tool_call("call_1", "get_weather", &["{\"location\":", " \"SF\"}"])
        .with_finish_reason(sparkflash_api::engine::FinishReason::ToolCalls);
    let (status, _, bytes) = send(
        app_with(engine),
        json_request("/v1/chat/completions", &chat_body("hi")),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let call = &body["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["id"], "call_1");
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], "{\"location\": \"SF\"}");
}

#[tokio::test]
async fn reasoning_effort_accepts_integers_and_names() {
    for effort in [
        json!(1),
        json!(100),
        json!(42),
        json!("low"),
        json!("minimal"),
        json!("medium"),
        json!("high"),
        json!("xhigh"),
        json!("max"),
        json!(null),
    ] {
        let mut body = chat_body("hi");
        body["reasoning_effort"] = effort.clone();
        let (status, _, _) = send(app(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::OK, "effort {effort} should be accepted");
    }
}

#[tokio::test]
async fn reasoning_effort_rejects_out_of_range_and_unknown_values() {
    for effort in [json!(0), json!(101), json!(-1), json!("bogus"), json!(true)] {
        let mut body = chat_body("hi");
        body["reasoning_effort"] = effort.clone();
        let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "effort {effort} must be rejected"
        );
        let error = body_json(&bytes);
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert_eq!(error["error"]["param"], "reasoning_effort");
        assert!(error["error"]["message"].is_string());
    }
}

#[tokio::test]
async fn thinking_flags_are_accepted_and_validated() {
    let mut body = chat_body("hi");
    body["enable_thinking"] = json!(false);
    let (status, _, _) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);

    let mut body = chat_body("hi");
    body["chat_template_kwargs"] = json!({"thinking": "non-thinking"});
    body["enable_thinking"] = json!(true);
    let (status, _, _) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);

    let mut body = chat_body("hi");
    body["chat_template_kwargs"] = json!({"thinking": "maybe"});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["code"], 400);
}

#[tokio::test]
async fn an_image_request_is_refused_with_the_missing_capability_and_never_reaches_the_engine() {
    let engine = RecordingEngine::default();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    // Two shapes for the same route. The second is the regression: an Anthropic image block sent to
    // the chat route used to deserialise to `ContentPart::Other`, be dropped, and reach generation
    // with no image - HTTP 200 from the text tokens alone (critic finding on 8a1b99b).
    let bodies = [
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ]}],
        }),
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
            ]}],
        }),
    ];
    for body in bodies {
        let (status, _, bytes) =
            send(app.clone(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let error = body_json(&bytes);
        assert_eq!(error["error"]["code"], StatusCode::BAD_REQUEST.as_u16());
        assert_eq!(
            error["error"]["type"], "invalid_request_error",
            "a capability the engine lacks is a client error, not a server failure"
        );
        let message = error["error"]["message"].as_str().expect("message");
        assert!(message.contains("vision"), "names the channel: {message}");
        assert!(
            message.contains("263"),
            "names the unwired tensors: {message}"
        );
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the engine was never entered: no visual-token count, fabricated or otherwise, was produced"
    );
}

#[tokio::test]
async fn a_nested_responses_image_cannot_bypass_the_refusal() {
    let engine = RecordingEngine::default();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let body = json!({
        "model": MODEL,
        "input": [{
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}],
        }],
    });
    let (status, _, bytes) = send(app, json_request("/v1/responses", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(&bytes)["error"]["code"],
        StatusCode::BAD_REQUEST.as_u16()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the engine was never entered for a nested image"
    );
}

#[tokio::test]
async fn the_documented_screenshot_shapes_are_refused_without_entering_the_engine() {
    // Review round 3: the canonical OpenAI computer_call_output nests its image under output as
    // an object, not an array of content parts, and the documented screenshot type is
    // computer_screenshot; a file-reference screenshot has no URL at all. Each shape used to reach
    // generation with no ImagePart recorded. The structural detector in prompt.rs refuses all of
    // them, plus a case-variant type and a carrier at an arbitrary depth.
    let engine = RecordingEngine::default();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let bodies = [
        // The documented object output.
        json!({
            "model": MODEL,
            "input": [{
                "type": "computer_call_output",
                "call_id": "c1",
                "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,aGVsbG8="},
            }],
        }),
        // The same carrier inside an output array.
        json!({
            "model": MODEL,
            "input": [{
                "type": "computer_call_output",
                "call_id": "c1",
                "output": [{"type": "computer_screenshot", "image_url": "data:image/png;base64,aGVsbG8="}],
            }],
        }),
        // A file_id-only screenshot: there is no URL to look for, only the carrier.
        json!({
            "model": MODEL,
            "input": [{
                "type": "computer_call_output",
                "call_id": "c1",
                "output": {"type": "computer_screenshot", "file_id": "file-123"},
            }],
        }),
        // A code_interpreter_call output carrying a future image part type.
        json!({
            "model": MODEL,
            "input": [{
                "type": "code_interpreter_call",
                "call_id": "c1",
                "outputs": [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
                }],
            }],
        }),
        // A case variant of a known type: serde's tag match is case-sensitive, the detector is not.
        json!({
            "model": MODEL,
            "input": [{"type": "Image", "image_url": "data:image/png;base64,aGVsbG8="}],
        }),
        // A carrier under an item type that does not exist yet, two containers deep.
        json!({
            "model": MODEL,
            "input": [{
                "type": "future_tool_call",
                "payload": {"nested": [{"frame": {"type": "Computer_Screenshot"}}]},
            }],
        }),
    ];
    for body in bodies {
        let (status, _, bytes) = send(app.clone(), json_request("/v1/responses", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
        let error = body_json(&bytes);
        assert_eq!(
            error["error"]["code"],
            StatusCode::BAD_REQUEST.as_u16(),
            "body: {body}"
        );
        assert_eq!(error["error"]["param"], Value::Null, "body: {body}");
        let message = error["error"]["message"].as_str().expect("message");
        assert!(
            message.contains("vision") && message.contains("263"),
            "names the channel and the tensors: {message}"
        );
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no screenshot shape reached generation"
    );
}

#[tokio::test]
async fn every_route_refuses_an_untyped_image_carrier_with_the_same_typed_error() {
    // The detector is route-independent: it scans the raw body the handler already parsed, so a
    // carrier the typed parsers missed is refused on all four routes with the capability error.
    let engine = RecordingEngine::default();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let bodies = [
        (
            "/v1/chat/completions",
            json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "Image", "image_url": "data:image/png;base64,aGVsbG8="},
                ]}],
            }),
        ),
        (
            "/v1/responses",
            json!({
                "model": MODEL,
                "input": [{"type": "computer_call_output", "call_id": "c1",
                           "output": {"type": "computer_screenshot", "file_id": "file-123"}}],
            }),
        ),
        (
            "/v1/messages",
            json!({
                "model": MODEL,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": [
                    {"type": "computer_screenshot", "image_url": "data:image/png;base64,aGVsbG8="},
                ]}],
            }),
        ),
        (
            // Any carrier key anywhere: a stray file_id on the completions route is refused too.
            "/v1/completions",
            json!({"model": MODEL, "prompt": "hi", "file_id": "file-123"}),
        ),
    ];
    for (uri, body) in bodies {
        let (status, _, bytes) = send(app.clone(), json_request(uri, &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {body}");
        assert_eq!(
            body_json(&bytes)["error"]["code"],
            StatusCode::BAD_REQUEST.as_u16(),
            "{uri}: {body}"
        );
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no untyped carrier reached generation"
    );
}

#[tokio::test]
async fn a_text_only_body_that_mentions_images_is_still_served() {
    // The detector looks for carriers, not words: prose about an image, and even a data: URL in a
    // text field, is text (D27's fail-open rule for genuinely non-image content).
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": [{
            "type": "text",
            "text": "an image of a cat; image_url and file_id are words here, and data:image/png;base64,aGVsbG8= is text",
        }]}],
    });
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body_json(&bytes)["choices"][0]["message"]["content"],
        "Hello world"
    );
}

#[tokio::test]
async fn a_non_image_unknown_chat_part_is_still_accepted_and_ignored() {
    // The ContentPart::Other catch-all stays fail-open for genuinely non-image parts (D27): an
    // audio part must not be mistaken for an image, so this request is served rather than refused.
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ]}],
    });
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK, "a non-image part is not refused");
    assert_eq!(
        body_json(&bytes)["choices"][0]["message"]["content"],
        "Hello world"
    );
}

#[tokio::test]
async fn every_route_refuses_an_image_request_with_the_same_typed_error() {
    // The image can arrive as a chat content part, an Anthropic image block on the chat route, a
    // completion prompt tag, a Responses input_image or an Anthropic image block. Each route must
    // refuse it at the same seam, with the same typed error, before generation - otherwise one of
    // them would silently fall back to text.
    let bodies = [
        (
            "/v1/chat/completions",
            json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
                ]}],
            }),
        ),
        (
            "/v1/chat/completions",
            json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                ]}],
            }),
        ),
        (
            "/v1/completions",
            json!({"model": MODEL, "prompt": "look <image>AAAA</image>"}),
        ),
        (
            "/v1/responses",
            json!({
                "model": MODEL,
                "input": [
                    {"type": "input_text", "text": "hi"},
                    {"type": "input_image", "image_url": "https://example.invalid/a.png"},
                ],
            }),
        ),
        (
            // Review round 2: an image nested under a function_call_output's output array used to
            // be skipped by the default arm and answered from text (HTTP 200, generate entered).
            "/v1/responses",
            json!({
                "model": MODEL,
                "input": [{
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [{"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}],
                }],
            }),
        ),
        (
            "/v1/messages",
            json!({
                "model": MODEL,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                ]}],
            }),
        ),
    ];
    for (uri, body) in bodies {
        let (status, _, bytes) = send(app(), json_request(uri, &body)).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{uri} must refuse an image"
        );
        let error = body_json(&bytes);
        assert_eq!(
            error["error"]["code"],
            StatusCode::BAD_REQUEST.as_u16(),
            "{uri}"
        );
        assert_eq!(error["error"]["param"], Value::Null, "{uri}");
    }
}

#[tokio::test]
async fn malformed_image_data_url_is_a_400() {
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!!"}},
        ]}],
    });
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes);
    assert_eq!(error["error"]["code"], 400);
    assert_eq!(error["error"]["param"], "image_url");
}

#[tokio::test]
async fn malformed_json_never_panics_and_is_openai_shaped() {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("{not json"))
        .expect("request");
    let (status, _, bytes) = send(app(), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes);
    assert_eq!(error["error"]["code"], 400);
    assert!(error["error"]["message"].is_string());
}

#[tokio::test]
async fn unknown_model_returns_404() {
    let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let error = body_json(&bytes);
    assert_eq!(error["error"]["code"], 404);
    assert_eq!(error["error"]["param"], "model");
}

#[tokio::test]
async fn trace_header_arms_scopes_and_echoes_them() {
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    let mut request = json_request("/v1/chat/completions", &chat_body("hi"));
    request.headers_mut().insert(
        "x-sparkflash-trace",
        "KV_RADIX;steps=45".parse().expect("header"),
    );
    let (status, headers, _) = send(app(), request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers.get("x-sparkflash-trace-steps").expect("steps"),
        "45"
    );
    let scopes = headers
        .get("x-sparkflash-trace-scopes")
        .expect("scopes")
        .to_str()
        .expect("ascii");
    assert!(scopes.contains("KV_RADIX"), "scopes were {scopes}");

    // Reversed order: "steps=7;<scopes>".
    let mut request = json_request("/v1/chat/completions", &chat_body("hi"));
    request.headers_mut().insert(
        "x-sparkflash-trace",
        "steps=7;engram,roce".parse().expect("header"),
    );
    let (status, headers, _) = send(app(), request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers.get("x-sparkflash-trace-steps").expect("steps"), "7");
    let scopes = headers
        .get("x-sparkflash-trace-scopes")
        .expect("scopes")
        .to_str()
        .expect("ascii");
    assert!(scopes.contains("ENGRAM"));
    assert!(scopes.contains("ROCENANTE_RDMA"));

    disarm_traces();
}

#[tokio::test]
async fn trace_header_rejects_unknown_scopes() {
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    let mut request = json_request("/v1/chat/completions", &chat_body("hi"));
    request
        .headers_mut()
        .insert("x-sparkflash-trace", "NOT_A_SCOPE".parse().expect("header"));
    let (status, _, bytes) = send(app(), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["code"], 400);

    let mut request = json_request("/v1/chat/completions", &chat_body("hi"));
    request.headers_mut().insert(
        "x-sparkflash-trace",
        "engram;steps=abc".parse().expect("header"),
    );
    let (status, _, _) = send(app(), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn admin_trace_arms_the_requested_scopes() {
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    let body = json!({"scopes": ["engram", "roce"], "steps": 45});
    let (status, _, bytes) = send(app(), json_request("/v1/admin/trace", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let response = body_json(&bytes);
    assert_ne!(response["active_mask"], 0);
    assert_eq!(response["steps_remaining"], 45);
    let scopes: Vec<String> = response["active_scopes"]
        .as_array()
        .expect("scopes")
        .iter()
        .map(|value| value.as_str().unwrap_or_default().to_owned())
        .collect();
    assert!(
        scopes.iter().any(|scope| scope.starts_with("ENGRAM")),
        "scopes {scopes:?}"
    );
    assert!(
        scopes.iter().any(|scope| scope == "ROCENANTE_RDMA"),
        "scopes {scopes:?}"
    );

    let body = json!({"scope_list": "bogus", "steps": 4});
    let (status, _, _) = send(app(), json_request("/v1/admin/trace", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    disarm_traces();
}

#[tokio::test]
async fn cluster_stats_round_trip() {
    let engine = MockEngine::new(MODEL).with_cluster_stats(ClusterStatsSnapshot::sample());
    let (status, _, bytes) = send(app_with(engine), get_request("/v1/cluster/stats")).await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    for key in [
        "model",
        "fabric_mode",
        "nodes",
        "kv",
        "engram",
        "transport",
        "dspark",
        "timing",
    ] {
        assert!(body.get(key).is_some(), "missing key {key}");
    }
    assert_eq!(body["model"], MODEL);
    assert_eq!(body["nodes"].as_array().expect("nodes").len(), 4);
    assert_eq!(body["nodes"][0]["rank"], 0);
    assert_eq!(body["engram"].as_array().expect("engram").len(), 2);
}

#[tokio::test]
async fn trace_dump_returns_gzip_chrome_trace() {
    // collect() reads the process-wide GPU ring and is unsafe to race a writer, so this test
    // serialises with the ring-writing tests through the same lock.
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    let (status, headers, bytes) = send(app(), get_request("/v1/admin/trace/dump")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers.get("content-type").expect("content type"),
        "application/gzip"
    );
    let disposition = headers
        .get("content-disposition")
        .expect("disposition")
        .to_str()
        .expect("ascii");
    assert!(
        disposition.contains("trace.json.gz"),
        "disposition was {disposition}"
    );
    let mut decoder = flate2::read::GzDecoder::new(bytes.as_slice());
    let mut text = String::new();
    decoder
        .read_to_string(&mut text)
        .expect("valid gzip payload");
    let document: Value = serde_json::from_str(&text).expect("valid JSON");
    assert!(document.get("traceEvents").is_some());
}

#[tokio::test]
async fn metrics_round_trip_after_a_request() {
    let app = app_with(MockEngine::new(MODEL).with_cluster_stats(ClusterStatsSnapshot::sample()));
    let (status, _, _) = send(
        app.clone(),
        json_request("/v1/chat/completions", &chat_body("hi")),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, headers, bytes) = send(app, get_request("/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers
            .get("content-type")
            .expect("content type")
            .to_str()
            .expect("ascii")
            .starts_with("text/plain")
    );
    let text = String::from_utf8(bytes).expect("utf8");
    for name in [
        "sparkflash_requests_total",
        "sparkflash_prompt_tokens_total",
        "sparkflash_global_kv_hit_tokens_total",
        "sparkflash_swa_tail_cache_hits_total",
        "sparkflash_engram_lookups_total",
        "sparkflash_engram_hit_rate",
        "sparkflash_trace_active_scope_mask",
        "sparkflash_uma_free_gib",
    ] {
        assert!(text.contains(name), "missing metric {name}\n{text}");
    }
    assert!(text.contains("sparkflash_requests_total 1"), "{text}");
}

#[tokio::test]
async fn models_round_trip() {
    let (status, _, bytes) = send(app(), get_request("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], MODEL);
    assert_eq!(
        body["data"][0]["context_length"],
        sparkflash_api::engine::max_seq_len()
    );
    assert_eq!(
        body["data"][0]["capabilities"]["reasoning_effort"]["max"],
        100
    );
}

#[tokio::test]
async fn models_reports_the_engines_own_capabilities() {
    // The most conservative answer: an engine that has declared nothing gets nothing advertised.
    let blind = MockEngine::new(MODEL);
    assert_eq!(blind.capabilities(), Capabilities::default());
    let (status, _, bytes) = send(app_with(blind), get_request("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    let caps = body["data"][0]["capabilities"]
        .as_object()
        .expect("capabilities is an object")
        .clone();
    assert_eq!(
        caps["multimodal"], false,
        "no engine in this workspace has a vision path, so multimodal is never advertised"
    );
    assert_eq!(caps["image_url"], false);
    assert_eq!(caps["tools"], false, "no tool calls, no tools");
    assert_eq!(caps["tool_choice"], false);
    assert_eq!(caps["thinking"], false, "no reasoning channel");
    assert_eq!(caps["chat_template_kwargs"], json!([]));
    assert!(
        !caps.contains_key("reasoning_effort"),
        "a driver that has not declared the reasoning-effort knob must not publish its schema"
    );
    assert_eq!(
        caps["streaming"], true,
        "streaming is the server's own capability, not the engine's"
    );

    // And an engine that declares every implemented channel is advertised with every implemented
    // channel. `Capabilities::ALL` has no vision member, so even the widest declaration cannot
    // advertise multimodal/image_url: the image path is not a channel to opt into while the 263
    // vision/aligner tensors are unwired (D18, D27).
    let seeing = MockEngine::new(MODEL).with_capabilities(Capabilities::ALL);
    assert_eq!(seeing.capabilities(), Capabilities::ALL);
    let (status, _, bytes) = send(app_with(seeing), get_request("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    let caps = body["data"][0]["capabilities"]
        .as_object()
        .expect("capabilities is an object")
        .clone();
    assert_eq!(caps["multimodal"], false);
    assert_eq!(caps["image_url"], false);
    assert_eq!(caps["tools"], true);
    assert_eq!(caps["tool_choice"], true);
    assert_eq!(caps["thinking"], true);
    assert_eq!(caps["chat_template_kwargs"], json!(["thinking"]));
    assert_eq!(caps["reasoning_effort"]["max"], 100);
}

#[tokio::test]
async fn completions_responses_and_messages_round_trip() {
    let (status, _, bytes) = send(
        app(),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], "Hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(body.get("x_deepseek_spark_stats").is_some());

    let (status, _, bytes) = send(
        app(),
        json_request("/v1/responses", &json!({"model": MODEL, "input": "hello"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["text"], "Hello world");
    assert_eq!(body["output_text"], "Hello world");
    assert_eq!(body["usage"]["output_tokens"], 2);

    let (status, _, bytes) = send(
        app(),
        json_request(
            "/v1/messages",
            &json!({"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hello"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "Hello world");
    assert_eq!(body["stop_reason"], "end_turn");
    assert!(
        body["usage"]["input_tokens"]
            .as_u64()
            .expect("input tokens")
            > 0
    );
}

#[tokio::test]
async fn anthropic_input_tokens_exclude_cache_reads_and_partition_the_prompt() {
    let long = "x".repeat(300);
    let body = json!({"model": MODEL, "max_tokens": 8,
        "messages": [{"role": "user", "content": long}]});
    let (status, _, uncached_bytes) = send(
        app_with(engine_with(&["Hello"])),
        json_request("/v1/messages", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let prompt = body_json(&uncached_bytes)["usage"]["input_tokens"]
        .as_u64()
        .expect("uncached input_tokens is the prompt total");
    let (status, _, bytes) = send(
        app_with(engine_with(&["Hello"]).with_cache(64, 0, false)),
        json_request("/v1/messages", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    let input = body["usage"]["input_tokens"]
        .as_u64()
        .expect("input_tokens");
    let cache_read = body["usage"]["cache_read_input_tokens"]
        .as_u64()
        .expect("cache_read_input_tokens");
    assert_eq!(
        cache_read, 64,
        "the mock's cache hit is reported separately"
    );
    assert_eq!(
        input + cache_read,
        prompt,
        "Anthropic fields must partition the prompt"
    );
    assert_eq!(input, prompt - 64);
}

#[tokio::test]
async fn openai_usage_carries_the_deepseek_cache_hit_and_miss_fields() {
    let app = app_with(engine_with(&["Hello"]).with_cache(64, 0, false));
    let (status, _, bytes) = send(
        app,
        json_request("/v1/chat/completions", &chat_body(&"x".repeat(300))),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    let usage = &body["usage"];
    let prompt = usage["prompt_tokens"].as_u64().expect("prompt tokens");
    assert_eq!(usage["prompt_cache_hit_tokens"], json!(64));
    assert_eq!(usage["prompt_cache_miss_tokens"], json!(prompt - 64));
}

#[tokio::test]
async fn max_tokens_is_honoured_and_reported_as_length() {
    let mut body = chat_body("hi");
    body["max_tokens"] = json!(1);
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello");
    assert_eq!(body["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn streaming_chat_request_is_counted_in_metrics() {
    let app = app_with(MockEngine::new(MODEL).with_cluster_stats(ClusterStatsSnapshot::sample()));
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    let (status, _, bytes) = send(app.clone(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(String::from_utf8_lossy(&bytes).ends_with("data: [DONE]\n\n"));
    let (status, _, metrics) = send(app, get_request("/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(metrics).expect("utf8");
    assert!(text.contains("sparkflash_requests_total 1"), "{text}");
    assert!(text.contains("sparkflash_prompt_tokens_total 8"), "{text}");
}

#[tokio::test]
async fn streaming_completions_request_is_counted_in_metrics() {
    let app = app_with(MockEngine::new(MODEL).with_cluster_stats(ClusterStatsSnapshot::sample()));
    let body = json!({"model": MODEL, "prompt": "hello", "stream": true});
    let (status, _, _) = send(app.clone(), json_request("/v1/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let (_, _, metrics) = send(app, get_request("/metrics")).await;
    let text = String::from_utf8(metrics).expect("utf8");
    assert!(text.contains("sparkflash_requests_total 1"), "{text}");
}

#[tokio::test]
async fn token_budget_is_validated_against_the_advertised_ceiling() {
    for bad in [json!(0), json!(393_217), json!(1_000_000)] {
        let mut body = chat_body("hi");
        body["max_tokens"] = bad.clone();
        let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "max_tokens {bad}");
        let error = body_json(&bytes);
        assert_eq!(error["error"]["code"], 400);
        assert_eq!(error["error"]["param"], "max_tokens");
    }
    let mut body = chat_body("hi");
    body["max_tokens"] = json!(393_216);
    let (status, _, _) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK, "the ceiling itself is accepted");

    let mut body = chat_body("hi");
    body["max_completion_tokens"] = json!(0);
    let (_, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(body_json(&bytes)["error"]["param"], "max_completion_tokens");
}

#[tokio::test]
async fn anthropic_streaming_is_rejected_loudly() {
    let body = json!({"model": MODEL, "max_tokens": 16, "stream": true,
        "messages": [{"role": "user", "content": "hello"}]});
    let (status, _, bytes) = send(app(), json_request("/v1/messages", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes);
    assert_eq!(error["error"]["code"], 400);
    assert_eq!(error["error"]["param"], "stream");
    assert!(
        error["error"]["message"]
            .as_str()
            .expect("message")
            .contains("stream")
    );
}

#[tokio::test]
async fn admin_trace_arms_the_gpu_ring_and_echoes_the_request_id() {
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    let body = json!({"scopes": ["engram"], "steps": 45, "request_id": "req-42"});
    let (status, _, bytes) = send(app(), json_request("/v1/admin/trace", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let response = body_json(&bytes);
    assert_eq!(response["request_id"], "req-42");
    assert_ne!(response["active_mask"], 0);
    assert_eq!(response["gpu_ring_capacity"], 4096);
    assert_eq!(response["gpu_events_written"], 0);
    assert_eq!(response["steps_remaining"], 45);
    disarm_traces();
}

#[tokio::test]
async fn admin_trace_installs_and_clears_the_request_filter() {
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    let body = json!({"scopes": ["engram"], "steps": 45, "request_id": "req-42"});
    let (status, _, bytes) = send(app(), json_request("/v1/admin/trace", &body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&bytes)["request_id"], "req-42");
    assert_eq!(
        sparkflash_trace::request_filter().as_deref(),
        Some("req-42")
    );
    assert!(
        !sparkflash_trace::is_request_traced(),
        "an unbound thread is not traced"
    );

    let previous = sparkflash_trace::bind_current_request(Some("req-42".to_owned()));
    assert!(
        sparkflash_trace::is_request_traced(),
        "the targeted request is traced"
    );
    sparkflash_trace::bind_current_request(previous);

    // A capture without a request_id clears the filter instead of swallowing later traces.
    let body = json!({"scopes": ["engram"], "steps": 45});
    let (status, _, bytes) = send(app(), json_request("/v1/admin/trace", &body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&bytes)["request_id"], Value::Null);
    assert!(sparkflash_trace::request_filter().is_none());
    assert!(sparkflash_trace::is_request_traced());
    disarm_traces();
}

#[tokio::test]
async fn disarm_traces_clears_the_request_filter() {
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    sparkflash_trace::set_request_filter(Some("req-x".to_owned()));
    assert_eq!(sparkflash_trace::request_filter().as_deref(), Some("req-x"));
    disarm_traces();
    assert!(
        sparkflash_trace::request_filter().is_none(),
        "disarm clears the filter"
    );
    assert!(sparkflash_trace::is_request_traced());
}

#[tokio::test]
async fn requests_bind_and_restore_their_trace_id() {
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    let previous = sparkflash_trace::bind_current_request(Some("outer".to_owned()));

    let mut request = json_request("/v1/chat/completions", &chat_body("hi"));
    request
        .headers_mut()
        .insert("x-request-id", "req-7".parse().expect("header"));
    let (status, _, _) = send(app(), request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        sparkflash_trace::current_request().as_deref(),
        Some("outer"),
        "the buffered handler must restore the previous binding"
    );

    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    let mut request = json_request("/v1/chat/completions", &body);
    request
        .headers_mut()
        .insert("x-request-id", "req-8".parse().expect("header"));
    let (status, _, _) = send(app(), request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        sparkflash_trace::current_request().as_deref(),
        Some("outer"),
        "the streaming handler must restore the previous binding"
    );

    sparkflash_trace::bind_current_request(previous);
    assert_eq!(sparkflash_trace::current_request(), None);
}

#[tokio::test]
async fn responses_streaming_is_rejected_loudly() {
    let body = json!({"model": MODEL, "input": "hello", "stream": true});
    let (status, _, bytes) = send(app(), json_request("/v1/responses", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes);
    assert_eq!(error["error"]["code"], 400);
    assert_eq!(error["error"]["param"], "stream");
    assert!(
        error["error"]["message"]
            .as_str()
            .expect("message")
            .contains("stream")
    );
}

#[tokio::test]
async fn trace_dump_includes_the_gpu_ring_and_reports_it_unmapped() {
    let _guard: MutexGuard<'_, ()> = trace_lock().lock().await;
    let ring = sparkflash_api::server::gpu_trace_ring();
    // scope_and_rank packs the 1-based scope bit index in the high byte and rank in the low.
    let event = sparkflash_trace::GpuTraceEvent {
        start_ns: 10,
        end_ns: 1_010,
        sub_wait_ns: 25,
        scope_and_rank: 1 << 8,
        layer_id: 3,
        payload_meta: 7,
    };
    // SAFETY: M1 has no GPU producer, so this test is the only writer of the ring; it
    // pushes and clears on one thread while holding the trace lock, so no capture can be
    // in flight concurrently.
    unsafe { ring.push(event) };

    let (status, _, bytes) = send(app(), get_request("/v1/admin/trace/dump")).await;
    assert_eq!(status, StatusCode::OK);
    let mut decoder = flate2::read::GzDecoder::new(bytes.as_slice());
    let mut text = String::new();
    decoder
        .read_to_string(&mut text)
        .expect("valid gzip payload");
    let document: Value = serde_json::from_str(&text).expect("valid JSON");

    let events = document["traceEvents"].as_array().expect("traceEvents");
    assert!(
        events.iter().any(|event| event["name"]
            .as_str()
            .is_some_and(|name| name.starts_with("gpu:"))),
        "a written GPU ring event must reach the exporter: {events:?}"
    );
    assert_eq!(document["metadata"]["gpu_ring"]["capacity"], 4096);
    assert_eq!(document["metadata"]["gpu_ring"]["events_included"], 1);
    assert_eq!(document["metadata"]["gpu_ring"]["mapped"], false);

    // SAFETY: same single-writer contract as push.
    unsafe { ring.clear() };
    assert!(
        unsafe { ring.collect() }.is_empty(),
        "ring must be left empty"
    );
}

#[tokio::test]
async fn five_independent_bypass_attacks_are_all_refused() {
    // My own attempt to defeat the detector before review: five shapes no existing test carried.
    // Each is an image carrier by protocol convention; each must be refused with the capability
    // error and never enter generation. This is the adversarial half of the pin, not a claim that
    // five shapes exhaust the space.
    let engine = RecordingEngine::default();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));

    // 1. A carrier four containers deep, under an item type that does not exist.
    let nested = json!({
        "model": MODEL,
        "input": [{
            "type": "future_call",
            "payload": {"steps": [{"result": {"frame": {"type": "Vision_Frame"}}}]},
        }],
    });
    // 2. An upper-case carrier key.
    let upper_key = json!({
        "model": MODEL,
        "input": [{"type": "future_call", "parameters": {"IMAGE_URL": "https://example.invalid/x.png"}}],
    });
    // 3. A carrier key nested inside an output object that is itself under a function_call_output.
    let nested_output = json!({
        "model": MODEL,
        "input": [{
            "type": "function_call_output",
            "call_id": "c1",
            "output": {"content": {"image": {"url": "https://example.invalid/x.png"}}},
        }],
    });
    // 4. A file reference on an item that also claims to be a screenshot by type.
    let file_ref = json!({
        "model": MODEL,
        "input": [{"type": "computer_call_output", "call_id": "c1",
                   "output": {"type": "computer_screenshot"}, "file_id": "file-9"}],
    });
    // 5. The carrier type spelled with a JSON unicode escape: a byte-level scan of the raw request
    //    would miss the word "image"; only the decoded Value carries it.
    let raw = format!(
        r##"{{"model":"{MODEL}","input":[{{"type":"future_call","parameters":{{"type":"im\u0061ge"}}}}]}}"##,
    );
    let escaped = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(raw))
        .expect("request");

    for body in [nested, upper_key, nested_output, file_ref] {
        let (status, _, bytes) = send(app.clone(), json_request("/v1/responses", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
        assert_eq!(
            body_json(&bytes)["error"]["code"],
            StatusCode::BAD_REQUEST.as_u16(),
            "body: {body}"
        );
    }
    let (status, _, bytes) = send(app.clone(), escaped).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(&bytes)["error"]["code"],
        StatusCode::BAD_REQUEST.as_u16()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "none of the five attacks reached generation"
    );
}

#[tokio::test]
async fn schema_name_maps_do_not_trip_the_refusal_but_data_carriers_still_do() {
    // Round-3 false-positive pin, narrowed. The immediate child keys of a JSON Schema name map are
    // names: a tool parameter or a response_format schema may legitimately be called "image", and
    // those requests are text requests served with generation entered. The exemption is the names
    // only; a real carrier under a non-exempt container - including a nonstandard arguments object
    // that carries image_url - is still refused fail-closed with generate not entered.
    let engine = RecordingMockEngine::new();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));

    let tool_schema = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "draw a cat"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "draw",
                "parameters": {
                    "type": "object",
                    "properties": {"image": {"type": "string"}, "image_url": {"type": "string"}},
                    "required": ["image"],
                },
            },
        }],
    });
    let response_format = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"image": {"type": "string"}}}},
        },
    });
    // Standard validation keywords that were missing from the first keyword list, plus the
    // draft-07 and OpenAPI spellings common tool schemas carry. Each is a valid schema value under
    // an image-named property, so each is a text request that must be served.
    let min_properties = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "draw",
            "parameters": {"type": "object", "properties": {
                "image": {"type": "object", "minProperties": 1},
            }},
        }}],
    });
    let draft07_dependencies = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "draw",
            "parameters": {"type": "object", "properties": {
                "image": {"type": "object", "dependencies": {"shot": ["caption"]}},
            }},
        }}],
    });
    let defs_min_properties = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "defs", "schema":
            {"type": "object", "$defs": {"image_url": {"type": "object", "minProperties": 1}}}}},
    });
    for body in [tool_schema, min_properties, draft07_dependencies] {
        let (status, _, bytes) =
            send(app.clone(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "a schema name must not refuse a text request: {}",
            String::from_utf8_lossy(&bytes)
        );
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "generation must be entered for every schema-named text request"
    );
    // A response_format json_schema is refused typed before generation now (no constrained
    // decoding backend), and the refusal is the response_format parameter, not an image carrier:
    // the schema-name exemption still applies to the carrier scan.
    for body in [response_format, defs_min_properties] {
        let (status, _, bytes) =
            send(app.clone(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
        let error = body_json(&bytes);
        assert_eq!(error["error"]["param"], "response_format", "body: {body}");
        assert_eq!(
            error["error"]["code"],
            StatusCode::BAD_REQUEST.as_u16(),
            "body: {body}"
        );
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "the response_format refusals must not enter generation"
    );

    // Fail-closed, not silent: a nonstandard arguments object, an arbitrary non-exempt container,
    // and a NON-schema value under a schema name (the name exemption requires the mapped value to be
    // a JSON Schema) all carry a real carrier. The typed parser cannot see any of them, so the
    // detector must, and generation must not be entered.
    let tool_arguments = json!({
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "draw"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "draw", "arguments": {"image": "a cat", "image_url": "https://example.invalid/x"}},
            }]},
        ],
    });
    let nested_under_payload = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "payload": {"nested": {"IMAGE_URL": "https://example.invalid/x.png"}},
    });
    let response_format_scalar_carrier = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {
            "properties": {"image": "data:image/png;base64,aGVsbG8="}}}},
    });
    let nested_schema_name_carrier = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"parameters": {
            "properties": {"outer": {"properties": {"image_url": {"url": "data:image/png;base64,aGVsbG8="}}}}}}}],
    });
    // Deliberate fail-closed policy: an unknown keyword cannot be proven to be a schema, so a
    // carrier name under it is payload and is refused rather than served.
    let unknown_schema_keyword = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"parameters": {
            "properties": {"image": {"type": "string", "x-not-a-schema-keyword": 1}}}}}],
    });
    for body in [
        tool_arguments,
        nested_under_payload,
        response_format_scalar_carrier,
        nested_schema_name_carrier,
        unknown_schema_keyword,
    ] {
        let engine2 = RecordingEngine::default();
        let calls2 = Arc::clone(&engine2.calls);
        let app2 = router(AppState::new(Arc::new(engine2)).with_fixed_time(FIXED_TIME));
        let (status, _, bytes) = send(app2, json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
        assert_eq!(
            body_json(&bytes)["error"]["code"],
            StatusCode::BAD_REQUEST.as_u16(),
            "body: {body}"
        );
        assert_eq!(
            calls2.load(Ordering::SeqCst),
            0,
            "the engine was not entered for a carrier: {body}"
        );
    }
}
// ---------------------------------------------------------------------------------------
// Logprobs
// ---------------------------------------------------------------------------------------
//
// The expected shapes are the ones vLLM emits (verified against the image's v1 logprobs code):
// the completion `top_logprobs` dict carries the sampled token in addition to the alternatives,
// the chat list is the first `top_logprobs` entries of that dict, `prompt_logprobs` is id-keyed
// with rank and decoded_token, and an `-inf` logprob is clamped to -9999.

fn alternative(token_id: u32, token: &str, logprob: f32, rank: u32) -> LogprobAlternative {
    LogprobAlternative {
        token_id,
        token: token.to_owned(),
        logprob,
        rank: Some(rank),
    }
}

fn scripted(
    token_id: u32,
    token: &str,
    logprob: f32,
    top: Vec<LogprobAlternative>,
) -> TokenLogprob {
    TokenLogprob {
        token_id,
        token: token.to_owned(),
        logprob,
        rank: Some(1),
        top,
    }
}

/// A mock with two generated tokens' logprobs and a three-token prompt's.
fn logprob_engine() -> MockEngine {
    let hello = scripted(
        11,
        "Hello",
        -0.5,
        vec![
            alternative(11, "Hello", -0.5, 1),
            alternative(7, "Hi", -1.5, 2),
        ],
    );
    let world = scripted(
        12,
        " world",
        -0.25,
        vec![
            alternative(12, " world", -0.25, 1),
            alternative(9, " there", -2.0, 2),
            alternative(10, " rare", f32::NEG_INFINITY, 3),
        ],
    );
    let prompt = PromptLogprobs {
        tokens: vec![
            TokenString {
                token_id: 21,
                token: "a".to_owned(),
            },
            TokenString {
                token_id: 22,
                token: "b".to_owned(),
            },
            TokenString {
                token_id: 23,
                token: "c".to_owned(),
            },
        ],
        positions: vec![
            None,
            Some(scripted(
                22,
                "b",
                -0.125,
                vec![alternative(22, "b", -0.125, 1)],
            )),
            Some(scripted(
                23,
                "c",
                -0.25,
                vec![
                    alternative(23, "c", -0.25, 2),
                    alternative(24, "d", -0.0625, 1),
                ],
            )),
        ],
    };
    MockEngine::new(MODEL)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .with_capabilities(Capabilities::ALL)
        .content(&["Hello", " world"])
        .with_vocab_size(128)
        .with_token_logprobs(vec![hello, world])
        .with_prompt_logprobs(prompt)
}

#[tokio::test]
async fn completion_logprobs_have_the_vllm_shape() {
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "logprobs": 1}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    let logprobs = &body["choices"][0]["logprobs"];
    assert_eq!(logprobs["tokens"], json!(["Hello", " world"]));
    assert_eq!(logprobs["text_offset"], json!([0, 5]));
    assert_eq!(logprobs["token_logprobs"], json!([-0.5, -0.25]));
    assert_eq!(
        logprobs["top_logprobs"][0],
        json!({"Hello": -0.5, "Hi": -1.5}),
        "the completion dict carries the sampled token as an entry"
    );
    assert_eq!(
        logprobs["top_logprobs"][1],
        json!({" world": -0.25, " there": -2.0, " rare": -9999.0}),
        "a -inf logprob is clamped to -9999"
    );
    assert_eq!(
        body["choices"][0]["prompt_logprobs"],
        Value::Null,
        "no prompt logprobs were requested"
    );
}

#[tokio::test]
async fn completion_prompt_logprobs_are_id_keyed_with_rank_and_decoded_token() {
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "prompt_logprobs": 1}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    let prompt = &body["choices"][0]["prompt_logprobs"];
    assert_eq!(prompt.as_array().expect("array").len(), 3);
    assert_eq!(
        prompt[0],
        Value::Null,
        "position 0 has no conditioning token"
    );
    assert_eq!(
        prompt[1],
        json!({"22": {"logprob": -0.125, "rank": 1, "decoded_token": "b"}})
    );
    assert_eq!(
        prompt[2],
        json!({
            "23": {"logprob": -0.25, "rank": 2, "decoded_token": "c"},
            "24": {"logprob": -0.0625, "rank": 1, "decoded_token": "d"},
        })
    );
    assert_eq!(
        body["choices"][0]["logprobs"],
        Value::Null,
        "prompt logprobs do not imply generated-token logprobs"
    );
}

#[tokio::test]
async fn echo_prepends_the_prompt_and_its_logprobs() {
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "echo": true, "logprobs": 1}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    let choice = &body["choices"][0];
    assert_eq!(choice["text"], "helloHello world");
    let logprobs = &choice["logprobs"];
    assert_eq!(
        logprobs["tokens"],
        json!(["a", "b", "c", "Hello", " world"]),
        "echo lists the prompt tokens before the generated ones"
    );
    assert_eq!(logprobs["text_offset"], json!([0, 1, 2, 3, 8]));
    assert_eq!(
        logprobs["token_logprobs"],
        json!([Value::Null, -0.125, -0.25, -0.5, -0.25])
    );
    assert_eq!(
        logprobs["top_logprobs"][0],
        Value::Null,
        "the echoed position 0 has no score"
    );
    assert!(
        choice["prompt_logprobs"].is_array(),
        "echo computed the prompt logprobs from logprobs (vLLM's rule): {}",
        choice["prompt_logprobs"]
    );
}

#[tokio::test]
async fn echo_without_generation_returns_the_prompt_only() {
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "echo": true, "logprobs": 1, "max_tokens": 0}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "echo makes max_tokens 0 legal");
    let body = body_json(&bytes);
    let choice = &body["choices"][0];
    assert_eq!(choice["text"], "hello");
    assert_eq!(choice["finish_reason"], "length");
    assert_eq!(choice["logprobs"]["tokens"], json!(["a", "b", "c"]));
    assert_eq!(
        choice["logprobs"]["text_offset"],
        json!([0, 1, 2]),
        "offset stays inside the prompt text"
    );
    assert_eq!(body["usage"]["completion_tokens"], 0);
}

#[tokio::test]
async fn chat_logprobs_entries_carry_bytes_and_the_top_list() {
    let mut body = chat_body("hi");
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(2);
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    let content = &body["choices"][0]["logprobs"]["content"];
    assert_eq!(content.as_array().expect("array").len(), 2);
    assert_eq!(content[0]["token"], "Hello");
    assert_eq!(content[0]["logprob"], -0.5);
    assert_eq!(content[0]["bytes"], json!([72, 101, 108, 108, 111]));
    let top = content[0]["top_logprobs"].as_array().expect("array");
    assert_eq!(top.len(), 2, "the list is the first n entries of the dict");
    assert_eq!(top[0]["token"], "Hello");
    assert_eq!(top[0]["logprob"], -0.5);
    assert_eq!(top[0]["bytes"], json!([72, 101, 108, 108, 111]));
    assert_eq!(top[1]["token"], "Hi");

    let mut second = chat_body("hi");
    second["logprobs"] = json!(true);
    second["top_logprobs"] = json!(0);
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request("/v1/chat/completions", &second),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(
        body["choices"][0]["logprobs"]["content"][0]["top_logprobs"],
        json!([]),
        "logprobs without top_logprobs reports the token alone"
    );
}

#[tokio::test]
async fn chat_prompt_logprobs_are_reported_at_the_top_level() {
    let mut body = chat_body("hi");
    body["prompt_logprobs"] = json!(1);
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["choices"][0]["logprobs"], Value::Null);
    let prompt = &body["prompt_logprobs"];
    assert_eq!(prompt.as_array().expect("array").len(), 3);
    assert_eq!(prompt[0], Value::Null);
    assert_eq!(
        prompt[1],
        json!({"22": {"logprob": -0.125, "rank": 1, "decoded_token": "b"}})
    );
}

#[tokio::test]
async fn logprobs_are_validated_the_way_vllm_validates_them() {
    let cases: Vec<(Value, &str)> = vec![
        (
            json!({"model": MODEL, "prompt": "hello", "logprobs": 21}),
            "Requested sample logprobs of 21, which is greater than max allowed: 20",
        ),
        (
            json!({"model": MODEL, "prompt": "hello", "logprobs": -1}),
            "Requested sample logprobs of 128, which is greater than max allowed: 20",
        ),
        (
            json!({"model": MODEL, "prompt": "hello", "logprobs": -2}),
            "`logprobs` must be a positive value or -1.",
        ),
        (
            json!({"model": MODEL, "prompt": "hello", "prompt_logprobs": 21}),
            "Requested prompt logprobs of 21, which is greater than max allowed: 20",
        ),
        (
            json!({"model": MODEL, "prompt": "hi", "logprob_token_ids": [1]}),
            "when using `logprob_token_ids`, `logprobs` must be set.",
        ),
        (
            json!({"model": MODEL, "prompt": "hi", "logprobs": 1, "logprob_token_ids": [129]}),
            "token_id(s) [129] in logprob_token_ids contain out-of-vocab token ids. Vocabulary size: 128",
        ),
    ];
    for (case, message) in cases {
        let (status, _, bytes) = send(
            app_with(logprob_engine()),
            json_request("/v1/completions", &case),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "body: {case}");
        let body = body_json(&bytes);
        assert_eq!(body["error"]["code"], 400, "body: {case}");
        assert_eq!(body["error"]["message"], message, "body: {case}");
    }

    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hi", "logprobs": 1, "stream": true, "prompt_logprobs": 1}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = body_json(&bytes);
    assert_eq!(body["error"]["code"], 400);
    assert_eq!(
        body["error"]["message"],
        "`prompt_logprobs` are not available when `stream=True`."
    );

    let mut chat = chat_body("hi");
    chat["top_logprobs"] = json!(2);
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request("/v1/chat/completions", &chat),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(&bytes)["error"]["message"],
        "when using `top_logprobs`, `logprobs` must be set to true."
    );
}

#[tokio::test]
async fn an_explicit_id_list_takes_precedence_over_the_natural_selection() {
    // vLLM drops `logprobs`/\`top_logprobs\` from the sampling parameters when `logprob_token_ids` is
    // set, so neither the cap nor the list length is checked: only presence and the -1 floor are.
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "logprobs": 21, "logprob_token_ids": [7]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {}", body_json(&bytes));
    let body = body_json(&bytes);
    let top = body["choices"][0]["logprobs"]["top_logprobs"][0]
        .as_object()
        .expect("a dict");
    assert!(
        top.keys().any(|key| key == "Hi"),
        "the explicit id is reported: {top:?}"
    );

    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "logprobs": -2, "logprob_token_ids": [7]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(&bytes)["error"]["message"],
        "`logprobs` must be a positive value or -1."
    );

    let mut chat = chat_body("hi");
    chat["logprobs"] = json!(true);
    chat["top_logprobs"] = json!(21);
    chat["logprob_token_ids"] = json!([7]);
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request("/v1/chat/completions", &chat),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {}", body_json(&bytes));

    let mut chat = chat_body("hi");
    chat["logprob_token_ids"] = json!([7]);
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request("/v1/chat/completions", &chat),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(&bytes)["error"]["message"],
        "when using `logprob_token_ids`, `logprobs` must be set to true."
    );
}

#[tokio::test]
async fn chat_logprobs_true_alone_asks_for_the_sampled_token() {
    let mut body = chat_body("hi");
    body["logprobs"] = json!(true);
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {}", body_json(&bytes));
    let body = body_json(&bytes);
    let content = &body["choices"][0]["logprobs"]["content"];
    assert_eq!(content.as_array().expect("array").len(), 2);
    assert_eq!(content[0]["token"], "Hello");
    assert_eq!(
        content[0]["top_logprobs"],
        json!([]),
        "vLLM's logprobs=0 reports the token with no alternatives"
    );
}

#[tokio::test]
async fn an_unstated_vocabulary_cannot_serve_the_whole_vocabulary() {
    let engine = MockEngine::new(MODEL)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .with_capabilities(Capabilities::ALL)
        .content(&["Hello"])
        .with_token_logprobs(vec![scripted(
            11,
            "Hello",
            -0.5,
            vec![alternative(11, "Hello", -0.5, 1)],
        )]);
    let (status, _, bytes) = send(
        app_with(engine),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "logprobs": -1}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body_json(&bytes)["error"]["message"]
            .as_str()
            .expect("message")
            .contains("does not report its vocabulary size"),
        "{}",
        body_json(&bytes)["error"]["message"]
    );
}

#[tokio::test]
async fn the_logprob_cap_can_be_lifted_like_max_logprobs() {
    let app = router(
        AppState::new(Arc::new(logprob_engine()))
            .with_fixed_time(FIXED_TIME)
            .with_max_logprobs(usize::MAX),
    );
    let (status, _, bytes) = send(
        app,
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "logprobs": -1}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {}", body_json(&bytes));
}

#[tokio::test]
async fn streaming_chat_carries_each_token_with_its_logprobs() {
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(2);
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let frames = sse_frames(&bytes);
    let scored: Vec<&Value> = frames
        .iter()
        .filter(|frame| frame["choices"][0]["logprobs"].is_object())
        .collect();
    assert_eq!(scored.len(), 2, "one chunk per generated token: {scored:?}");
    assert_eq!(scored[0]["choices"][0]["delta"]["content"], "Hello");
    assert_eq!(
        scored[0]["choices"][0]["logprobs"]["content"][0]["token"],
        "Hello"
    );
    assert_eq!(scored[1]["choices"][0]["delta"]["content"], " world");
}

#[tokio::test]
async fn streaming_completions_echo_the_prompt_in_the_first_chunk() {
    let (status, _, bytes) = send(
        app_with(logprob_engine()),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "stream": true, "echo": true, "logprobs": 1}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let frames = sse_frames(&bytes);
    let scored: Vec<&Value> = frames
        .iter()
        .filter(|frame| frame["choices"][0]["logprobs"].is_object())
        .collect();
    assert_eq!(scored.len(), 2, "one chunk per generated token: {scored:?}");
    assert_eq!(scored[0]["choices"][0]["text"], "helloHello");
    assert_eq!(
        scored[0]["choices"][0]["logprobs"]["tokens"],
        json!(["a", "b", "c", "Hello"])
    );
    assert_eq!(
        scored[0]["choices"][0]["logprobs"]["text_offset"],
        json!([0, 1, 2, 3])
    );
    assert_eq!(scored[1]["choices"][0]["text"], " world");
    assert_eq!(
        scored[1]["choices"][0]["logprobs"]["text_offset"],
        json!([10]),
        "the streamed offset continues from the text already sent"
    );
}

#[tokio::test]
async fn health_returns_200_with_an_empty_body() {
    // vLLM's liveness probe is a 200 with no body
    // (vllm/entrypoints/serve/instrumentator/health.py); orchestrator probes only read the status.
    let response = app()
        .oneshot(get_request("/health"))
        .await
        .expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    assert!(bytes.is_empty(), "health has no body: {bytes:?}");
}

#[tokio::test]
async fn readyz_reports_serving_then_draining_while_health_stays_up() {
    // Readiness is the drain signal's client-visible edge: 200 while accepting, 503 the moment the
    // drain starts, while liveness (/health) stays 200 because the process still finishes its
    // in-flight requests. The route is public (a probe has no bearer key) and answers from a flag,
    // never from the engine.
    let state = AppState::new(Arc::new(engine_with(&["Hello"]))).with_fixed_time(FIXED_TIME);
    let app = router(state.clone());
    let (status, _, bytes) = send(app.clone(), get_request("/readyz")).await;
    assert_eq!(status, StatusCode::OK, "serving before the drain signal");
    assert!(bytes.is_empty(), "readiness has no body: {bytes:?}");
    state.begin_drain();
    let (status, _, bytes) = send(app.clone(), get_request("/readyz")).await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "draining turns readyz 503"
    );
    assert!(bytes.is_empty(), "readiness has no body: {bytes:?}");
    let (status, _, _) = send(app.clone(), get_request("/health")).await;
    assert_eq!(status, StatusCode::OK, "liveness survives the drain signal");
    let (status, _, _) = send(app, get_request("/metrics")).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the scraper keeps working during a drain"
    );
}

#[tokio::test]
async fn an_empty_stop_string_is_refused_like_vllm() {
    // vLLM's SamplingParams._verify_args: "stop cannot contain an empty string."
    let body =
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stop": [""]});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes);
    assert_eq!(
        error["error"]["message"],
        "stop cannot contain an empty string."
    );
    assert_eq!(error["error"]["param"], "stop");
    assert_eq!(error["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn a_non_empty_stop_string_is_accepted_and_passed_to_the_engine() {
    let (status, _, _) = send(
        app(),
        json_request(
            "/v1/chat/completions",
            &json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stop": ["END"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// vLLM's `SamplingParams._verify_args` ranges: every refusal names its parameter and carries the
/// vLLM message text, so a client sees the same error shape the oracle would give it.
#[tokio::test]
async fn out_of_range_sampling_parameters_are_refused_like_vllm() {
    let cases = [
        (
            json!({"top_p": 1.5}),
            "top_p",
            "top_p must be in [0, 1], got 1.5.",
        ),
        (
            json!({"top_k": -2}),
            "top_k",
            "top_k must be 0 (disable), or at least 1, got -2.",
        ),
        (
            json!({"repetition_penalty": 0.0}),
            "repetition_penalty",
            "repetition_penalty must be greater than zero, got 0.0.",
        ),
        (
            json!({"presence_penalty": 2.5}),
            "presence_penalty",
            "presence_penalty must be in [-2, 2], got 2.5.",
        ),
        (
            json!({"frequency_penalty": -2.5}),
            "frequency_penalty",
            "frequency_penalty must be in [-2, 2], got -2.5.",
        ),
    ];
    for (extra, parameter, message) in cases {
        let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut()
            .expect("request body is an object")
            .extend(extra.as_object().expect("case is an object").clone());
        let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "case {parameter}: {body}");
        let error = body_json(&bytes);
        assert_eq!(error["error"]["message"], message, "case {parameter}");
        assert_eq!(error["error"]["param"], parameter, "case {parameter}");
        assert_eq!(
            error["error"]["type"], "invalid_request_error",
            "case {parameter}"
        );
    }
}

/// `top_p` 0.0 is our documented extension: vLLM's range is (0, 1] and refuses it, while the
/// reference sampler accepts it as "keep the single most likely token". The value must not be
/// refused here, and the census probe relies on that.
#[tokio::test]
async fn top_p_zero_is_accepted_as_the_documented_extension() {
    let body =
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "top_p": 0.0});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
}

/// The parsed sampling parameters reach the engine's normalized request unchanged.
#[tokio::test]
async fn sampling_parameters_reach_the_engine_request() {
    let engine = RecordingRequestEngine::new();
    let requests = engine.requests.clone();
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "enable_thinking": false,
        "repetition_penalty": 1.2,
        "presence_penalty": 0.5,
        "frequency_penalty": -0.25,
        "skip_special_tokens": false,
        "seed": 7
    });
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let recorded = requests.lock().expect("the recorder is not poisoned");
    assert_eq!(recorded.len(), 1, "exactly one request reached the engine");
    let request = &recorded[0];
    assert_eq!(request.temperature, 0.7);
    assert_eq!(request.top_p, 0.9);
    assert_eq!(request.top_k, 40);
    assert_eq!(request.repetition_penalty, 1.2);
    assert_eq!(request.presence_penalty, 0.5);
    assert_eq!(request.frequency_penalty, -0.25);
    assert!(
        !request.skip_special_tokens,
        "an explicit false reaches the engine"
    );
}

/// The completions endpoint maps and validates the same five sampling parameters.
#[tokio::test]
async fn completions_sampling_parameters_reach_the_engine_request() {
    let engine = RecordingRequestEngine::new();
    let requests = engine.requests.clone();
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let body = json!({
        "model": MODEL,
        "prompt": "hi",
        "max_tokens": 4,
        "top_k": 3,
        "repetition_penalty": 1.5,
        "presence_penalty": -0.5,
        "frequency_penalty": 0.25
    });
    let (status, _, bytes) = send(app, json_request("/v1/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    {
        let recorded = requests.lock().expect("the recorder is not poisoned");
        assert_eq!(recorded.len(), 1, "exactly one request reached the engine");
        let request = &recorded[0];
        assert_eq!(request.top_k, 3);
        assert_eq!(request.repetition_penalty, 1.5);
        assert_eq!(request.presence_penalty, -0.5);
        assert_eq!(request.frequency_penalty, 0.25);
        assert!(
            request.skip_special_tokens,
            "the absent field resolves to vLLM's true default"
        );
    }

    // The same endpoint refuses an out-of-range value with the vLLM message.
    let (status, _, bytes) = send(
        router(AppState::new(Arc::new(RecordingRequestEngine::new())).with_fixed_time(FIXED_TIME)),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hi", "top_k": -3}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes);
    assert_eq!(
        error["error"]["message"],
        "top_k must be 0 (disable), or at least 1, got -3."
    );
    assert_eq!(error["error"]["param"], "top_k");
}

/// vLLM's `--enable-auto-tool-choice --tool-call-parser deepseek_v41` equivalent for the
/// buffered answer: the V4.1 DSML block the model emits becomes an OpenAI `tool_calls` array and
/// the finish reason becomes `tool_calls`.
#[tokio::test]
async fn deepseek_v41_dsml_tool_calls_are_parsed_on_chat_completions() {
    let answer = [
        "I will check the weather.",
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"get_weather\">\n",
        "<｜DSML｜ parameter name=\"city\" string=\"true\">Beijing</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    ]
    .concat();
    let app = app_with(engine_with(&[&answer]));
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Read the weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }}],
    });
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let json = body_json(&bytes);
    let choice = &json["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    let message = &choice["message"];
    assert_eq!(
        message["content"], "I will check the weather.",
        "the text before the block stays content"
    );
    let calls = message["tool_calls"].as_array().expect("tool_calls array");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(
        calls[0]["function"]["arguments"],
        r#"{"city": "Beijing", "days": 3}"#
    );
    assert!(
        calls[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("chatcmpl-tool-")),
        "vLLM tool-call ids use the chatcmpl-tool- shape: {calls:?}"
    );
}

/// The parser only runs when the client offered tools: the same DSML text stays content.
#[tokio::test]
async fn deepseek_v41_dsml_without_tools_stays_content() {
    let answer = [
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"ping\">\n",
        "<｜DSML｜ parameter name=\"n\" string=\"false\">1</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    ]
    .concat();
    let app = app_with(engine_with(&[&answer]));
    let body = json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let json = body_json(&bytes);
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["message"]["content"], answer);
    assert!(
        json["choices"][0]["message"].get("tool_calls").is_none(),
        "no tools were offered, so no calls are lifted: {json}"
    );
}

// ---------------------------------------------------------------------------------------
// Streaming DSML tool calls: the V4.1 block is parsed incrementally while the content
// deltas arrive, and the streamed fragments must assemble to the buffered payload.
// ---------------------------------------------------------------------------------------

/// The same answer the buffered DSML test uses, as one string.
fn dsml_stream_answer() -> String {
    [
        "I will check the weather.",
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"get_weather\">\n",
        "<｜DSML｜ parameter name=\"city\" string=\"true\">Beijing</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    ]
    .concat()
}

/// Splits an answer into content pieces at character boundaries, the way the engine emits them.
///
/// The pieces deliberately ignore DSML token/JSON boundaries: every split is a case the
/// incremental parser has to hold state across.
fn chunked(text: &str, char_size: usize) -> Vec<String> {
    text.chars()
        .collect::<Vec<_>>()
        .chunks(char_size)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect()
}

fn stream_pieces(pieces: &[String]) -> Vec<&str> {
    pieces.iter().map(String::as_str).collect()
}

fn tools_body(stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Read the weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }}],
        "stream": stream,
    })
}

#[derive(Debug, PartialEq, Eq)]
struct AssembledCall {
    id: String,
    name: String,
    arguments: String,
}

fn assemble_streamed_calls(frames: &[Value]) -> Vec<AssembledCall> {
    let mut calls: Vec<AssembledCall> = Vec::new();
    for frame in frames {
        for call in frame["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let index = call["index"].as_u64().expect("tool_call index") as usize;
            if let Some(id) = call["id"].as_str() {
                assert_eq!(calls.len(), index, "a call starts before its fragments");
                calls.push(AssembledCall {
                    id: id.to_owned(),
                    name: call["function"]["name"].as_str().expect("name").to_owned(),
                    arguments: call["function"]["arguments"]
                        .as_str()
                        .expect("the first delta carries an arguments string")
                        .to_owned(),
                });
            } else {
                calls[index].arguments.push_str(
                    call["function"]["arguments"]
                        .as_str()
                        .expect("fragments are strings"),
                );
            }
        }
    }
    calls
}

fn streamed_content(frames: &[Value]) -> String {
    let mut content = String::new();
    for frame in frames {
        if let Some(text) = frame["choices"][0]["delta"]["content"].as_str() {
            content.push_str(text);
        }
    }
    content
}

#[tokio::test]
async fn deepseek_v41_dsml_tool_calls_stream_like_the_buffered_answer() {
    let answer = dsml_stream_answer();
    let pieces = chunked(&answer, 7);
    // Buffered answer from a fresh state: its id counter starts at zero.
    let buffered_request = tools_body(false);
    let (status, _, buffered_bytes) = send(
        app_with(engine_with(&stream_pieces(&pieces))),
        json_request("/v1/chat/completions", &buffered_request),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&buffered_bytes)
    );
    let buffered = body_json(&buffered_bytes);
    let buffered_choice = &buffered["choices"][0];
    assert_eq!(buffered_choice["finish_reason"], "tool_calls");
    let buffered_calls: Vec<AssembledCall> = buffered_choice["message"]["tool_calls"]
        .as_array()
        .expect("buffered tool_calls")
        .iter()
        .map(|call| AssembledCall {
            id: call["id"].as_str().expect("id").to_owned(),
            name: call["function"]["name"].as_str().expect("name").to_owned(),
            arguments: call["function"]["arguments"]
                .as_str()
                .expect("arguments")
                .to_owned(),
        })
        .collect();

    // Streaming answer from another fresh state, so the response id shapes line up.
    let (status, _, bytes) = send(
        app_with(engine_with(&stream_pieces(&pieces))),
        json_request("/v1/chat/completions", &tools_body(true)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let text = String::from_utf8(bytes.clone()).expect("utf8");
    assert_eq!(
        text.matches("data: ").count(),
        sse_frames(&bytes).len() + 1,
        "one data frame per event plus the terminator"
    );
    assert!(text.ends_with("data: [DONE]\n\n"));
    let frames = sse_frames(&bytes);
    assert_eq!(
        frames[0]["choices"][0]["delta"]["role"], "assistant",
        "the role frame is first"
    );
    let finish = &frames[frames.len() - 1]["choices"][0]["finish_reason"];
    assert_eq!(finish, "tool_calls", "the final chunk carries the override");
    let streamed_calls = assemble_streamed_calls(&frames);
    assert_eq!(
        streamed_calls, buffered_calls,
        "streamed fragments must reassemble the buffered tool_calls byte for byte"
    );
    assert_eq!(
        streamed_content(&frames),
        buffered_choice["message"]["content"]
            .as_str()
            .expect("buffered content"),
        "the text before the block streams as content"
    );
    assert_eq!(
        streamed_calls[0].arguments,
        r#"{"city": "Beijing", "days": 3}"#
    );
    assert!(
        !streamed_content(&frames).contains("DSML"),
        "the DSML block itself is never sent as content"
    );
}

#[tokio::test]
async fn several_streamed_dsml_calls_keep_their_indices_and_order() {
    let answer = [
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"first\">\n",
        "<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "<｜DSML｜ invoke name=\"second\">\n",
        "<｜DSML｜ parameter name=\"b\" string=\"false\">true</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    ]
    .concat();
    let pieces = chunked(&answer, 3);
    let (status, _, bytes) = send(
        app_with(engine_with(&stream_pieces(&pieces))),
        json_request("/v1/chat/completions", &tools_body(true)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let frames = sse_frames(&bytes);
    let calls = assemble_streamed_calls(&frames);
    assert_eq!(
        calls
            .iter()
            .map(|call| (call.name.as_str(), call.arguments.as_str()))
            .collect::<Vec<_>>(),
        vec![("first", r#"{"a": "x"}"#), ("second", r#"{"b": true}"#)]
    );
    assert_eq!(calls[0].id, "chatcmpl-tool-00000001");
    assert_eq!(calls[1].id, "chatcmpl-tool-00000002");
    assert_eq!(
        frames[frames.len() - 1]["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert_eq!(streamed_content(&frames), "");
}

#[tokio::test]
async fn a_malformed_streamed_dsml_block_ends_with_a_typed_error() {
    let answer = [
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"x\">\n",
        "<｜DSML｜ junk name=\"a\" string=\"true\">v</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    ]
    .concat();
    let pieces = chunked(&answer, 4);
    let (status, _, bytes) = send(
        app_with(engine_with(&stream_pieces(&pieces))),
        json_request("/v1/chat/completions", &tools_body(true)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let text = String::from_utf8(bytes.clone()).expect("utf8");
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    let raw: Vec<&str> = text
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .collect();
    assert_eq!(
        raw.first().copied(),
        Some(
            r#"data: {"id":"chatcmpl-dsv41-00000000","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-ai/DeepSeek-V4.1-Flash","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#
        ),
        "no frame precedes the role frame"
    );
    let error = raw
        .iter()
        .find(|frame| frame.contains("\"error\""))
        .expect("a typed error frame");
    assert!(
        error.contains("\"code\":\"tool_call_parse_error\""),
        "the refusal code is typed: {error}"
    );
    assert_eq!(
        raw.last().copied(),
        Some("data: [DONE]"),
        "the terminator is last"
    );
    let frames = sse_frames(&bytes);
    assert!(
        frames
            .iter()
            .all(|frame| frame["choices"][0]["finish_reason"].is_null()),
        "no finish chunk rides a refused stream: {frames:?}"
    );
    // The invoke name is already known before the malformed tag arrives, so the call start may
    // have been sent; the refusal still ends the stream without a finish chunk and later events
    // are dropped instead of being interleaved with a broken call.
    let calls = assemble_streamed_calls(&frames);
    assert!(
        calls.len() <= 1,
        "a malformed block cannot produce more than the started call: {calls:?}"
    );
    assert!(
        calls.iter().all(|call| call.name == "x"),
        "only the call whose name was already known can start: {calls:?}"
    );
}

#[tokio::test]
async fn an_unterminated_streamed_dsml_block_ends_with_a_typed_error() {
    let answer = [
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"x\">\n",
        "<｜DSML｜ parameter name=\"a\" string=\"true\">unfinished",
    ]
    .concat();
    let pieces = chunked(&answer, 5);
    let (status, _, bytes) = send(
        app_with(engine_with(&stream_pieces(&pieces))),
        json_request("/v1/chat/completions", &tools_body(true)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let text = String::from_utf8(bytes.clone()).expect("utf8");
    assert!(text.ends_with("data: [DONE]\n\n"));
    assert!(
        text.contains("\"code\":\"tool_call_parse_error\""),
        "the unterminated block is a typed error: {text}"
    );
    let frames = sse_frames(&bytes);
    assert!(
        frames
            .iter()
            .all(|frame| frame["choices"][0]["finish_reason"].is_null()),
        "the error frame is terminal, without a finish chunk"
    );
    assert!(assemble_streamed_calls(&frames).len() <= 1);
}

#[tokio::test]
async fn streamed_dsml_tool_call_fragments_survive_keep_alive_comments() {
    let answer = dsml_stream_answer();
    let pieces = chunked(&answer, 9);
    let engine = engine_with(&stream_pieces(&pieces)).with_delay(Duration::from_millis(20));
    let state = AppState::new(Arc::new(engine))
        .with_fixed_time(FIXED_TIME)
        .with_sse_keep_alive(Some(Duration::from_millis(5)));
    let (status, _, bytes) = send(
        router(state),
        json_request("/v1/chat/completions", &tools_body(true)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let text = String::from_utf8(bytes.clone()).expect("utf8");
    let raw: Vec<&str> = text
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .collect();
    let comments = raw.iter().filter(|frame| **frame == ": ping").count();
    assert!(
        comments > 0,
        "an idle engine gap should carry keep-alive comments: {text}"
    );
    assert_ne!(
        raw.first().copied(),
        Some(": ping"),
        "no comment precedes the first event"
    );
    assert_eq!(
        raw.last().copied(),
        Some("data: [DONE]"),
        "comments never follow the terminator"
    );
    assert!(
        raw.iter()
            .filter(|frame| **frame != ": ping")
            .all(|frame| frame.starts_with("data: ")),
        "a comment is a whole frame, never interleaved into a data frame: {text}"
    );
    let frames = sse_frames(&bytes);
    let calls = assemble_streamed_calls(&frames);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, r#"{"city": "Beijing", "days": 3}"#);
    assert_eq!(
        frames[frames.len() - 1]["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert_eq!(streamed_content(&frames), "I will check the weather.");
}

#[tokio::test]
async fn dsml_streams_as_raw_content_when_no_tools_are_offered() {
    let answer = dsml_stream_answer();
    let pieces = chunked(&answer, 7);
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    let (status, _, bytes) = send(
        app_with(engine_with(&stream_pieces(&pieces))),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let frames = sse_frames(&bytes);
    assert_eq!(
        streamed_content(&frames),
        answer,
        "without tools the DSML text is plain content"
    );
    assert!(
        assemble_streamed_calls(&frames).is_empty(),
        "no tools were offered, so nothing is lifted"
    );
    assert_eq!(
        frames[frames.len() - 1]["choices"][0]["finish_reason"],
        "stop"
    );
}

// ---------------------------------------------------------------------------------------
// Task-20: multiple choices, response_format, bearer auth
// ---------------------------------------------------------------------------------------

/// An engine that reports it cannot draw independent choices (the TP=4 shape).
struct SharedSamplingEngine {
    calls: Arc<AtomicUsize>,
    inner: MockEngine,
}

impl InferenceEngine for SharedSamplingEngine {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> futures::stream::BoxStream<'static, EngineEvent> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.generate(request)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.inner.cluster_stats()
    }

    fn independent_choices(&self) -> bool {
        false
    }
}

fn keyed_app(key: &str) -> Router {
    router(
        AppState::new(Arc::new(engine_with(&["Hello", " world"])))
            .with_fixed_time(FIXED_TIME)
            .with_api_key(Some(key.to_owned())),
    )
}

fn authorized_request(uri: &str, body: Option<&Value>, header: &str) -> Request<Body> {
    let builder = Request::builder().method(if body.is_some() { "POST" } else { "GET" });
    let builder = builder
        .uri(uri)
        .header("authorization", header)
        .header("content-type", "application/json");
    match body {
        Some(body) => builder
            .body(Body::from(serde_json::to_vec(body).expect("json body")))
            .expect("request"),
        None => builder.body(Body::empty()).expect("request"),
    }
}

#[tokio::test]
async fn chat_n_two_choices_buffered_counts_the_prompt_once() {
    let app = app();
    let baseline = body_json(
        &send(
            app.clone(),
            json_request("/v1/chat/completions", &chat_body("hi")),
        )
        .await
        .2,
    );
    let mut body = chat_body("hi");
    body["n"] = json!(2);
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let response = body_json(&bytes);
    let choices = response["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 2);
    for (index, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], index);
        assert_eq!(choice["message"]["content"], "Hello world");
        assert_eq!(choice["finish_reason"], "stop");
    }
    let baseline_usage = &baseline["usage"];
    let usage = &response["usage"];
    assert_eq!(usage["prompt_tokens"], baseline_usage["prompt_tokens"]);
    assert_eq!(
        usage["completion_tokens"].as_u64().expect("completion"),
        baseline_usage["completion_tokens"]
            .as_u64()
            .expect("baseline completion")
            * 2
    );
    assert_eq!(
        usage["total_tokens"].as_u64().expect("total"),
        usage["prompt_tokens"].as_u64().expect("prompt")
            + usage["completion_tokens"].as_u64().expect("completion")
    );
    assert!(
        response["x_deepseek_spark_stats"].is_object(),
        "the extension is aggregated over the choices: {}",
        String::from_utf8_lossy(&bytes)
    );
}

#[tokio::test]
async fn chat_n_choices_carry_per_choice_ids_and_seeds() {
    let engine = RecordingRequestEngine::new();
    let requests = Arc::clone(&engine.requests);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let mut body = chat_body("hi");
    body["n"] = json!(3);
    body["seed"] = json!(7);
    let (status, _, bytes) = send(app.clone(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let response = body_json(&bytes);
    let id = response["id"].as_str().expect("id");
    let recorded: Vec<EngineRequest> = requests.lock().expect("recorder").clone();
    assert_eq!(recorded.len(), 3);
    for (choice, request) in recorded.iter().enumerate() {
        assert_eq!(request.request_id, format!("{id}-{choice}"));
        assert_eq!(
            request.seed,
            Some(7 + choice as u64),
            "vLLM derives choice i's seed as seed + i"
        );
        assert_eq!(request.prompt_tokens, recorded[0].prompt_tokens);
    }
    // n == 1 keeps the response id and the request's own seed, byte-compatible with pre-task-20.
    let single = RecordingRequestEngine::new();
    let single_requests = Arc::clone(&single.requests);
    let app = router(AppState::new(Arc::new(single)).with_fixed_time(FIXED_TIME));
    let mut body = chat_body("hi");
    body["seed"] = json!(11);
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let response = body_json(&bytes);
    let recorded: Vec<EngineRequest> = single_requests.lock().expect("recorder").clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].request_id, response["id"]);
    assert_eq!(recorded[0].seed, Some(11));
}

/// An engine whose visible text is the seed it was called with.
///
/// This is the end-to-end independence probe: two choices are independent exactly when the router
/// handed the engine two different seeds, and the wire body shows that difference as text. The
/// inner MockEngine supplies the rest of the event stream (prompt accounting, usage, stats).
#[derive(Clone)]
struct SeedEchoEngine {
    inner: MockEngine,
}

impl SeedEchoEngine {
    fn new() -> Self {
        Self {
            inner: MockEngine::new(MODEL).content(&["a", "b"]),
        }
    }
}

impl InferenceEngine for SeedEchoEngine {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> futures::stream::BoxStream<'static, EngineEvent> {
        let text = request
            .seed
            .map_or_else(|| "seed=None".to_owned(), |seed| format!("seed={seed}"));
        let stream = self.inner.generate(request).map(move |event| match event {
            EngineEvent::ContentDelta(_) => EngineEvent::ContentDelta(text.clone()),
            other => other,
        });
        Box::pin(stream)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.inner.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
}

fn seed_echo_app(start: u64) -> Router {
    let state = AppState::new(Arc::new(SeedEchoEngine::new()))
        .with_fixed_time(FIXED_TIME)
        .with_choice_seeds(Arc::new(SequentialChoiceSeeds::starting_at(start)));
    router(state)
}

#[tokio::test]
async fn chat_n_unseeded_draws_an_independent_seed_per_choice() {
    // Before this, every unseeded choice reached the driver with seed None, which the host driver
    // maps to seed 0: n choices were n copies of one deterministic sample. The router now draws a
    // fresh seed per choice when the caller named none, so the choices are independent draws.
    let app = seed_echo_app(1000);
    let mut body = chat_body("hi");
    body["n"] = json!(2);
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let response = body_json(&bytes);
    let first = response["choices"][0]["message"]["content"]
        .as_str()
        .expect("choice 0 content");
    let second = response["choices"][1]["message"]["content"]
        .as_str()
        .expect("choice 1 content");
    assert_eq!(
        first, "seed=1000seed=1000",
        "the first draw is the first seed"
    );
    assert_eq!(
        second, "seed=1001seed=1001",
        "the second draw is a new seed"
    );
    assert_ne!(
        first, second,
        "unseeded n > 1 choices must be independent draws, not copies of seed 0"
    );
}

#[tokio::test]
async fn chat_streaming_n_unseeded_draws_an_independent_seed_per_choice() {
    // The streaming fan-out resolves its seeds separately from the buffered path; this pins that
    // the same per-choice draw reaches it (each choice's streamed text is its own seed).
    let app = seed_echo_app(2000);
    let mut body = chat_body("hi");
    body["n"] = json!(2);
    body["stream"] = json!(true);
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let frames = sse_frames(&bytes);
    let mut choice_0 = String::new();
    let mut choice_1 = String::new();
    for frame in &frames {
        for choice in frame["choices"].as_array().into_iter().flatten() {
            let Some(text) = choice["delta"]["content"].as_str() else {
                continue;
            };
            match choice["index"].as_u64() {
                Some(0) => choice_0.push_str(text),
                Some(1) => choice_1.push_str(text),
                other => panic!("unexpected choice index {other:?}"),
            }
        }
    }
    assert_eq!(choice_0, "seed=2000seed=2000");
    assert_eq!(choice_1, "seed=2001seed=2001");
    assert_ne!(
        choice_0, choice_1,
        "streaming unseeded n > 1 choices must be independent draws"
    );
}

#[tokio::test]
async fn unseeded_n_one_keeps_the_seed_none_path() {
    // n == 1 byte compatibility: an unseeded single choice still reaches the engine with seed
    // None, exactly as before per-choice seed resolution was added.
    let app = seed_echo_app(1000);
    let (status, _, bytes) = send(
        app.clone(),
        json_request("/v1/chat/completions", &chat_body("hi")),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body_json(&bytes)["choices"][0]["message"]["content"],
        "seed=Noneseed=None"
    );
    // And it does not consume a draw from the seed source: the next unseeded n > 1 request still
    // starts at the configured first seed.
    let mut body = chat_body("hi");
    body["n"] = json!(2);
    let (_, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    let response = body_json(&bytes);
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "seed=1000seed=1000"
    );
}

#[tokio::test]
async fn completions_n_unseeded_draws_an_independent_seed_per_prompt_choice_pair() {
    // Multi-prompt ordering: each (prompt, choice) pair is its own generate call and gets its own
    // fresh seed, so a two-prompt n=2 request draws four independent seeds. Buffered completions
    // keep prompt-major choice ordering.
    let app = seed_echo_app(500);
    let mut body = json!({
        "model": MODEL,
        "prompt": ["one", "two"],
        "n": 2,
        "temperature": 1.0,
    });
    body["max_tokens"] = json!(4);
    let (status, _, bytes) = send(app, json_request("/v1/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let response = body_json(&bytes);
    let texts: Vec<&str> = response["choices"]
        .as_array()
        .expect("choices")
        .iter()
        .map(|choice| choice["text"].as_str().expect("choice text"))
        .collect();
    assert_eq!(texts.len(), 4);
    assert_eq!(texts[0], "seed=500seed=500");
    assert_eq!(texts[1], "seed=501seed=501");
    assert_eq!(texts[2], "seed=502seed=502");
    assert_eq!(texts[3], "seed=503seed=503");
    for index in 0..texts.len() {
        for other in index + 1..texts.len() {
            assert_ne!(
                texts[index], texts[other],
                "choices must be independent draws: {texts:?}"
            );
        }
    }
}

#[tokio::test]
async fn seeded_n_choices_stay_reproducible_and_distinct() {
    // The caller-supplied seed keeps vLLM's seed + i derivation: a repeat of the same request
    // reproduces the same choices, and the choices differ from each other.
    let app = seed_echo_app(1000);
    let mut body = chat_body("hi");
    body["n"] = json!(2);
    body["seed"] = json!(7);
    let (_, _, first_bytes) = send(app.clone(), json_request("/v1/chat/completions", &body)).await;
    let (_, _, second_bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    let first = body_json(&first_bytes);
    let second = body_json(&second_bytes);
    assert_eq!(first["choices"][0]["message"]["content"], "seed=7seed=7");
    assert_eq!(first["choices"][1]["message"]["content"], "seed=8seed=8");
    assert_eq!(
        first["choices"], second["choices"],
        "a seeded request must reproduce its choices"
    );
}

#[tokio::test]
async fn chat_n_zero_is_refused_before_generation() {
    let engine = RecordingEngine::default();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let mut body = chat_body("hi");
    body["n"] = json!(0);
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes);
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["param"], "n");
    assert!(
        error["error"]["message"]
            .as_str()
            .expect("message")
            .contains("must be at least 1"),
        "{error}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn chat_n_above_one_is_refused_on_an_engine_with_shared_sampling() {
    let engine = SharedSamplingEngine {
        calls: Arc::new(AtomicUsize::new(0)),
        inner: MockEngine::new(MODEL),
    };
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let mut body = chat_body("hi");
    body["n"] = json!(2);
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes);
    assert_eq!(error["error"]["param"], "n");
    assert!(
        error["error"]["message"]
            .as_str()
            .expect("message")
            .contains("independently"),
        "{error}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the refusal is before generation"
    );
}

#[tokio::test]
async fn chat_n_two_streams_each_choice_and_aggregates_usage() {
    let app = app();
    let baseline = body_json(
        &send(
            app.clone(),
            json_request("/v1/chat/completions", &chat_body("hi")),
        )
        .await
        .2,
    );
    let mut body = chat_body("hi");
    body["n"] = json!(2);
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert!(
        String::from_utf8_lossy(&bytes).ends_with("data: [DONE]\n\n"),
        "the stream closes with [DONE]"
    );
    let frames = sse_frames(&bytes);
    let role_frames: Vec<&Value> = frames
        .iter()
        .filter(|frame| frame["choices"][0]["delta"]["role"] == "assistant")
        .collect();
    assert_eq!(role_frames.len(), 1, "one role chunk opens the response");
    assert_eq!(role_frames[0]["choices"][0]["index"], 0);
    let finish_frames: Vec<&Value> = frames
        .iter()
        .filter(|frame| {
            frame["choices"]
                .as_array()
                .is_some_and(|choices| !choices.is_empty())
                && frame["choices"][0]["finish_reason"].is_string()
        })
        .collect();
    assert_eq!(finish_frames.len(), 2);
    assert_eq!(finish_frames[0]["choices"][0]["index"], 0);
    assert_eq!(finish_frames[1]["choices"][0]["index"], 1);
    assert!(
        finish_frames[0].get("x_deepseek_spark_stats").is_none(),
        "only the last choice carries the aggregate extension"
    );
    assert!(
        finish_frames[1]["x_deepseek_spark_stats"].is_object(),
        "the last choice carries the aggregate extension"
    );
    let usage_frame = frames
        .iter()
        .find(|frame| {
            frame["choices"]
                .as_array()
                .is_some_and(|choices| choices.is_empty())
                && frame.get("usage").is_some()
        })
        .expect("usage chunk");
    let baseline_usage = &baseline["usage"];
    assert_eq!(
        usage_frame["usage"]["prompt_tokens"],
        baseline_usage["prompt_tokens"]
    );
    assert_eq!(
        usage_frame["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion"),
        baseline_usage["completion_tokens"]
            .as_u64()
            .expect("baseline completion")
            * 2
    );
    let choices: Vec<u64> = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["index"].as_u64())
        .collect();
    assert_eq!(choices, vec![0, 0, 0, 0, 1, 1, 1]);
}

#[tokio::test]
async fn completions_n_two_choices_per_prompt_keep_prompt_major_indexes() {
    let app = app();
    let baseline_body = json!({"model": MODEL, "prompt": ["aa", "bb"]});
    let baseline = body_json(
        &send(app.clone(), json_request("/v1/completions", &baseline_body))
            .await
            .2,
    );
    let body = json!({"model": MODEL, "prompt": ["aa", "bb"], "n": 2});
    let (status, _, bytes) = send(app, json_request("/v1/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let response = body_json(&bytes);
    let choices = response["choices"].as_array().expect("choices");
    let indexes: Vec<u64> = choices
        .iter()
        .map(|choice| choice["index"].as_u64().expect("index"))
        .collect();
    assert_eq!(indexes, vec![0, 1, 2, 3]);
    let baseline_usage = &baseline["usage"];
    let usage = &response["usage"];
    assert_eq!(
        usage["prompt_tokens"], baseline_usage["prompt_tokens"],
        "a prompt is counted once however many choices it yields"
    );
    assert_eq!(
        usage["completion_tokens"].as_u64().expect("completion"),
        baseline_usage["completion_tokens"]
            .as_u64()
            .expect("baseline completion")
            * 2
    );
}

#[tokio::test]
async fn completions_n_two_streams_choices_and_one_prompt_usage() {
    let app = app();
    let baseline = body_json(
        &send(
            app.clone(),
            json_request("/v1/completions", &json!({"model": MODEL, "prompt": "aa"})),
        )
        .await
        .2,
    );
    let body = json!({
        "model": MODEL,
        "prompt": "aa",
        "n": 2,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let (status, _, bytes) = send(app, json_request("/v1/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let frames = sse_frames(&bytes);
    let finish_indexes: Vec<u64> = frames
        .iter()
        .filter(|frame| {
            frame["choices"]
                .as_array()
                .is_some_and(|choices| !choices.is_empty())
                && frame["choices"][0]["finish_reason"].is_string()
        })
        .map(|frame| frame["choices"][0]["index"].as_u64().expect("index"))
        .collect();
    assert_eq!(finish_indexes, vec![0, 1]);
    let usage_frame = frames
        .iter()
        .find(|frame| frame["choices"].as_array().is_some_and(Vec::is_empty))
        .expect("usage chunk");
    let baseline_usage = &baseline["usage"];
    assert_eq!(
        usage_frame["usage"]["prompt_tokens"],
        baseline_usage["prompt_tokens"]
    );
    assert_eq!(
        usage_frame["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion"),
        baseline_usage["completion_tokens"]
            .as_u64()
            .expect("baseline completion")
            * 2
    );
}

#[tokio::test]
async fn completions_n_zero_is_refused_before_generation() {
    let engine = RecordingEngine::default();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let body = json!({"model": MODEL, "prompt": "aa", "n": 0});
    let (status, _, bytes) = send(app, json_request("/v1/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["param"], "n");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn response_format_text_is_a_no_op_and_json_modes_are_refused_typed() {
    let mut text = chat_body("hi");
    text["response_format"] = json!({"type": "text"});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &text)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the text format is the OpenAI default and a no-op: {}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(
        body_json(&bytes)["choices"][0]["message"]["content"],
        "Hello world"
    );

    for response_format in [
        json!({"type": "json_object"}),
        json!({"type": "json_schema", "json_schema": {
            "name": "answer",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        }}),
    ] {
        let mut body = chat_body("hi");
        body["response_format"] = response_format.clone();
        let engine = RecordingEngine::default();
        let calls = Arc::clone(&engine.calls);
        let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
        let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{}", response_format);
        let error = body_json(&bytes);
        assert_eq!(error["error"]["param"], "response_format");
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(
            error["error"]["message"]
                .as_str()
                .expect("message")
                .contains("constrained decoding"),
            "{error}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "the structured-output refusal is before generation"
        );
    }
}

#[tokio::test]
async fn response_format_malformed_json_schema_is_an_invalid_body() {
    let engine = RecordingEngine::default();
    let calls = Arc::clone(&engine.calls);
    let app = router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME));
    let mut body = chat_body("hi");
    body["response_format"] = json!({"type": "json_schema"});
    let (status, _, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["type"], "invalid_request_error");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn bearer_key_protects_every_route_except_health_and_metrics() {
    let app = keyed_app("s3cret");
    // Missing Authorization.
    let (status, _, bytes) = send(app.clone(), get_request("/v1/models")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(
        body_json(&bytes),
        json!({"error": {
            "message": "Unauthorized",
            "type": "invalid_request_error",
            "param": null,
            "code": 401,
        }})
    );
    // Wrong key and wrong scheme are both 401.
    for header in ["Bearer wrong", "Basic s3cret", "s3cret", ""] {
        let (status, _, _) =
            send(app.clone(), authorized_request("/v1/models", None, header)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "header: {header:?}");
    }
    // Correct key, and the scheme is case-insensitive.
    for header in ["Bearer s3cret", "bearer s3cret", "BEARER s3cret"] {
        let (status, _, bytes) =
            send(app.clone(), authorized_request("/v1/models", None, header)).await;
        assert_eq!(status, StatusCode::OK, "header: {header:?}");
        assert!(body_json(&bytes)["data"].is_array());
    }
    // POST endpoints are protected too, and the missing-key answer is the JSON envelope, not SSE.
    let (status, headers, _) = send(
        app.clone(),
        json_request("/v1/chat/completions", &chat_body("hi")),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_ne!(
        headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    // Liveness, readiness and the local scrape stay open.
    for uri in ["/health", "/readyz", "/metrics"] {
        let (status, _, _) = send(app.clone(), get_request(uri)).await;
        assert_eq!(status, StatusCode::OK, "{uri} must stay reachable");
    }
}

#[tokio::test]
async fn no_api_key_keeps_the_server_open() {
    let app = app();
    for uri in ["/v1/models", "/health", "/readyz", "/metrics"] {
        let (status, _, _) = send(app.clone(), get_request(uri)).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
    }
    let (status, _, _) = send(app, json_request("/v1/chat/completions", &chat_body("hi"))).await;
    assert_eq!(status, StatusCode::OK);
}

// ---------------------------------------------------------------------------------------
// Device-chain token envelope scaffold (S7/S8 preparation)
//
// These tests use a synthetic device token source only. The nine tokens, the 130-token prompt
// count and the streamed text are paper values: nothing here was measured on a device, and no
// tok/s, TTFT or real token claim is asserted or implied. The host-twin fallback tests prove the
// existing engine path is still the answer whenever the scaffold cannot serve a request.
// ---------------------------------------------------------------------------------------

/// A synthetic device token source: records the claim request and answers claimed or fallback.
struct TestDeviceTokenSource {
    claimed: bool,
    calls: Arc<AtomicUsize>,
    last: Arc<std::sync::Mutex<Option<DeviceTokenRequest>>>,
}

impl TestDeviceTokenSource {
    fn new(claimed: bool) -> Self {
        Self {
            claimed,
            calls: Arc::new(AtomicUsize::new(0)),
            last: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    fn calls(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.calls)
    }
}

impl DeviceTokenSource for TestDeviceTokenSource {
    fn claim(&self, request: &DeviceTokenRequest) -> DeviceTokenClaim {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last.lock().expect("device source request log") = Some(request.clone());
        if !self.claimed {
            return DeviceTokenClaim::Fallback {
                reason: "synthetic host-twin fallback".to_owned(),
            };
        }
        // Nine synthetic tokens, matching the scaffold's fixed-window contract. The text is a
        // placeholder for the decoder; no device token exists for it.
        let tokens = (0..9u32)
            .map(|index| DeviceToken {
                position: 129 + index,
                token_id: 11 + index,
                text: match index {
                    0 => "Hello".to_owned(),
                    1 => " device".to_owned(),
                    _ => String::new(),
                },
            })
            .collect();
        DeviceTokenClaim::Claimed(Box::new(DeviceTokenStream {
            tokens,
            prompt_tokens: 130,
            finish_reason: FinishReason::Length,
            reference_ids: vec![11, 12, 13, 14, 15, 16, 17, 18, 19],
            decode_positions: 9,
            logits_dump_present: false,
            claim_mode: ClaimMode::Asserted,
            source: "device-chain-envelope",
            provenance: "synthetic test stream; UNMEASURED".to_owned(),
        }))
    }
}

/// The device source's last observed claim request.
type DeviceSourceLog = Arc<std::sync::Mutex<Option<DeviceTokenRequest>>>;

fn device_app(claimed: bool) -> (Router, Arc<AtomicUsize>, Arc<AtomicUsize>, DeviceSourceLog) {
    let engine = RecordingMockEngine::with_content(&["Hello", " world"]);
    let engine_calls = Arc::clone(&engine.calls);
    let source = Arc::new(TestDeviceTokenSource::new(claimed));
    let source_calls = source.calls();
    let last = Arc::clone(&source.last);
    let app = router(
        AppState::new(Arc::new(engine))
            .with_fixed_time(FIXED_TIME)
            .with_device_token_source(Some(source)),
    );
    (app, engine_calls, source_calls, last)
}

#[tokio::test]
async fn device_claimed_completions_buffered_serve_the_envelope_and_never_call_the_engine() {
    let (app, engine_calls, source_calls, last) = device_app(true);
    let request = json!({
        "model": MODEL,
        "prompt": [11, 22, 33],
        "max_tokens": 5,
    });
    let (status, headers, bytes) = send(app, json_request("/v1/completions", &request)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(
        headers
            .get(DEVICE_TOKEN_SOURCE_HEADER)
            .expect("the device provenance header is present"),
        DEVICE_TOKEN_SOURCE_UNMEASURED
    );
    assert_eq!(
        headers
            .get(DEVICE_TOKEN_CLAIM_HEADER)
            .expect("the claim form header is present"),
        "explicit"
    );
    let body = body_json(&bytes);
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["id"], "cmpl-dsv41-00000000");
    assert_eq!(body["created"], FIXED_TIME);
    assert_eq!(body["model"], MODEL);
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][0]["text"], "Hello device");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["prompt_tokens"], 130);
    assert_eq!(body["usage"]["completion_tokens"], 9);
    assert_eq!(body["usage"]["total_tokens"], 139);
    // The scaffold makes no stats/tok-s/TTFT claim: the extension is absent, not zeroed.
    assert!(body["x_deepseek_spark_stats"].is_null());
    assert_eq!(
        engine_calls.load(Ordering::SeqCst),
        0,
        "the engine must not be entered"
    );
    assert_eq!(source_calls.load(Ordering::SeqCst), 1);
    let captured = last
        .lock()
        .expect("device source request log")
        .clone()
        .expect("device source was called");
    assert_eq!(captured.endpoint, DeviceEndpoint::Completions);
    assert!(!captured.stream);
    assert!(!captured.include_usage);
    assert_eq!(captured.max_tokens, Some(5));
    assert_eq!(captured.prompt_token_ids, Some(vec![11, 22, 33]));
}

#[tokio::test]
async fn device_claimed_completions_stream_the_same_answer_with_an_optional_usage_chunk() {
    let (app, engine_calls, _source_calls, _last) = device_app(true);
    let request = json!({
        "model": MODEL,
        "prompt": [11, 22, 33],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let (status, headers, bytes) = send(app, json_request("/v1/completions", &request)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers.get(DEVICE_TOKEN_SOURCE_HEADER).expect("header"),
        DEVICE_TOKEN_SOURCE_UNMEASURED
    );
    assert_eq!(
        headers
            .get(DEVICE_TOKEN_CLAIM_HEADER)
            .expect("claim header"),
        "explicit"
    );
    assert!(
        bytes.ends_with(b"data: [DONE]\n\n"),
        "the stream must terminate with [DONE]: {}",
        String::from_utf8_lossy(&bytes)
    );
    let frames = sse_frames(&bytes);
    // Nine content chunks, one finish chunk and the include_usage chunk.
    assert_eq!(frames.len(), 11);
    assert_eq!(frames[0]["choices"][0]["text"], "Hello");
    assert_eq!(frames[1]["choices"][0]["text"], " device");
    assert_eq!(frames[9]["choices"][0]["finish_reason"], "length");
    assert_eq!(frames[10]["choices"], serde_json::json!([]));
    assert_eq!(frames[10]["usage"]["prompt_tokens"], 130);
    assert_eq!(frames[10]["usage"]["completion_tokens"], 9);
    assert_eq!(engine_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn device_claimed_chat_serves_the_envelope_body() {
    let (app, engine_calls, _source_calls, last) = device_app(true);
    let (status, headers, bytes) =
        send(app, json_request("/v1/chat/completions", &chat_body("hi"))).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(
        headers.get(DEVICE_TOKEN_SOURCE_HEADER).expect("header"),
        DEVICE_TOKEN_SOURCE_UNMEASURED
    );
    assert_eq!(
        headers
            .get(DEVICE_TOKEN_CLAIM_HEADER)
            .expect("claim header"),
        "explicit"
    );
    let body = body_json(&bytes);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello device");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["prompt_tokens"], 130);
    assert!(body["x_deepseek_spark_stats"].is_null());
    assert_eq!(engine_calls.load(Ordering::SeqCst), 0);
    let captured = last
        .lock()
        .expect("device source request log")
        .clone()
        .expect("device source was called");
    assert_eq!(captured.endpoint, DeviceEndpoint::ChatCompletions);
    assert_eq!(captured.prompt_token_ids, None);
}

#[tokio::test]
async fn device_claimed_chat_with_tokenizer_seam_captures_prompt_token_ids() {
    let engine = RecordingMockEngine::with_content(&["Hello", " world"]);
    let engine_calls = Arc::clone(&engine.calls);
    let source = Arc::new(TestDeviceTokenSource::new(true));
    let last = Arc::clone(&source.last);
    let seam = Arc::new(
        sparkflash_api::encoding::TokenizerSeam::from_path(
            &sparkflash_api::encoding::reference_tokenizer_path(),
        )
        .expect("load reference tokenizer seam"),
    );
    let app = router(
        AppState::new(Arc::new(engine))
            .with_fixed_time(FIXED_TIME)
            .with_tokenizer(seam)
            .with_device_token_source(Some(source)),
    );
    let (status, _headers, bytes) =
        send(app, json_request("/v1/chat/completions", &chat_body("hi"))).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(engine_calls.load(Ordering::SeqCst), 0);
    let captured = last
        .lock()
        .expect("device source request log")
        .clone()
        .expect("device source was called");
    assert_eq!(captured.endpoint, DeviceEndpoint::ChatCompletions);
    let prompt_ids = captured
        .prompt_token_ids
        .expect("prompt token ids must be captured by tokenizer seam for W4");
    assert!(!prompt_ids.is_empty());
}

#[tokio::test]
async fn device_fallback_serves_the_host_twin_and_leaves_its_bytes_and_stats_intact() {
    let (app, engine_calls, source_calls, _last) = device_app(false);
    let (status, headers, bytes) =
        send(app, json_request("/v1/chat/completions", &chat_body("hi"))).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers.get(DEVICE_TOKEN_SOURCE_HEADER).is_none(),
        "a host-twin answer must not carry the device provenance header"
    );
    let body = body_json(&bytes);
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(body["x_deepseek_spark_stats"].is_object());
    assert_eq!(source_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        engine_calls.load(Ordering::SeqCst),
        1,
        "the host twin must serve the fallback"
    );
}

#[tokio::test]
async fn device_scaffold_never_silently_ignores_an_advanced_request_field() {
    let (app, engine_calls, source_calls, _last) = device_app(true);
    let mut body = chat_body("hi");
    body["temperature"] = json!(0.5);
    let (status, headers, bytes) = send(app, json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers.get(DEVICE_TOKEN_SOURCE_HEADER).is_none());
    let response = body_json(&bytes);
    assert_eq!(response["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(
        source_calls.load(Ordering::SeqCst),
        0,
        "an ineligible request must not even consult the device source"
    );
    assert_eq!(engine_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn device_scaffold_falls_back_for_a_text_prompt_completion() {
    let (app, engine_calls, _source_calls, _last) = device_app(true);
    let request = json!({"model": MODEL, "prompt": "hi", "max_tokens": 4});
    let (status, headers, bytes) = send(app, json_request("/v1/completions", &request)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers.get(DEVICE_TOKEN_SOURCE_HEADER).is_none());
    let body = body_json(&bytes);
    assert_eq!(body["choices"][0]["text"], "Hello world");
    assert_eq!(engine_calls.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------------------
// WS-F: CORS, tool policy, streaming tool calls with logprobs, continuous usage, endpoints
// ---------------------------------------------------------------------------------------

use sparkflash_api::state::CorsConfig;

/// A preflight request, the shape a browser sends before a cross-origin POST.
fn cors_preflight(uri: &str, origin: &str) -> Request<Body> {
    Request::builder()
        .method("OPTIONS")
        .uri(uri)
        .header("origin", origin)
        .header("access-control-request-method", "POST")
        .header(
            "access-control-request-headers",
            "authorization,content-type",
        )
        .body(Body::empty())
        .expect("request")
}

fn request_with_origin(uri: &str, body: &Value, origin: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .header("origin", origin)
        .body(Body::from(serde_json::to_vec(body).expect("json body")))
        .expect("request")
}

fn app_with_key() -> Router {
    router(
        AppState::new(Arc::new(engine_with(&["Hello", " world"])))
            .with_fixed_time(FIXED_TIME)
            .with_api_key(Some("s3cret".to_owned())),
    )
}

#[tokio::test]
async fn cors_preflight_is_answered_before_the_bearer_middleware() {
    // The browser preflight carries no Authorization header, so the CORS layer has to answer it
    // before auth would reject it; the reference vLLM server answers preflight first too.
    let (status, headers, _) = send(
        app_with_key(),
        cors_preflight("/v1/chat/completions", "https://app.example"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "preflight is not a 401");
    assert_eq!(
        headers
            .get("access-control-allow-origin")
            .expect("allow origin"),
        "*",
        "the reference server answers access-control-allow-origin: *"
    );
    let methods = headers
        .get("access-control-allow-methods")
        .expect("allow methods")
        .to_str()
        .expect("ascii");
    assert!(methods.contains("POST"), "POST is allowed: {methods}");
    let allow_headers = headers
        .get("access-control-allow-headers")
        .expect("allow headers")
        .to_str()
        .expect("ascii");
    assert!(
        allow_headers.to_ascii_lowercase().contains("authorization"),
        "the explicit list has to name Authorization (the Fetch wildcard does not cover it): {allow_headers}"
    );
    assert_eq!(
        headers.get("access-control-max-age").expect("max age"),
        "600"
    );

    // A real cross-origin request without the key still gets the 401 envelope, and the CORS layer
    // still stamps it so the browser can read the error instead of reporting a network failure.
    let (status, headers, bytes) = send(
        app_with_key(),
        request_with_origin(
            "/v1/chat/completions",
            &chat_body("hi"),
            "https://app.example",
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(
        headers
            .get("access-control-allow-origin")
            .expect("allow origin"),
        "*"
    );
    assert_eq!(body_json(&bytes)["error"]["code"], 401);
}

#[tokio::test]
async fn a_cors_origin_list_echoes_only_the_allowed_origin() {
    let app = || {
        router(
            AppState::new(Arc::new(engine_with(&["Hello"])))
                .with_fixed_time(FIXED_TIME)
                .with_cors(Some(CorsConfig::allow_list(vec![
                    "https://allowed.example".to_owned(),
                ]))),
        )
    };
    let (status, headers, _) = send(
        app(),
        request_with_origin(
            "/v1/chat/completions",
            &chat_body("hi"),
            "https://allowed.example",
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers
            .get("access-control-allow-origin")
            .expect("allow origin"),
        "https://allowed.example"
    );
    let (status, headers, _) = send(
        app(),
        request_with_origin(
            "/v1/chat/completions",
            &chat_body("hi"),
            "https://other.example",
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "the request itself is not refused");
    assert!(
        headers.get("access-control-allow-origin").is_none(),
        "an origin outside the list gets no allow-origin header"
    );
}

#[tokio::test]
async fn cors_can_be_turned_off_for_a_proxy_owned_policy() {
    let app = router(
        AppState::new(Arc::new(engine_with(&["Hello"])))
            .with_fixed_time(FIXED_TIME)
            .with_cors(None),
    );
    let (status, headers, _) = send(
        app,
        request_with_origin(
            "/v1/chat/completions",
            &chat_body("hi"),
            "https://app.example",
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers.get("access-control-allow-origin").is_none(),
        "the pre-CORS wire surface has no access-control-* headers"
    );
}

/// A two-call DSML answer, one string, for the parallel-tool-call tests.
fn two_call_dsml() -> String {
    [
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"first\">\n",
        "<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "<｜DSML｜ invoke name=\"second\">\n",
        "<｜DSML｜ parameter name=\"b\" string=\"false\">true</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    ]
    .concat()
}

#[tokio::test]
async fn parallel_tool_calls_false_reports_only_the_first_call_buffered() {
    let engine = engine_with(&[two_call_dsml().as_str()]);
    let mut body = tools_body(false);
    body["parallel_tool_calls"] = json!(false);
    let (status, _, bytes) = send(
        app_with(engine),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", body_json(&bytes));
    let body = body_json(&bytes);
    let calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls");
    assert_eq!(
        calls.len(),
        1,
        "the at-most-one contract is enforced at the parser boundary: {body}"
    );
    assert_eq!(calls[0]["function"]["name"], "first");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn parallel_tool_calls_false_reports_only_the_first_call_streamed() {
    let answer = two_call_dsml();
    let pieces = chunked(&answer, 5);
    let mut body = tools_body(true);
    body["parallel_tool_calls"] = json!(false);
    let (status, _, bytes) = send(
        app_with(engine_with(&stream_pieces(&pieces))),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let frames = sse_frames(&bytes);
    let calls = assemble_streamed_calls(&frames);
    assert_eq!(calls.len(), 1, "only the first call streams: {calls:?}");
    assert_eq!(calls[0].name, "first");
    assert_eq!(calls[0].arguments, r#"{"a": "x"}"#);
    assert_eq!(
        frames[frames.len() - 1]["choices"][0]["finish_reason"],
        "tool_calls"
    );
}

#[tokio::test]
async fn tool_choice_none_keeps_the_block_in_the_content() {
    let engine = engine_with(&[dsml_stream_answer().as_str()]);
    let mut body = tools_body(false);
    body["tool_choice"] = json!("none");
    let (status, _, bytes) = send(
        app_with(engine),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", body_json(&bytes));
    let body = body_json(&bytes);
    assert!(
        body["choices"][0]["message"]["tool_calls"].is_null(),
        "tool_choice none never lifts a call: {body}"
    );
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert!(
        content.contains("DSML"),
        "the block stays plain content instead of being parsed: {content}"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn tool_choice_required_and_named_are_refused_typed() {
    for choice in [
        json!("required"),
        json!({"type": "function", "function": {"name": "get_weather"}}),
    ] {
        let mut body = tools_body(false);
        body["tool_choice"] = choice;
        let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{}", body_json(&bytes));
        let body = body_json(&bytes);
        let error = &body["error"];
        assert_eq!(error["param"], "tool_choice");
        assert_eq!(
            error["code"], 400,
            "the wire code is the HTTP status, vLLM's ErrorInfo.code"
        );
        assert!(
            error["message"]
                .as_str()
                .expect("message")
                .contains("forces a tool call"),
            "the refusal names the reason: {body}"
        );
    }
    let mut body = tools_body(false);
    body["tool_choice"] = json!("bogus");
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["param"], "tool_choice");
    let mut body = tools_body(false);
    body["tool_choice"] = json!(5);
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["param"], "tool_choice");
}

/// A mock whose content pieces are the DSML answer split at character boundaries, with one
/// scripted logprob per piece - the engine contract the router pairs with each delta.
fn logprob_dsml_engine(pieces: &[String]) -> MockEngine {
    let logprobs = pieces
        .iter()
        .enumerate()
        .map(|(index, piece)| {
            let token_id = u32::try_from(index).expect("small index");
            TokenLogprob {
                token_id,
                token: piece.clone(),
                logprob: -0.5,
                rank: Some(1),
                top: vec![alternative(token_id, piece, -0.5, 1)],
            }
        })
        .collect();
    MockEngine::new(MODEL)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .with_capabilities(Capabilities::ALL)
        .content(&stream_pieces(pieces))
        .with_token_logprobs(logprobs)
}

#[tokio::test]
async fn streaming_tool_calls_and_logprobs_share_chunks_like_the_reference() {
    // Measured on the reference deployment 2026-10-21: with tools offered and logprobs requested,
    // vLLM parses the DSML block into delta.tool_calls and still reports every generated token's
    // logprob, including the marker tokens the parser removed from the content.
    let answer = dsml_stream_answer();
    let pieces = chunked(&answer, 7);
    let mut body = tools_body(false);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(0);
    let (status, _, buffered_bytes) = send(
        app_with(logprob_dsml_engine(&pieces)),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", body_json(&buffered_bytes));
    let buffered = body_json(&buffered_bytes);
    let buffered_calls: Vec<(String, String)> = buffered["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls")
        .iter()
        .map(|call| {
            (
                call["function"]["name"].as_str().expect("name").to_owned(),
                call["function"]["arguments"]
                    .as_str()
                    .expect("arguments")
                    .to_owned(),
            )
        })
        .collect();
    assert_eq!(buffered_calls.len(), 1);

    let mut body = tools_body(true);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(0);
    let (status, _, streamed_bytes) = send(
        app_with(logprob_dsml_engine(&pieces)),
        json_request("/v1/chat/completions", &body),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&streamed_bytes)
    );
    let frames = sse_frames(&streamed_bytes);
    let calls = assemble_streamed_calls(&frames);
    assert_eq!(
        calls
            .iter()
            .map(|call| (call.name.clone(), call.arguments.clone()))
            .collect::<Vec<_>>(),
        buffered_calls,
        "the streamed fragments reassemble the buffered answer"
    );
    assert_eq!(
        streamed_content(&frames),
        buffered["choices"][0]["message"]["content"]
            .as_str()
            .expect("buffered content"),
        "the content before the block still streams as content"
    );
    // Every generated token is scored exactly once, across all chunks, including the DSML markers.
    let mut scored = Vec::new();
    for frame in &frames {
        for entry in frame["choices"][0]["logprobs"]["content"]
            .as_array()
            .into_iter()
            .flatten()
        {
            scored.push(entry["token"].as_str().expect("token").to_owned());
        }
    }
    assert_eq!(
        scored, pieces,
        "one logprob entry per generated token, in order"
    );
    assert_eq!(
        frames[frames.len() - 1]["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert!(
        !streamed_content(&frames).contains("DSML"),
        "the DSML block is never content"
    );
}

#[tokio::test]
async fn continuous_usage_stats_put_the_running_usage_on_every_chunk() {
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true, "continuous_usage_stats": true});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let frames = sse_frames(&bytes);
    assert!(
        frames.iter().all(|frame| frame["usage"].is_object()),
        "every chunk carries usage: {frames:?}"
    );
    assert_eq!(frames[0]["usage"]["prompt_tokens"], 8);
    assert_eq!(frames[0]["usage"]["completion_tokens"], 0);
    // The choice chunks grow the completion count; the final choices=[] chunk repeats the total.
    let completions: Vec<u64> = frames
        .iter()
        .map(|frame| frame["usage"]["completion_tokens"].as_u64().expect("count"))
        .collect();
    assert_eq!(completions.first(), Some(&0));
    assert_eq!(
        completions.last(),
        Some(&2),
        "the engine's total closes the stream"
    );
    assert!(completions.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(frames[frames.len() - 1]["usage"]["total_tokens"], 10);
}

#[tokio::test]
async fn continuous_usage_stats_without_include_usage_adds_no_usage() {
    // Measured on the reference deployment: continuous_usage_stats alone is ignored (no usage
    // block anywhere), so this server does not invent one.
    let mut body = chat_body("hi");
    body["stream"] = json!(true);
    body["stream_options"] = json!({"continuous_usage_stats": true});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let frames = sse_frames(&bytes);
    assert!(frames.iter().all(|frame| frame["usage"].is_null()));
}

#[tokio::test]
async fn models_retrieve_returns_the_same_card_as_the_list() {
    let (status, _, listed) = send(app(), get_request("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    let listed = body_json(&listed);
    let (status, _, card) = send(app(), get_request(&format!("/v1/models/{MODEL}"))).await;
    assert_eq!(status, StatusCode::OK);
    let card = body_json(&card);
    assert_eq!(card["id"], MODEL);
    assert_eq!(card["object"], "model");
    assert_eq!(card["max_model_len"], listed["data"][0]["max_model_len"]);
    assert_eq!(
        card["capabilities"], listed["data"][0]["capabilities"],
        "the retrieve route and the list route share one card"
    );
    let (status, _, bytes) = send(app(), get_request("/v1/models/no-such-model")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body_json(&bytes)["error"]["param"], "model");
}

#[tokio::test]
async fn best_of_is_accepted_as_the_documented_no_op() {
    // Measured on the reference deployment 2026-10-21: best_of=2 with n absent answers one choice.
    let (status, _, bytes) = send(
        app(),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hello", "best_of": 2}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", body_json(&bytes));
    assert_eq!(
        body_json(&bytes)["choices"]
            .as_array()
            .expect("choices")
            .len(),
        1
    );
}

#[tokio::test]
async fn completions_response_format_is_refused_like_chat() {
    let (status, _, bytes) = send(
        app(),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hi", "response_format": {"type": "json_object"}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["param"], "response_format");
}

#[tokio::test]
async fn logit_bias_and_min_p_are_validated_then_refused_typed() {
    // Non-integer key: vLLM's own conversion error text.
    let mut body = chat_body("hi");
    body["logit_bias"] = json!({"1.5": 1});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes)["error"].clone();
    assert_eq!(error["param"], "logit_bias");
    assert!(
        error["message"]
            .as_str()
            .expect("message")
            .contains("cannot be converted to integer token IDs")
    );

    let mut body = chat_body("hi");
    body["logit_bias"] = json!({"7": 200});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body_json(&bytes)["error"]["message"]
            .as_str()
            .expect("message")
            .contains("between -100 and 100")
    );

    let mut body = chat_body("hi");
    body["logit_bias"] = json!({"7": 1});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body_json(&bytes)["error"].clone();
    assert_eq!(error["param"], "logit_bias");
    assert!(
        error["message"]
            .as_str()
            .expect("message")
            .contains("logits processor")
    );

    // The disabled values are no-ops, not refusals.
    let mut body = chat_body("hi");
    body["logit_bias"] = json!({});
    body["min_p"] = json!(0.0);
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK, "{}", body_json(&bytes));

    // min_p > 0 needs the candidate filter the seam does not have; out of range is a plain 400.
    let mut body = chat_body("hi");
    body["min_p"] = json!(0.5);
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["param"], "min_p");
    assert!(
        body_json(&bytes)["error"]["message"]
            .as_str()
            .expect("message")
            .contains("candidate filter")
    );
    let mut body = chat_body("hi");
    body["min_p"] = json!(2.0);
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body_json(&bytes)["error"]["message"]
            .as_str()
            .expect("message")
            .contains("min_p must be between 0 and 1")
    );

    // The completions schema carries the same validation.
    let (status, _, bytes) = send(
        app(),
        json_request(
            "/v1/completions",
            &json!({"model": MODEL, "prompt": "hi", "logit_bias": {"3": -1}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body_json(&bytes)["error"]["param"], "logit_bias");
}
