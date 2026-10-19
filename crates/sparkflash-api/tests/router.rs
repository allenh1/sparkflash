//! End-to-end router tests: no GPU, no checkpoint, no sockets.
//!
//! Every case drives the real axum Router through tower::ServiceExt::oneshot against
//! MockEngine, so response bodies (including exact SSE framing) are asserted directly.

use std::io::Read as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use tokio::sync::{Mutex, MutexGuard};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, InferenceEngine,
    LogprobAlternative, MissingCapability, MockEngine, PromptLogprobs, TokenLogprob, TokenString,
};
use sparkflash_api::server::{disarm_traces, router};
use sparkflash_api::state::AppState;
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
    r#"data: {"id":"chatcmpl-dsv41-00000000","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-ai/DeepSeek-V4.1-Flash","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":0,"uma_cached_tokens":0,"nvme_cached_tokens":0,"image_tokens":0},"completion_tokens_details":{"reasoning_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0}}}"#,
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
    assert_eq!(body_json(&bytes)["error"]["code"], "invalid_thinking");
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
        assert_eq!(error["error"]["code"], MissingCapability::vision().code());
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
        MissingCapability::vision().code()
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
            MissingCapability::vision().code(),
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
            MissingCapability::vision().code(),
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
            MissingCapability::vision().code(),
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
    assert_eq!(error["error"]["code"], "invalid_image_data_url");
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
    assert_eq!(error["error"]["code"], "invalid_json");
    assert!(error["error"]["message"].is_string());
}

#[tokio::test]
async fn unknown_model_returns_404() {
    let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let error = body_json(&bytes);
    assert_eq!(error["error"]["code"], "model_not_found");
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
    assert_eq!(body_json(&bytes)["error"]["code"], "invalid_trace_scopes");

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
    for bad in [json!(0), json!(65_537), json!(1_000_000)] {
        let mut body = chat_body("hi");
        body["max_tokens"] = bad.clone();
        let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "max_tokens {bad}");
        let error = body_json(&bytes);
        assert_eq!(error["error"]["code"], "invalid_max_tokens");
        assert_eq!(error["error"]["param"], "max_tokens");
    }
    let mut body = chat_body("hi");
    body["max_tokens"] = json!(65_536);
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
    assert_eq!(error["error"]["code"], "unsupported_parameter");
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
    assert_eq!(error["error"]["code"], "unsupported_parameter");
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
            MissingCapability::vision().code(),
            "body: {body}"
        );
    }
    let (status, _, bytes) = send(app.clone(), escaped).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(&bytes)["error"]["code"],
        MissingCapability::vision().code()
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
        "response_format": {"type": "json_schema", "json_schema": {"schema":
            {"type": "object", "$defs": {"image_url": {"type": "object", "minProperties": 1}}}}},
    });
    for body in [
        tool_schema,
        response_format,
        min_properties,
        draft07_dependencies,
        defs_min_properties,
    ] {
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
        5,
        "generation must be entered for every schema-named text request"
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
            MissingCapability::vision().code(),
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
        assert_eq!(body["error"]["code"], "invalid_logprobs", "body: {case}");
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
    assert_eq!(body["error"]["code"], "unsupported_parameter");
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
