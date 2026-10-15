//! End-to-end router tests: no GPU, no checkpoint, no sockets.
//!
//! Every case drives the real axum Router through tower::ServiceExt::oneshot against
//! MockEngine, so response bodies (including exact SSE framing) are asserted directly.

use std::io::Read as _;
use std::sync::{Arc, OnceLock};
use tokio::sync::{Mutex, MutexGuard};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::engine::{ClusterStatsSnapshot, MockEngine};
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
    MockEngine::new(MODEL)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .content(pieces)
}

fn app() -> Router {
    router(AppState::new(Arc::new(engine_with(&["Hello", " world"]))).with_fixed_time(FIXED_TIME))
}

fn app_with(engine: MockEngine) -> Router {
    router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME))
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
async fn image_url_content_parts_are_accepted_and_counted() {
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]}],
    });
    let (status, _, bytes) = send(app(), json_request("/v1/chat/completions", &body)).await;
    assert_eq!(status, StatusCode::OK);
    let body = body_json(&bytes);
    assert_eq!(body["usage"]["prompt_tokens_details"]["image_tokens"], 121);
    assert_eq!(body["usage"]["prompt_tokens"], 8 + 121);
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
