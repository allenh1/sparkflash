//! /v1/completions batch semantics: one choice per prompt, per-prompt finish reasons, usage summed
//! over the batch, and the streamed form's per-prompt choice index.
//!
//! The contract recorded here is vLLM's: serving_completion.py builds one
//! CompletionResponseChoice per prompt (index = choice position), merges one result generator per
//! prompt for streaming, and sums prompt/completion tokens into the single usage block. The prompt
//! ids come from the real checkpoint tokenizer (the seam), so prompt_tokens assertions are the
//! tokenizer's count, not the Milestone 1 estimate.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::encoding::{TokenizerSeam, reference_tokenizer_path};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, InferenceEngine,
    MockEngine,
};
use sparkflash_api::server::router;
use sparkflash_api::state::AppState;
use sparkflash_api::{
    CompletionTokensDetails, DSparkStepStats, DeepSeekSparkRequestStats, MODEL_ID,
    OpenAiUsageWithCache, PromptTokensDetails, RequestTimingStats, RoceTransportStats,
};
use sparkflash_engram::EngramTelemetrySnapshot;
use tower::ServiceExt as _;

const FIXED_TIME: u64 = 1_700_000_000;

fn seam() -> Arc<TokenizerSeam> {
    Arc::new(
        TokenizerSeam::from_path(&reference_tokenizer_path())
            .expect("the vendored reference tokenizer loads"),
    )
}

fn app_with_engine(engine: Arc<dyn InferenceEngine>) -> Router {
    router(
        AppState::new(engine)
            .with_fixed_time(FIXED_TIME)
            .with_tokenizer(seam()),
    )
}

fn mock_engine() -> Arc<dyn InferenceEngine> {
    Arc::new(
        MockEngine::new(MODEL_ID)
            .with_cluster_stats(ClusterStatsSnapshot::default())
            .with_capabilities(Capabilities::ALL)
            .content(&["Hello", " world"]),
    )
}

async fn post(app: Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

async fn post_stream(app: Router, path: &str, body: Value) -> (StatusCode, Vec<Value>) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let text = String::from_utf8(bytes.to_vec()).expect("SSE is UTF-8");
    let events = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|frame| *frame != "[DONE]")
        .map(|frame| serde_json::from_str(frame).expect("an SSE JSON frame"))
        .collect();
    (status, events)
}

#[tokio::test]
async fn two_prompts_produce_two_choices_in_one_response() {
    let (status, body) = post(
        app_with_engine(mock_engine()),
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": ["Hello", "Hello, world!"],
            "echo": true,
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "text_completion", "{body}");
    let choices = body["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 2, "{body}");
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(choices[1]["index"], 1);
    // Each choice echoes its own prompt; the mock engine's generated text is "Hello world".
    assert_eq!(choices[0]["text"], "HelloHello world", "{body}");
    assert_eq!(choices[1]["text"], "Hello, world!Hello world", "{body}");
    assert_eq!(choices[0]["stop_reason"], Value::Null);
}

#[tokio::test]
async fn choice_order_follows_prompt_order() {
    let (status, body) = post(
        app_with_engine(mock_engine()),
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": ["first", "second", "third"],
            "echo": true,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let texts: Vec<&str> = body["choices"]
        .as_array()
        .expect("choices")
        .iter()
        .map(|choice| choice["text"].as_str().expect("text"))
        .collect();
    assert_eq!(texts, ["firstHello", "secondHello", "thirdHello"]);
}

#[tokio::test]
async fn usage_sums_over_the_batch() {
    let (status, body) = post(
        app_with_engine(mock_engine()),
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": ["Hello", "Hello, world!"],
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // The checkpoint tokenizer gives "Hello" 1 id and "Hello, world!" 4; the mock generates two
    // tokens for each prompt.
    assert_eq!(body["usage"]["prompt_tokens"], 5, "{body}");
    assert_eq!(body["usage"]["completion_tokens"], 4, "{body}");
    assert_eq!(body["usage"]["total_tokens"], 9, "{body}");
    assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    // The extension stats aggregate the per-prompt cache plan too.
    assert_eq!(
        body["x_deepseek_spark_stats"]["kv_cache"]["total_prompt_tokens"], 5,
        "{body}"
    );
    assert_eq!(
        body["x_deepseek_spark_stats"]["kv_cache"]["uncached_suffix_tokens"], 5,
        "{body}"
    );
}

/// An engine that finishes short prompts with "stop" and longer prompts with "length", so the
/// per-choice finish reason of a batch is observable.
struct PromptAwareEngine {
    model: String,
    cluster: ClusterStatsSnapshot,
}

fn engram_snapshot(layer_id: usize) -> EngramTelemetrySnapshot {
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

impl InferenceEngine for PromptAwareEngine {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn generate(&self, request: EngineRequest) -> futures::stream::BoxStream<'static, EngineEvent> {
        let prompt_tokens = request.prompt_tokens.len();
        let finish_reason = if prompt_tokens > 2 {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        let usage = OpenAiUsageWithCache {
            prompt_tokens,
            completion_tokens: 1,
            total_tokens: prompt_tokens + 1,
            prompt_tokens_details: PromptTokensDetails::default(),
            completion_tokens_details: CompletionTokensDetails::default(),
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: prompt_tokens,
        };
        let stats = DeepSeekSparkRequestStats {
            kv_cache: sparkflash_api::engine::plan_for(prompt_tokens, 0, 0, false),
            kv_cache_hit_rate: 0.0,
            engram_layers: [engram_snapshot(1), engram_snapshot(14)],
            dspark_speculative: DSparkStepStats::from_counts(0, 0, 0, 0),
            transport: RoceTransportStats {
                fabric_mode: String::new(),
                rocenante_oneshot_calls: 0,
                nccl_ring_calls: 0,
                rail0_rocep1s0f0_mb: 0.0,
                rail1_rocep2p1s0f0_mb: 0.0,
                l2_weight_prefetch_overlaps: 0,
            },
            timing: RequestTimingStats {
                ttft_ms: 0.0,
                prefill_tok_per_s: 0.0,
                decode_tok_per_s: 0.0,
                min_rank_uma_free_gib: 0.0,
            },
        };
        let events = vec![
            EngineEvent::Started {
                prompt_tokens,
                cached_tokens: 0,
            },
            EngineEvent::ContentDelta(format!("p{prompt_tokens}")),
            EngineEvent::Finished {
                finish_reason,
                usage,
                stats: Box::new(stats),
            },
        ];
        Box::pin(futures::stream::iter(events))
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.cluster.clone()
    }
}

#[tokio::test]
async fn finish_reason_is_per_choice() {
    let engine = Arc::new(PromptAwareEngine {
        model: MODEL_ID.to_owned(),
        cluster: ClusterStatsSnapshot::default(),
    });
    let (status, body) = post(
        app_with_engine(engine),
        "/v1/completions",
        // No tokenizer seam text: the placeholder estimate gives "hi" one token and the long
        // prompt more than two, which is what the engine turns into stop vs length.
        json!({
            "model": MODEL_ID,
            "prompt": ["hi", "a much longer prompt for the length finish"],
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let choices = body["choices"].as_array().expect("choices");
    assert_eq!(choices[0]["finish_reason"], "stop", "{body}");
    assert_eq!(choices[1]["finish_reason"], "length", "{body}");
    // The engine reported one completion token per prompt, and usage is its own sum.
    assert_eq!(body["usage"]["completion_tokens"], 2, "{body}");
    assert_eq!(
        body["usage"]["total_tokens"],
        body["usage"]["prompt_tokens"].as_u64().unwrap() + 2,
        "{body}"
    );
}

#[tokio::test]
async fn a_single_prompt_keeps_the_previous_shape() {
    let (status, body) = post(
        app_with_engine(mock_engine()),
        "/v1/completions",
        json!({"model": MODEL_ID, "prompt": "Hello", "max_tokens": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let choices = body["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 1, "{body}");
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(body["usage"]["prompt_tokens"], 1, "{body}");
}

/// A single-prompt stream is not part of the batch change: its usage chunk stays usage-only, as
/// it was before task-40, so a batched stream cannot silently widen the single-prompt wire shape.
#[tokio::test]
async fn single_prompt_stream_usage_chunk_stays_usage_only() {
    let (status, events) = post_stream(
        app_with_engine(mock_engine()),
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": "Hello",
            "max_tokens": 2,
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let usage = events
        .iter()
        .find(|frame| frame.get("usage").is_some())
        .expect("an include_usage stream ends with a usage chunk");
    assert!(
        usage.get("x_deepseek_spark_stats").is_none(),
        "a single-prompt usage chunk keeps the pre-task-40 shape: {usage}"
    );
}

/// One HTTP request with N prompts is one metric request, with the batch's aggregate tokens -
/// vLLM's per-request accounting, not N engine requests.
#[tokio::test]
async fn a_batch_counts_as_one_request_with_aggregate_tokens() {
    let state = AppState::new(mock_engine())
        .with_fixed_time(FIXED_TIME)
        .with_tokenizer(seam());
    let metrics = Arc::clone(&state.metrics);
    let app = router(state);
    let (status, body) = post(
        app,
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": ["Hello", "Hello, world!"],
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let rendered = metrics.render();
    assert!(
        rendered.contains("sparkflash_requests_total 1"),
        "one HTTP request, one counter bump: {rendered}"
    );
    assert!(
        rendered.contains("sparkflash_prompt_tokens_total 5"),
        "the aggregate prompt tokens are counted: {rendered}"
    );
}

#[tokio::test]
async fn an_empty_prompt_list_is_a_400() {
    let (status, body) = post(
        app_with_engine(mock_engine()),
        "/v1/completions",
        json!({"model": MODEL_ID, "prompt": []}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], 400);
}

#[tokio::test]
async fn streaming_batch_chunks_carry_the_prompt_index() {
    let (status, events) = post_stream(
        app_with_engine(mock_engine()),
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": ["Hello", "Hello, world!"],
            "max_tokens": 2,
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let mut content_indices: Vec<u64> = Vec::new();
    let mut finish_indices: Vec<u64> = Vec::new();
    let mut usage_chunks: Vec<&Value> = Vec::new();
    for frame in &events {
        if frame.get("usage").is_some() {
            usage_chunks.push(frame);
            continue;
        }
        let choice = &frame["choices"][0];
        let index = choice["index"].as_u64().expect("choice index");
        if choice["finish_reason"].is_null() {
            content_indices.push(index);
        } else {
            finish_indices.push(index);
        }
    }
    content_indices.sort_unstable();
    content_indices.dedup();
    finish_indices.sort_unstable();
    assert_eq!(content_indices, [0, 1], "{events:?}");
    assert_eq!(finish_indices, [0, 1], "{events:?}");
    assert_eq!(usage_chunks.len(), 1, "{events:?}");
    let usage = usage_chunks[0];
    assert_eq!(usage["choices"], json!([]), "{usage}");
    assert_eq!(usage["usage"]["prompt_tokens"], 5, "{usage}");
    assert_eq!(usage["usage"]["completion_tokens"], 4, "{usage}");
    assert_eq!(
        usage["x_deepseek_spark_stats"]["kv_cache"]["total_prompt_tokens"], 5,
        "{usage}"
    );
}

#[tokio::test]
async fn streaming_batch_echoes_each_prompt_in_its_own_choice() {
    let (status, events) = post_stream(
        app_with_engine(mock_engine()),
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": ["Hello", "Hello, world!"],
            "echo": true,
            "logprobs": 1,
            "max_tokens": 2,
            "stream": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // The first echo chunk per index must start with that prompt's own text.
    for (index, prompt_text) in [(0u64, "Hello"), (1, "Hello, world!")] {
        let first = events
            .iter()
            .find(|frame| {
                let choice = &frame["choices"][0];
                choice["index"] == index
                    && choice["text"]
                        .as_str()
                        .is_some_and(|text| text.starts_with(prompt_text))
            })
            .unwrap_or_else(|| panic!("no echo chunk for index {index}: {events:?}"));
        assert!(
            first["choices"][0]["text"]
                .as_str()
                .unwrap()
                .starts_with(prompt_text)
        );
    }
}
