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

//! The wire surface the numerics policy relies on (docs/NUMERICS.md, DECISIONS.md D67).
//!
//! A reference capture identifies every token by id (`return_tokens_as_token_ids`), scores a
//! frozen sequence by teacher forcing (`forced_tokens`), and reads which declared arithmetic
//! answered from the model card's `numerics` member. Each of those is a promise the capture tool
//! checks against, so each is pinned here: the shapes, and every refusal that keeps a forced
//! request from quietly turning into an ordinary generation.

use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, InferenceEngine,
    LogprobAlternative, MockEngine, TokenLogprob,
};
use sparkflash_api::server::router;
use sparkflash_api::state::AppState;
use sparkflash_model::numerics::{ConfigKey, NumericsIdentity, NumericsProfile, Registry};
use tower::ServiceExt as _;

const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
const FIXED_TIME: u64 = 1_700_000_000;

/// A mock that records every request it was handed and may declare forced decoding.
struct Recorder {
    requests: Arc<Mutex<Vec<EngineRequest>>>,
    inner: MockEngine,
}

impl Recorder {
    fn new(forced_decoding: bool) -> Self {
        let hello = TokenLogprob {
            token_id: 11,
            token: "Hello".to_owned(),
            logprob: -0.5,
            rank: Some(1),
            top: vec![
                LogprobAlternative {
                    token_id: 11,
                    token: "Hello".to_owned(),
                    logprob: -0.5,
                    rank: Some(1),
                },
                LogprobAlternative {
                    token_id: 7,
                    token: "Hi".to_owned(),
                    logprob: -1.5,
                    rank: Some(2),
                },
            ],
        };
        let world = TokenLogprob {
            token_id: 12,
            token: " world".to_owned(),
            logprob: -0.25,
            rank: Some(1),
            top: vec![LogprobAlternative {
                token_id: 12,
                token: " world".to_owned(),
                logprob: -0.25,
                rank: Some(1),
            }],
        };
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            inner: MockEngine::new(MODEL)
                .with_cluster_stats(ClusterStatsSnapshot::default())
                .with_capabilities(Capabilities {
                    forced_decoding,
                    ..Capabilities::ALL
                })
                .content(&["Hello", " world"])
                .with_vocab_size(128)
                .with_token_logprobs(vec![hello, world]),
        }
    }
}

impl InferenceEngine for Recorder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> futures::stream::BoxStream<'static, EngineEvent> {
        self.requests
            .lock()
            .expect("the recorder is not poisoned")
            .push(request.clone());
        self.inner.generate(request)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.inner.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    fn vocab_size(&self) -> Option<usize> {
        self.inner.vocab_size()
    }
}

fn app(engine: Recorder) -> (Router, Arc<Mutex<Vec<EngineRequest>>>) {
    let requests = Arc::clone(&engine.requests);
    let state = AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME);
    (router(state), requests)
}

async fn post(app: Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).expect("json body")))
        .expect("request");
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).expect("body is JSON"),
    )
}

/// The `data:` frames of a streamed response, without the closing `[DONE]`.
async fn post_stream(app: Router, uri: &str, body: &Value) -> Vec<Value> {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).expect("json body")))
        .expect("request");
    let response = app.oneshot(request).await.expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    String::from_utf8(bytes.to_vec())
        .expect("the stream is UTF-8")
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("a frame is JSON"))
        .collect()
}

async fn get(app: Router, uri: &str) -> Value {
    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("request");
    let response = app.oneshot(request).await.expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

#[tokio::test]
async fn completion_tokens_can_be_reported_as_token_ids() {
    let (app, _) = app(Recorder::new(false));
    let body = json!({
        "model": MODEL, "prompt": "hello", "logprobs": 1, "return_tokens_as_token_ids": true
    });
    let (status, body) = post(app, "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let logprobs = &body["choices"][0]["logprobs"];
    assert_eq!(logprobs["tokens"], json!(["token_id:11", "token_id:12"]));
    // vLLM advances the offsets by the reported strings, and so does this server.
    assert_eq!(logprobs["text_offset"], json!([0, 11]));
    assert_eq!(
        logprobs["top_logprobs"][0],
        json!({"token_id:11": -0.5, "token_id:7": -1.5})
    );
    assert_eq!(
        body["choices"][0]["text"], "Hello world",
        "the text itself is still decoded"
    );
}

#[tokio::test]
async fn streamed_token_id_offsets_are_the_unstreamed_ones() {
    let (app, _) = app(Recorder::new(false));
    let body = json!({
        "model": MODEL, "prompt": "hello", "logprobs": 1, "return_tokens_as_token_ids": true,
        "stream": true
    });
    let frames = post_stream(app, "/v1/completions", &body).await;
    let offsets: Vec<&Value> = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["logprobs"].get("text_offset"))
        .collect();
    // The second chunk starts after `token_id:11`, not after the decoded "Hello".
    assert_eq!(offsets, [&json!([0]), &json!([11])], "{frames:?}");
}

#[tokio::test]
async fn chat_tokens_can_be_reported_as_token_ids() {
    let (app, _) = app(Recorder::new(false));
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": true,
        "top_logprobs": 2,
        "return_tokens_as_token_ids": true
    });
    let (status, body) = post(app, "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first = &body["choices"][0]["logprobs"]["content"][0];
    assert_eq!(first["token"], "token_id:11");
    assert_eq!(first["bytes"], json!(b"token_id:11".to_vec()));
    assert_eq!(first["top_logprobs"][1]["token"], "token_id:7");
}

#[tokio::test]
async fn decoded_tokens_stay_the_default() {
    let (app, _) = app(Recorder::new(false));
    let body = json!({"model": MODEL, "prompt": "hello", "logprobs": 0});
    let (status, body) = post(app, "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["choices"][0]["logprobs"]["tokens"],
        json!(["Hello", " world"])
    );
}

#[tokio::test]
async fn a_forced_request_reaches_the_engine_as_greedy_teacher_forcing() {
    let (app, requests) = app(Recorder::new(true));
    let body = json!({
        "model": MODEL, "prompt": [1, 2, 3], "forced_tokens": [11, 12], "logprobs": 1
    });
    let (status, body) = post(app, "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let requests = requests.lock().expect("not poisoned");
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.forced_tokens, vec![11, 12]);
    assert_eq!(
        request.max_tokens, 2,
        "max_tokens defaults to the forced length"
    );
    assert_eq!(request.temperature, 0.0, "forced decoding is greedy");
    assert_eq!(request.prompt_tokens, vec![1, 2, 3]);
}

#[tokio::test]
async fn an_ordinary_request_carries_no_forced_tokens() {
    let (app, requests) = app(Recorder::new(true));
    let body = json!({"model": MODEL, "prompt": [1, 2, 3], "max_tokens": 2});
    let (status, _) = post(app, "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        requests.lock().expect("not poisoned")[0]
            .forced_tokens
            .is_empty()
    );
}

#[tokio::test]
async fn an_engine_without_forced_decoding_refuses_rather_than_samples() {
    let (app, requests) = app(Recorder::new(false));
    let body = json!({
        "model": MODEL, "prompt": [1, 2, 3], "forced_tokens": [11, 12], "logprobs": 1
    });
    let (status, body) = post(app, "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("forced_decoding")),
        "{body}"
    );
    assert!(requests.lock().expect("not poisoned").is_empty());
}

#[tokio::test]
async fn forced_requests_refuse_whatever_would_change_the_sequence_or_its_scores() {
    let base = json!({
        "model": MODEL, "prompt": [1, 2, 3], "forced_tokens": [11, 12], "logprobs": 1
    });
    let cases: [(&str, Value); 12] = [
        ("needs `logprobs`", json!({"logprobs": null})),
        ("at least one token", json!({"forced_tokens": []})),
        ("`max_tokens` is 3", json!({"max_tokens": 3})),
        ("`temperature`", json!({"temperature": 0.7})),
        ("`top_p`", json!({"top_p": 0.9})),
        ("`top_k`", json!({"top_k": 5})),
        ("`repetition_penalty`", json!({"repetition_penalty": 1.1})),
        ("`presence_penalty`", json!({"presence_penalty": 0.5})),
        ("`frequency_penalty`", json!({"frequency_penalty": 0.5})),
        ("stop string", json!({"stop": ["x"]})),
        (
            "outside the vocabulary",
            json!({"forced_tokens": [11, 500]}),
        ),
        ("n=2", json!({"n": 2})),
    ];
    for (needle, overrides) in cases {
        let mut body = base.clone();
        for (key, value) in overrides.as_object().expect("object") {
            if value.is_null() {
                body.as_object_mut().expect("object").remove(key);
            } else {
                body[key] = value.clone();
            }
        }
        let (app, requests) = app(Recorder::new(true));
        let (status, response) = post(app, "/v1/completions", &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{needle}: {response}");
        let message = response["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains(needle), "{needle}: {message}");
        assert!(
            requests.lock().expect("not poisoned").is_empty(),
            "{needle}: a refused request never reaches the engine"
        );
    }
    // Neutral values are accepted: they change nothing about a forced sequence.
    let mut neutral = base.clone();
    for (key, value) in [
        ("temperature", json!(0.0)),
        ("top_p", json!(1.0)),
        ("top_k", json!(0)),
        ("repetition_penalty", json!(1.0)),
        ("max_tokens", json!(2)),
    ] {
        neutral[key] = value;
    }
    let (app, _) = app(Recorder::new(true));
    let (status, response) = post(app, "/v1/completions", &neutral).await;
    assert_eq!(status, StatusCode::OK, "{response}");
}

#[tokio::test]
async fn the_engine_card_advertises_forced_decoding_only_where_declared() {
    let (with, _) = app(Recorder::new(true));
    let card = get(with, "/v1/models").await;
    assert_eq!(card["data"][0]["capabilities"]["forced_decoding"], true);
    let (without, _) = app(Recorder::new(false));
    let card = get(without, "/v1/models").await;
    assert_eq!(
        card["data"][0]["capabilities"]["forced_decoding"],
        Value::Null
    );
}

#[tokio::test]
async fn the_model_card_reports_the_numerics_identity() {
    let registry = Registry::embedded().expect("the embedded registry parses");
    let identity = NumericsIdentity::resolve(
        &registry,
        ConfigKey::tp(MODEL, 4, NumericsProfile::Reference, "sm_121a"),
        |name| (name == "SPARKFLASH_DECODE_MMA").then(|| "0".to_owned()),
    );
    let state = AppState::new(Arc::new(Recorder::new(false)))
        .with_fixed_time(FIXED_TIME)
        .with_numerics(identity);
    let card = get(router(state), "/v1/models").await;
    let numerics = &card["data"][0]["numerics"];
    assert_eq!(numerics["status"], "overridden");
    assert_eq!(numerics["version"], Value::Null);
    assert_eq!(numerics["world"], 4);
    assert_eq!(numerics["arch"], "sm_121a");
    assert_eq!(numerics["overrides"], json!(["SPARKFLASH_DECODE_MMA=0"]));
    assert_eq!(numerics["policy"], "docs/NUMERICS.md");

    // A state no serve path resolved has no member at all, rather than a claim it cannot make.
    let (bare, _) = app(Recorder::new(false));
    let card = get(bare, "/v1/models").await;
    assert_eq!(card["data"][0]["numerics"], Value::Null);
}
