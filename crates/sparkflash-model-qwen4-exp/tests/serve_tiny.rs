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

//! The Qwen4-Exp device arm served through the real OpenAI router.
//!
//! This is the integration the pack has to pass before a serve flag is worth adding: the generic
//! host driver (queue, worker thread, stop buffer, streaming) over this backend, the real
//! sparkflash-api router, and an ordinary completion request. The checkpoint is the tiny fixture,
//! so the test needs no weights - only a GPU whose driver can load the cubin.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::engine::InferenceEngine;
use sparkflash_api::state::AppState;
use sparkflash_model::ModelBackend;
use sparkflash_model_qwen4_exp::ByteCodec;
use sparkflash_model_qwen4_exp::device::{Qwen4ExpDeviceBackend, cubin_available};
use sparkflash_model_qwen4_exp::{MODEL_ID, SyntheticCheckpoint};
use sparkflash_serve::{HostDriver, HostOptions};
use tower::ServiceExt as _;

/// The context window the served fixture allocates.
const CONTEXT: usize = 32;

/// The pack's fixture codec: every id maps to a printable character, so a decoded completion is
/// never empty by construction, and no id it produces is an end-of-generation id.
/// The router the tests drive, or None when this build/machine cannot run the device.
///
/// Built once and cloned: several tests sharing one process must not each open their own CUDA
/// context, and one driver serving the queue in order is what production looks like anyway.
fn app() -> Option<Router> {
    static APP: std::sync::OnceLock<Option<Router>> = std::sync::OnceLock::new();
    APP.get_or_init(|| build(1)).clone()
}

/// A second driver over the same tiny fixture with two concurrent sessions.
fn concurrent_app() -> Option<Router> {
    static APP: std::sync::OnceLock<Option<Router>> = std::sync::OnceLock::new();
    APP.get_or_init(|| build(2)).clone()
}

fn build(concurrency: usize) -> Option<Router> {
    if !cubin_available() {
        eprintln!("skipping the serve integration: this build has no Qwen4-Exp cubin");
        return None;
    }
    let temp = tempfile::tempdir().expect("a temporary directory");
    let fixture = SyntheticCheckpoint::write(temp.path()).expect("the fixture writes");
    // The directory has to outlive the backend: keep the temp path alive by leaking the guard,
    // which is what a test process does with a few kilobytes anyway.
    let directory = fixture.directory.clone();
    std::mem::forget(temp);
    let backend = match Qwen4ExpDeviceBackend::open(&directory, CONTEXT) {
        Ok(backend) => backend,
        Err(error)
            if error.code() == "qwen4_device"
                && (error.message().contains("CUDA:")
                    || error.message().contains("cuModuleLoadData failed: 209")) =>
        {
            eprintln!("skipping the serve integration: {error}");
            return None;
        }
        Err(error) => panic!("the device arm refused the fixture checkpoint: {error}"),
    };
    let descriptor = backend.descriptor().clone();
    let codec = Arc::new(ByteCodec::new(descriptor.vocab_size));
    let driver = HostDriver::start(
        Box::new(backend),
        codec.clone(),
        HostOptions {
            max_tokens: 8,
            concurrency,
            ..HostOptions::default()
        },
    )
    .expect("the host driver starts");
    let engine: Arc<dyn InferenceEngine> = Arc::new(driver);
    let state = AppState::new(engine)
        .with_model_descriptor(Arc::new(descriptor))
        .with_tokenizer(codec);
    Some(sparkflash_api::router(state))
}

async fn post(app: Router, path: &str, body: Value) -> (StatusCode, String) {
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
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn get(app: Router, path: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(path)
                .body(Body::empty())
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
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn the_qwen_descriptor_is_what_the_models_route_reports() {
    let Some(app) = app() else { return };
    let (status, body) = get(app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains(MODEL_ID), "{body}");
    assert!(
        body.contains("\"context_tokens\":32") || body.contains("\"max_model_len\":32"),
        "{body}"
    );
}

#[tokio::test]
async fn a_greedy_completion_is_served_through_the_router() {
    let Some(app) = app() else { return };
    let (status, body) = post(
        app,
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": "hello",
            "max_tokens": 3,
            "temperature": 0.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("the completion is JSON");
    assert_eq!(value["model"], MODEL_ID);
    let choice = &value["choices"][0];
    let text = choice["text"].as_str().expect("a text completion");
    assert!(!text.is_empty(), "the completion was empty: {body}");
    let generated = value["usage"]["completion_tokens"]
        .as_u64()
        .expect("a completion token count");
    assert!(generated >= 1, "{body}");
    assert!(
        matches!(choice["finish_reason"].as_str(), Some("length" | "stop")),
        "{body}"
    );
}

#[tokio::test]
async fn a_seeded_temperature_request_is_sampled_and_reproducible() {
    let request = json!({
        "model": MODEL_ID,
        "prompt": "hello",
        "max_tokens": 4,
        "temperature": 1.0,
        "seed": 1234,
    });
    let Some(first_router) = app() else { return };
    let (status, first) = post(first_router, "/v1/completions", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let Some(second_router) = app() else { return };
    let (status, second) = post(second_router, "/v1/completions", request).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let left: Value = serde_json::from_str(&first).expect("JSON");
    let right: Value = serde_json::from_str(&second).expect("JSON");
    assert_eq!(
        left["choices"][0]["text"], right["choices"][0]["text"],
        "the same seed over the same fixture must draw the same tokens"
    );
}

#[tokio::test]
async fn a_streaming_completion_emits_sse_frames() {
    let Some(app) = app() else { return };
    let (status, body) = post(
        app,
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": "hello",
            "max_tokens": 3,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains("data: "), "no SSE frames: {body}");
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "the stream did not close: {body}"
    );
}

#[tokio::test]
async fn a_repetition_penalty_request_is_served() {
    let Some(app) = app() else { return };
    let (status, body) = post(
        app,
        "/v1/completions",
        json!({
            "model": MODEL_ID,
            "prompt": "hello",
            "max_tokens": 3,
            "temperature": 0.0,
            "repetition_penalty": 1.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("the completion is JSON");
    assert!(
        value["usage"]["completion_tokens"].as_u64().unwrap_or(0) >= 1,
        "a penalised greedy request still generates: {body}"
    );
}
#[tokio::test]
async fn the_tokenize_and_detokenize_routes_use_the_served_codec() {
    let Some(app) = app() else { return };
    let (status, body) = post(
        app.clone(),
        "/tokenize",
        json!({"model": MODEL_ID, "prompt": "abc"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("tokenize answers JSON");
    let tokens = value["tokens"]
        .as_array()
        .expect("a token list")
        .iter()
        .filter_map(Value::as_u64)
        .collect::<Vec<u64>>();
    assert_eq!(tokens.len(), 3, "one token per byte: {body}");

    let (status, body) = post(
        app,
        "/detokenize",
        json!({"model": MODEL_ID, "tokens": tokens}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).expect("detokenize answers JSON");
    let text = value["prompt"].as_str().or(value["text"].as_str());
    assert!(text.is_some(), "detokenize must answer text: {body}");
    assert_eq!(text.unwrap().len(), 3, "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_requests_are_both_served() {
    let Some(app) = concurrent_app() else { return };
    let request = || {
        json!({
            "model": MODEL_ID,
            "prompt": "hello",
            "max_tokens": 2,
            "temperature": 0.0,
        })
    };
    let (first, second) = tokio::join!(
        post(app.clone(), "/v1/completions", request()),
        post(app, "/v1/completions", request()),
    );
    let mut texts = Vec::new();
    for (status, body) in [first, second] {
        assert_eq!(status, StatusCode::OK, "{body}");
        let value: Value = serde_json::from_str(&body).expect("the completion is JSON");
        assert!(
            value["usage"]["completion_tokens"].as_u64().unwrap_or(0) >= 1,
            "{body}"
        );
        texts.push(
            value["choices"][0]["text"]
                .as_str()
                .expect("a text completion")
                .to_owned(),
        );
    }
    assert_eq!(
        texts[0], texts[1],
        "the two sessions share no KV, scratch or stream state"
    );
}
