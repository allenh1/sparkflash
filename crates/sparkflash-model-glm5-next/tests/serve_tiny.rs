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

//! The GLM-5.3-Flash device arm served through the real OpenAI router.
//!
//! This is the integration the pack has to pass before its serve flag is worth adding: the generic
//! host driver (queue, worker thread, stop buffer, streaming) over this backend, the real
//! sparkflash-api router, and an ordinary chat request. The checkpoint is the tiny fixture, so the
//! test needs no weights - only a GPU whose driver can load the cubin.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::engine::InferenceEngine;
use sparkflash_api::state::AppState;
use sparkflash_model::ModelBackend;
use sparkflash_model_glm5_next::checkpoint::write_synthetic_checkpoint;
use sparkflash_model_glm5_next::device::CudaHandle;
use sparkflash_model_glm5_next::{
    Glm5NextConfig, Glm5NextDeviceBackend, MODEL_ID, cubin_available,
};
use sparkflash_model_qwen4_exp::ByteCodec;
use sparkflash_serve::{HostDriver, HostOptions};
use tower::ServiceExt as _;

/// The context window the served fixture allocates.
const CONTEXT: usize = 32;

/// The router the tests drive, or None when this build/machine cannot run the device.
///
/// Built once and cloned: several tests sharing one process must not each open their own CUDA
/// context, and one driver serving the queue in order is what production looks like anyway.
fn app() -> Option<Router> {
    static APP: std::sync::OnceLock<Option<Router>> = std::sync::OnceLock::new();
    APP.get_or_init(build).clone()
}

fn build() -> Option<Router> {
    if !cubin_available() {
        eprintln!("skipping the serve integration: this build has no Qwen4-Exp cubin");
        return None;
    }
    let temp = tempfile::tempdir().expect("a temporary directory");
    let config = Glm5NextConfig::tiny();
    write_synthetic_checkpoint(temp.path(), &config, 7, 2).expect("the fixture writes");
    // The directory has to outlive the backend: keep the temp path alive by leaking the guard,
    // which is what a test process does with a few kilobytes anyway.
    let directory = temp.path().to_path_buf();
    std::mem::forget(temp);
    let cuda = match sparkflash_kernels::cuda_driver::Cuda::load() {
        Ok(cuda) => Arc::new(CudaHandle::new(cuda)),
        Err(error) => {
            eprintln!("skipping the serve integration: no CUDA driver ({error})");
            return None;
        }
    };
    let backend = match Glm5NextDeviceBackend::open_single_rank(cuda, config, &directory, CONTEXT) {
        Ok(backend) => backend,
        Err(error)
            if error.code() == "glm5_device"
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
            concurrency: 1,
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chat_request_runs_through_the_device_backend() {
    let Some(app) = app() else {
        return;
    };
    let (status, body) = post(
        app,
        "/v1/chat/completions",
        json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).expect("a JSON response");
    assert_eq!(response["model"], MODEL_ID);
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .expect("a completion");
    assert!(
        !content.is_empty(),
        "the byte codec decodes every id to text"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_models_endpoint_reports_the_glm_descriptor() {
    let Some(app) = app() else {
        return;
    };
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let body: Value = serde_json::from_slice(&bytes).expect("a JSON response");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("a model list")
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect();
    assert!(ids.contains(&MODEL_ID), "{ids:?}");
}
