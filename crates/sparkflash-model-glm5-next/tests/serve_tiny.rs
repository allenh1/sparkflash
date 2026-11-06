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
//! sparkflash-api router, and an ordinary chat request. The checkpoint is the small fixture, so the
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
use sparkflash_ops::codec::ByteCodec;
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
    build_with(1, |_| Ok(()))
}

/// A driver over the tiny fixture, after `configure` has touched the backend.
fn build_with(
    concurrency: usize,
    configure: impl FnOnce(&mut Glm5NextDeviceBackend) -> Result<(), String>,
) -> Option<Router> {
    if !cubin_available() {
        eprintln!("skipping the serve integration: this build has no Qwen4-Exp cubin");
        return None;
    }
    let temp = tempfile::tempdir().expect("a temporary directory");
    let config = Glm5NextConfig::small();
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
    let mut backend =
        match Glm5NextDeviceBackend::open_single_rank(cuda, config, &directory, CONTEXT) {
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
    configure(&mut backend).expect("the backend accepts the configuration");
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

/// A drafting session decodes the tokens a plain one does.
///
/// This is the exactness claim speculation rests on: a verify row stands in for the one-token
/// step it replaces, so a drafted request must emit the plain request's tokens in the same order,
/// whatever it kept. The two arms run side by side on the same fixture, and the drafting one must
/// also prove it actually drafted - a build where drafting was silently off would pass the first
/// assertion alone.
#[test]
fn a_drafted_decode_is_the_bits_of_a_plain_one() {
    let Some((mut plain, mut drafting)) = two_backends() else {
        return;
    };
    let prompt = [3u32, 7, 11, 13, 17];
    let mut agreed = 0usize;
    let mut a = plain.prefill(&prompt).expect("the plain prefill");
    let mut b = drafting.prefill(&prompt).expect("the drafting prefill");
    for step in 0..8 {
        assert_eq!(
            a.next_token, b.next_token,
            "step {step} disagreed: {:?} against {:?}",
            a.next_token, b.next_token
        );
        let Some(token) = a.next_token else {
            break;
        };
        a = plain.decode(token).expect("the plain decode");
        b = drafting.decode(token).expect("the drafting decode");
        agreed += 1;
    }
    assert!(agreed > 0, "the fixture decoded nothing");
    let speculation = drafting.speculation();
    assert!(
        speculation.drafts_proposed > 0,
        "the drafting session proposed nothing: {speculation:?}"
    );
    // The fixture's MTP layer has synthetic weights, so a draft it proposes is a lottery and a
    // round usually keeps nothing: the accept path itself is [sparkflash_model::verdict]'s and is
    // tested there. What this asserts is that rounds ran and the tokens did not move.
    assert!(
        speculation.verify_steps > 0,
        "the session never drafted: {speculation:?}"
    );
    assert_eq!(
        speculation.drafts_proposed,
        speculation.verify_steps * 2,
        "every round proposed the two drafts it verified: {speculation:?}"
    );
}

/// Two requests through one fused engine are both served, off one pass over the weights.
///
/// The host driver gives each request its own worker and its own session, so without fusion the
/// two sessions each step the whole rank for one row. With it, both sessions are handles on one
/// engine thread, which steps whatever is waiting as a single batched forward.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_requests_are_fused_into_one_pass() {
    let Some(app) = build_with(2, |backend| {
        backend
            .enable_fusion(2)
            .map_err(|error| error.to_string())?;
        assert!(backend.fused(), "the backend reports its fused batch");
        // A fused batch and a drafter are two trades on the same step: this arm runs one.
        let refused = backend
            .enable_speculation(2, None)
            .expect_err("fusion excludes speculation");
        assert_eq!(refused.code(), "glm5_device");
        Ok(())
    }) else {
        return;
    };
    let ask = |prompt: &'static str| {
        let app = app.clone();
        async move {
            post(
                app,
                "/v1/completions",
                json!({"model": MODEL_ID, "prompt": prompt, "max_tokens": 4}),
            )
            .await
        }
    };
    let (first, second) = tokio::join!(ask("hello"), ask("world"));
    for (status, body) in [first, second] {
        assert_eq!(status, StatusCode::OK, "{body}");
        let response: Value = serde_json::from_str(&body).expect("a JSON response");
        assert!(
            response["choices"][0]["text"]
                .as_str()
                .is_some_and(|text| !text.is_empty()),
            "a fused member generated nothing: {response}"
        );
    }
}

/// A sequence reports the tokens it holds, and a prompt that extends them is forwarded in part.
///
/// This is the arm half of [D66]: the driver asks the session what it can continue from and hands
/// it the whole prompt either way, so a session that reports nothing gets a cold prefill and a
/// session that reports too much continues from a prefix the caller never saw.
#[test]
fn a_sequence_reports_the_prefix_it_holds() {
    let Some(mut session) = one_session(false) else {
        return;
    };
    let prompt: Vec<u32> = "hello".bytes().map(u32::from).collect();
    let step = session.prefill(&prompt).expect("the prefill");
    let mut history = prompt.clone();
    let mut next = step.next_token;
    for _ in 0..2 {
        let token = next.expect("the fixture generates");
        history.push(token);
        next = session.decode(token).expect("the decode").next_token;
    }
    let held = session
        .reusable_prefix()
        .expect("the sequence holds every token it forwarded")
        .to_vec();
    assert_eq!(held, history);
    // An extension forwards only the addition and reports the rest as a cache hit.
    let mut extended = held.clone();
    extended.extend(" again".bytes().map(u32::from));
    session.prefill(&extended).expect("the extended prefill");
    assert_eq!(session.cached_tokens(), held.len());
}

/// One session over the tiny fixture, drafting or not, or None without a device.
fn one_session(speculate: bool) -> Option<Box<dyn sparkflash_model::ModelSession>> {
    if !cubin_available() {
        eprintln!("skipping the session test: this build has no cubin");
        return None;
    }
    let temp = tempfile::tempdir().expect("a temporary directory");
    let config = Glm5NextConfig::small();
    write_synthetic_checkpoint(temp.path(), &config, 7, 2).expect("the fixture writes");
    let directory = temp.path().to_path_buf();
    std::mem::forget(temp);
    let cuda = sparkflash_kernels::cuda_driver::Cuda::load().ok()?;
    let mut backend = Glm5NextDeviceBackend::open_single_rank(
        Arc::new(CudaHandle::new(cuda)),
        config,
        &directory,
        CONTEXT,
    )
    .ok()?;
    if speculate {
        backend.enable_speculation(2, None).ok()?;
    }
    backend.open_session().ok()
}

/// A plain session and a drafting one over the same tiny fixture, or None without a device.
fn two_backends() -> Option<(
    Box<dyn sparkflash_model::ModelSession>,
    Box<dyn sparkflash_model::ModelSession>,
)> {
    if !cubin_available() {
        eprintln!("skipping the drafting comparison: this build has no cubin");
        return None;
    }
    let temp = tempfile::tempdir().expect("a temporary directory");
    let config = Glm5NextConfig::small();
    write_synthetic_checkpoint(temp.path(), &config, 7, 2).expect("the fixture writes");
    let directory = temp.path().to_path_buf();
    std::mem::forget(temp);
    let open = |speculate: bool| {
        let cuda = sparkflash_kernels::cuda_driver::Cuda::load().ok()?;
        let mut backend = Glm5NextDeviceBackend::open_single_rank(
            Arc::new(CudaHandle::new(cuda)),
            config.clone(),
            &directory,
            CONTEXT,
        )
        .ok()?;
        if speculate {
            backend.enable_speculation(2, None).ok()?;
        }
        backend.open_session().ok()
    };
    Some((open(false)?, open(true)?))
}

/// Two requests through the real driver: the second is served because the first ended cleanly.
///
/// The driver's reuse is append-only ([D66]): a prompt must *extend* the tokens the held sequence
/// forwarded, so an identical prompt gets a fresh sequence and a full prefill. The extension case
/// is [a_sequence_reports_the_prefix_it_holds]'s, at the arm, because a text round trip through
/// this test's byte codec does not reproduce the ids a completion emitted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_repeated_prompt_is_served_from_a_fresh_sequence() {
    // A driver of its own: the shared one is one worker, and a test running beside this one
    // would hand that worker its own request between these two and drop the held sequence.
    let Some(app) = build() else {
        return;
    };
    for attempt in 0..2 {
        let (status, body) = post(
            app.clone(),
            "/v1/completions",
            json!({"model": MODEL_ID, "prompt": "hello", "max_tokens": 4}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let response: Value = serde_json::from_str(&body).expect("a JSON response");
        assert_eq!(
            response["usage"]["prompt_cache_hit_tokens"], 0,
            "attempt {attempt} is a fresh sequence: {response}"
        );
        assert_eq!(
            response["usage"]["prompt_tokens"], 5,
            "the byte codec's five tokens: {response}"
        );
        assert!(
            response["choices"][0]["text"]
                .as_str()
                .is_some_and(|text| !text.is_empty()),
            "the fixture generated nothing: {response}"
        );
    }
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
