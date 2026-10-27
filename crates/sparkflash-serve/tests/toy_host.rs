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

//! A toy model served through the real router on the generic host path.
//!
//! The toy has its own tiny vocabulary, an ASCII prompt codec and a dense backend. It is not
//! V4.1-shaped: it has no MLA, no MoE, no Engram, no DSpark and no device path. Passing through
//! the same OpenAI router is the Phase 3 acceptance check.

use std::collections::VecDeque;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::router;
use sparkflash_api::state::AppState;
use sparkflash_model::{
    BackendError, Capabilities, ModelBackend, ModelDescriptor, ModelSession, PromptCodec,
    PromptError, Sampling, StepOutput,
};
use sparkflash_serve::{HostDriver, HostOptions};
use tower::ServiceExt as _;

const MODEL: &str = "toy/tiny-dense";
const VOCAB: usize = 258;
const RESPONSE: &str = "hello";

struct ToyCodec;

impl PromptCodec for ToyCodec {
    fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, PromptError> {
        let messages = request
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                PromptError::invalid("messages must be an array", Some("messages"), None)
            })?;
        let mut text = String::new();
        for message in messages {
            let content = message
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    PromptError::invalid("content must be a string", Some("messages"), None)
                })?;
            text.push_str(content);
        }
        self.encode_text(&text, true)
    }

    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        let mut ids = Vec::with_capacity(text.len() + usize::from(add_special_tokens));
        if add_special_tokens {
            ids.push(1);
        }
        for byte in text.bytes() {
            if byte >= 32 {
                ids.push(u32::from(byte) + 2);
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Option<String> {
        let mut text = String::new();
        for &id in ids {
            if id == 1 {
                continue;
            }
            text.push(char::from(u8::try_from(id.checked_sub(2)?).ok()?));
        }
        Some(text)
    }

    fn token_str(&self, id: u32) -> Option<String> {
        self.decode(&[id])
    }
}

fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL.to_owned(),
        family: "toy-dense".to_owned(),
        weights_id: "toy-dense".to_owned(),
        context_tokens: 128,
        max_output_tokens: 16,
        vocab_size: VOCAB,
        capabilities: Capabilities::default(),
    }
}

struct ToyBackend {
    descriptor: ModelDescriptor,
}

impl ModelBackend for ToyBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(ToySession::new()))
    }
}

struct ToySession {
    remaining: VecDeque<u32>,
}

impl ToySession {
    fn new() -> Self {
        let remaining = RESPONSE.bytes().map(|byte| u32::from(byte) + 2).collect();
        Self { remaining }
    }

    fn step(&mut self) -> StepOutput {
        let next = self.remaining.pop_front();
        let mut logits = vec![f32::NEG_INFINITY; VOCAB];
        if let Some(token) = next {
            logits[token as usize] = 0.0;
        }
        StepOutput {
            logits,
            next_token: next,
            draft: Vec::new(),
        }
    }
}

impl ModelSession for ToySession {
    fn set_sampling(&mut self, _sampling: Sampling) {}

    fn prefill(&mut self, _tokens: &[u32]) -> Result<StepOutput, BackendError> {
        Ok(self.step())
    }

    fn decode(&mut self, _token: u32) -> Result<StepOutput, BackendError> {
        Ok(self.step())
    }
}

fn app() -> Router {
    let backend = Box::new(ToyBackend {
        descriptor: descriptor(),
    });
    let driver = HostDriver::start(
        backend,
        Arc::new(ToyCodec),
        HostOptions {
            max_tokens: 16,
            ..HostOptions::default()
        },
    )
    .expect("the host driver starts");
    let state = AppState::new(Arc::new(driver))
        .with_model_descriptor(Arc::new(descriptor()))
        .with_tokenizer(Arc::new(ToyCodec))
        .with_fixed_time(1_700_000_000);
    router(state)
}

async fn post(path: &str, body: Value) -> (StatusCode, Value) {
    let response = app()
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

#[tokio::test]
async fn the_toy_model_serves_a_chat_completion_through_the_router() {
    let (status, body) = post(
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "say hello"}],
            "max_tokens": 8
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], MODEL);
    assert_eq!(body["choices"][0]["message"]["content"], RESPONSE);
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["completion_tokens"], RESPONSE.len());
}
