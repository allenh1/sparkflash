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

//! The Qwen dialect through the real router: reasoning becomes `reasoning_content` and a
//! `<tool_call>` XML block becomes an OpenAI `tool_calls` array, buffered and streamed.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::engine::{Capabilities, FinishReason, MockEngine};
use sparkflash_api::qwen::QwenResponseEngine;
use sparkflash_api::server::router;
use sparkflash_api::state::AppState;
use tower::ServiceExt as _;

const MODEL: &str = "nvidia/Qwen3.8-Flash-Next-NVFP4";

fn app() -> Router {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .content(&[
            "I need to inspect the repository.",
            "\n</think>\n\n",
            "I'll look.\n\n",
            "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/lib.rs\n</parameter>\n",
            "</function>\n</tool_call>",
        ])
        .with_finish_reason(FinishReason::Stop);
    router(
        AppState::new(QwenResponseEngine::layer(Arc::new(engine))).with_fixed_time(1_700_000_000),
    )
}

async fn send(app: Router, body: &Value) -> (StatusCode, Vec<u8>) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).expect("json body")))
        .expect("request");
    let response = app.oneshot(request).await.expect("router answer");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes()
        .to_vec();
    (status, bytes)
}

fn body(content: &str) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }],
    })
}

#[tokio::test]
async fn buffered_chat_splits_reasoning_and_parses_the_xml_call() {
    let (status, bytes) = send(app(), &body("inspect the repo")).await;
    assert_eq!(status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&bytes).expect("JSON body");
    let message = &body["choices"][0]["message"];
    assert_eq!(
        message["reasoning_content"],
        "I need to inspect the repository.\n"
    );
    assert_eq!(message["content"], "I'll look.\n\n");
    assert_eq!(message["tool_calls"][0]["type"], "function");
    assert_eq!(
        message["tool_calls"][0]["function"]["name"],
        Value::String("read_file".to_owned())
    );
    let arguments: Value = serde_json::from_str(
        message["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .expect("arguments string"),
    )
    .expect("arguments JSON");
    assert_eq!(arguments, json!({"path": "src/lib.rs"}));
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn streamed_chat_carries_reasoning_content_and_tool_call_deltas() {
    let mut request = body("inspect the repo");
    request["stream"] = json!(true);
    let (status, bytes) = send(app(), &request).await;
    assert_eq!(status, StatusCode::OK);
    let frames: Vec<Value> = String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE frame"))
        .collect();
    let mut reasoning = String::new();
    let mut arguments = String::new();
    let mut name = None;
    let mut finish_reason = None;
    for frame in &frames {
        let Some(choice) = frame["choices"].get(0) else {
            continue;
        };
        if let Some(text) = choice["delta"]["reasoning_content"].as_str() {
            reasoning.push_str(text);
        }
        if let Some(call) = choice["delta"]["tool_calls"].get(0) {
            if let Some(value) = call["function"]["name"].as_str() {
                name = Some(value.to_owned());
            }
            if let Some(value) = call["function"]["arguments"].as_str() {
                arguments.push_str(value);
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            finish_reason = Some(reason.to_owned());
        }
    }
    assert_eq!(reasoning, "I need to inspect the repository.\n");
    assert_eq!(name.as_deref(), Some("read_file"));
    assert_eq!(arguments, r#"{"path":"src/lib.rs"}"#);
    assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
}
