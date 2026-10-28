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

//! Router tests for the Responses and Anthropic adapters: request conversion, buffered
//! assembly, streaming events, tool calls and the typed refusals.

use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineRequest, InferenceEngine, MockEngine,
};
use sparkflash_api::server::router;
use sparkflash_api::state::AppState;
use sparkflash_model::{PromptCodec, PromptError};
use tower::ServiceExt as _;

const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
const FIXED_TIME: u64 = 1_700_000_000;

fn app_with(engine: MockEngine) -> Router {
    router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME))
}

async fn send(app: Router, uri: &str, body: &Value) -> (StatusCode, Vec<u8>) {
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
        .to_bytes()
        .to_vec();
    (status, bytes)
}

fn body_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("body is JSON")
}

fn sse_events(bytes: &[u8]) -> Vec<(String, Value)> {
    let text = String::from_utf8_lossy(bytes);
    let mut event = String::new();
    let mut events = Vec::new();
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("event: ") {
            event = name.to_owned();
        } else if let Some(data) = line.strip_prefix("data: ")
            && data != "[DONE]"
        {
            events.push((
                std::mem::take(&mut event),
                serde_json::from_str(data).expect("SSE frame is JSON"),
            ));
        }
    }
    events
}

#[derive(Clone, Default)]
struct RecordingCodec {
    requests: Arc<Mutex<Vec<Value>>>,
}

impl PromptCodec for RecordingCodec {
    fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, PromptError> {
        self.requests
            .lock()
            .expect("codec log")
            .push(request.clone());
        Ok(vec![11, 12, 13])
    }

    fn encode_text(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        Ok(vec![11])
    }

    fn decode(&self, _ids: &[u32]) -> Option<String> {
        Some("decoded".to_owned())
    }

    fn token_str(&self, _id: u32) -> Option<String> {
        Some("token".to_owned())
    }
}

fn codec_app(engine: MockEngine) -> (Router, Arc<Mutex<Vec<Value>>>) {
    let codec = RecordingCodec::default();
    let requests = Arc::clone(&codec.requests);
    let app = router(
        AppState::new(Arc::new(engine))
            .with_fixed_time(FIXED_TIME)
            .with_tokenizer(Arc::new(codec)),
    );
    (app, requests)
}

fn function_tool() -> Value {
    json!({
        "type": "function",
        "name": "get_weather",
        "description": "Read the weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    })
}

#[tokio::test]
async fn responses_converts_instructions_history_tools_and_reasoning() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .content(&["ok"]);
    let (app, captured) = codec_app(engine);
    let body = json!({
        "model": MODEL,
        "instructions": "be brief",
        "input": [
            {"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": "context"}
            ]},
            {"role": "user", "content": "weather?"},
            {"type": "function_call", "call_id": "call-1", "name": "get_weather",
             "arguments": "{\"city\":\"Beijing\"}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "sunny"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I should look"}]},
        ],
        "tools": [function_tool()],
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "reasoning": {"effort": "low"},
        "text": {"format": {"type": "text"}, "verbosity": "low"},
        "store": false,
        "truncation": "auto",
    });
    let (status, bytes) = send(app, "/v1/responses", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let requests = captured.lock().expect("codec log");
    let chat = requests.first().expect("the prompt was rendered");
    let messages = chat["messages"].as_array().expect("messages");
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "be brief");
    // A developer message is folded onto the system role: the released encoder drops it in
    // thinking mode and refuses it in chat mode, so keeping the wire role would lose the content.
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(messages[1]["content"][0]["text"], "context");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"], "weather?");
    assert_eq!(messages[3]["role"], "assistant");
    assert_eq!(
        messages[3]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(messages[3]["tool_calls"][0]["id"], "call-1");
    assert_eq!(messages[4]["role"], "tool");
    assert_eq!(messages[4]["tool_call_id"], "call-1");
    assert_eq!(messages[4]["content"], "sunny");
    assert_eq!(messages[5]["role"], "assistant");
    assert_eq!(messages[5]["reasoning_content"], "I should look");
    assert_eq!(chat["tools"][0]["function"]["parameters"]["type"], "object");
    assert_eq!(chat["parallel_tool_calls"], false);
    assert_eq!(chat["reasoning_effort"], "low");
}

#[tokio::test]
async fn responses_buffered_assembles_reasoning_message_and_calls() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .reasoning(&["thinking"])
        .content(&["I will check."])
        .tool_call(
            "engine-call",
            "get_weather",
            &["{\"city\":", "\"Beijing\"}"],
        );
    let body = json!({
        "model": MODEL,
        "input": "weather?",
        "tools": [function_tool()],
        "parallel_tool_calls": true,
    });
    let (status, bytes) = send(app_with(engine), "/v1/responses", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let json = body_json(&bytes);
    let output = json["output"].as_array().expect("output");
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["summary"][0]["text"], "thinking");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "I will check.");
    assert_eq!(output[2]["type"], "function_call");
    assert_eq!(output[2]["name"], "get_weather");
    assert_eq!(output[2]["arguments"], "{\"city\":\"Beijing\"}");
    assert_eq!(output[2]["status"], "completed");
    assert!(
        output[2]["id"]
            .as_str()
            .expect("item id")
            .starts_with("fc_dsv41_")
    );
    assert!(
        output[2]["call_id"]
            .as_str()
            .expect("call id")
            .starts_with("call_dsv41_")
    );
    assert_eq!(json["output_text"], "I will check.");
    assert_eq!(json["usage"]["output_tokens"], 4);
}

#[tokio::test]
async fn responses_reports_truncation_as_incomplete() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .content(&["one", "two"]);
    let body = json!({
        "model": MODEL,
        "input": "hello",
        "max_output_tokens": 1,
    });
    let (status, bytes) = send(app_with(engine.clone()), "/v1/responses", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let json = body_json(&bytes);
    assert_eq!(json["status"], "incomplete");
    assert_eq!(json["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(json["output"][0]["content"][0]["text"], "one");

    let mut stream_body = body.clone();
    stream_body["stream"] = json!(true);
    let (status, bytes) = send(app_with(engine), "/v1/responses", &stream_body).await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_events(&bytes);
    assert!(
        events.iter().all(|(name, _)| name != "response.completed"),
        "a truncated stream must not report completion: {events:?}"
    );
    let incomplete = events
        .iter()
        .find(|(name, _)| name == "response.incomplete")
        .map(|(_, data)| data)
        .expect("response.incomplete");
    assert_eq!(incomplete["response"]["status"], "incomplete");
    assert_eq!(
        incomplete["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
}

#[tokio::test]
async fn responses_tool_only_answer_has_no_empty_message_and_honours_parallel_false() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .tool_call("one", "get_weather", &["{}"])
        .tool_call("two", "get_weather", &["{}"]);
    let body = json!({
        "model": MODEL,
        "input": "weather?",
        "tools": [function_tool()],
        "parallel_tool_calls": false,
    });
    let (status, bytes) = send(app_with(engine), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::OK);
    let json = body_json(&bytes);
    let output = json["output"].as_array().expect("output");
    assert_eq!(output.len(), 1, "only the first call is reported: {json}");
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[0]["name"], "get_weather");
}

#[tokio::test]
async fn responses_streams_reasoning_text_and_tool_calls() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .reasoning(&["plan"])
        .content(&["checking"])
        .tool_call(
            "engine-call",
            "get_weather",
            &["{\"city\":", "\"Beijing\"}"],
        );
    let body = json!({
        "model": MODEL,
        "input": "weather?",
        "tools": [function_tool()],
        "parallel_tool_calls": true,
        "stream": true,
    });
    let (status, bytes) = send(app_with(engine), "/v1/responses", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let events = sse_events(&bytes);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert!(
        names.contains(&"response.reasoning_summary_text.delta"),
        "{names:?}"
    );
    assert!(names.contains(&"response.output_text.delta"), "{names:?}");
    assert!(
        names.contains(&"response.function_call_arguments.delta"),
        "{names:?}"
    );
    assert!(names.contains(&"response.output_item.done"), "{names:?}");
    let completed = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, data)| data)
        .expect("response.completed");
    let output = completed["response"]["output"].as_array().expect("output");
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["summary"][0]["text"], "plan");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "checking");
    assert_eq!(output[2]["type"], "function_call");
    assert_eq!(output[2]["arguments"], "{\"city\":\"Beijing\"}");
    assert_eq!(completed["response"]["usage"]["output_tokens"], 4);
}

#[tokio::test]
async fn responses_stream_honours_parallel_false() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .tool_call("one", "get_weather", &["{}"])
        .tool_call("two", "get_weather", &["{}"]);
    let body = json!({
        "model": MODEL,
        "input": "weather?",
        "tools": [function_tool()],
        "parallel_tool_calls": false,
        "stream": true,
    });
    let (status, bytes) = send(app_with(engine), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_events(&bytes);
    let completed = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, data)| data)
        .expect("response.completed");
    let output = completed["response"]["output"].as_array().expect("output");
    assert_eq!(
        output.len(),
        1,
        "only the first call is reported: {completed}"
    );
}

#[tokio::test]
async fn responses_refuses_items_and_formats_it_cannot_honour() {
    let cases = [
        (
            json!({"model": MODEL, "input": "hi", "previous_response_id": "resp_1"}),
            "previous_response_id",
        ),
        (
            json!({"model": MODEL, "input": "hi", "text": {"format": {"type": "json_schema"}}}),
            "text.format",
        ),
        (
            json!({"model": MODEL, "input": [{"type": "item_reference", "id": "x"}]}),
            "input",
        ),
        (
            json!({"model": MODEL, "input": [{"type": "computer_call_output", "call_id": "c"}]}),
            "input",
        ),
        (
            json!({"model": MODEL, "input": "hi", "tools": [{"type": "custom", "name": "shell"}]}),
            "tools",
        ),
        (
            json!({"model": MODEL, "input": "hi", "tools": [{"type": "function"}]}),
            "name",
        ),
        (
            json!({"model": MODEL, "input": "hi", "tools": "not an array"}),
            "tools",
        ),
        (
            json!({"model": MODEL, "input": "hi", "tool_choice": "required",
                   "tools": [function_tool()]}),
            "tool_choice",
        ),
        (
            json!({"model": MODEL, "input": [{"type": "function_call", "name": "x"}]}),
            "call_id",
        ),
        (
            json!({"model": MODEL, "input": [{"type": "function_call_output", "call_id": "c"}]}),
            "output",
        ),
        (
            json!({"model": MODEL, "input": [{"type": "message", "role": "wizard", "content": "x"}]}),
            "role",
        ),
        (
            json!({"model": MODEL, "input": [{"type": "message", "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {}}]}]}),
            "input",
        ),
        (
            json!({"model": MODEL, "input": [{"role": "user", "content": 7}]}),
            "input",
        ),
        (json!({"model": MODEL, "input": [7]}), "input"),
        (
            json!({"model": MODEL, "input": [], "instructions": null}),
            "input",
        ),
        (
            json!({"model": MODEL, "input": "hi", "text": "not an object"}),
            "text",
        ),
        (
            json!({"model": MODEL, "input": "hi", "text": {"format": "nope"}}),
            "text.format",
        ),
        (
            json!({"model": MODEL, "input": "hi", "text": {"format": {"type": "yaml"}}}),
            "text.format",
        ),
    ];
    for (body, param) in cases {
        let (status, bytes) = send(app_with(MockEngine::new(MODEL)), "/v1/responses", &body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{body}: {}",
            String::from_utf8_lossy(&bytes)
        );
        assert_eq!(
            body_json(&bytes)["error"]["param"],
            param,
            "{body}: {}",
            String::from_utf8_lossy(&bytes)
        );
    }
}

#[tokio::test]
async fn anthropic_converts_system_blocks_tool_history_and_thinking() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .content(&["ok"]);
    let (app, captured) = codec_app(engine);
    let body = json!({
        "model": MODEL,
        "max_tokens": 16,
        "system": [{"type": "text", "text": "be brief"}, {"type": "text", "text": "and kind"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "look it up"},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Beijing"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "sunny"}]},
            ]},
        ],
        "tools": [{"name": "get_weather", "description": "Read the weather.",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    });
    let (status, bytes) = send(app, "/v1/messages", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let requests = captured.lock().expect("codec log");
    let chat = requests.first().expect("the prompt was rendered");
    let messages = chat["messages"].as_array().expect("messages");
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "be brief\n\nand kind");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["reasoning_content"], "look it up");
    assert_eq!(
        messages[2]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Beijing\"}"
    );
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "toolu_1");
    assert_eq!(messages[3]["content"], "sunny");
    assert_eq!(chat["tools"][0]["function"]["parameters"]["type"], "object");
    assert_eq!(chat["enable_thinking"], true);
    assert_eq!(chat["parallel_tool_calls"], false);
}

#[tokio::test]
async fn anthropic_buffered_assembles_thinking_text_and_tool_use() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .reasoning(&["plan"])
        .content(&["checking"])
        .tool_call("engine", "get_weather", &["{\"city\":", "\"Beijing\"}"]);
    let body = json!({
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "top_k": 5,
    });
    let (status, bytes) = send(app_with(engine), "/v1/messages", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let json = body_json(&bytes);
    let content = json["content"].as_array().expect("content");
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "plan");
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "checking");
    assert_eq!(content[2]["type"], "tool_use");
    assert_eq!(content[2]["name"], "get_weather");
    assert_eq!(content[2]["input"]["city"], "Beijing");
    assert!(
        content[2]["id"]
            .as_str()
            .expect("tool id")
            .starts_with("toolu_dsv41_")
    );
    assert_eq!(json["stop_reason"], "tool_use");
    assert_eq!(json["usage"]["output_tokens"], 4);
}

#[tokio::test]
async fn anthropic_malformed_native_arguments_are_an_empty_object() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .tool_call("engine-call", "get_weather", &["not-json"]);
    let body = json!({
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    });
    let (status, bytes) = send(app_with(engine), "/v1/messages", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let json = body_json(&bytes);
    let tool = json["content"]
        .as_array()
        .expect("content")
        .iter()
        .find(|block| block["type"] == "tool_use")
        .expect("a tool_use block");
    assert_eq!(
        tool["input"],
        json!({}),
        "Anthropic tool_use.input must stay an object"
    );
}

#[tokio::test]
async fn anthropic_streaming_usage_partitions_the_prompt_cache() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .with_cache(64, 0, false)
        .content(&["ok"]);
    let body = json!({
        "model": MODEL,
        "max_tokens": 16,
        "stream": true,
        "messages": [{"role": "user", "content": "x".repeat(300)}],
    });
    let (status, bytes) = send(app_with(engine), "/v1/messages", &body).await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_events(&bytes);
    let usage = |name: &str| {
        events
            .iter()
            .find(|(event, _)| event == name)
            .map(|(_, data)| data)
            .unwrap_or_else(|| panic!("missing {name}: {events:?}"))
    };
    let start = &usage("message_start")["message"]["usage"];
    let delta = &usage("message_delta")["usage"];
    assert_eq!(start["cache_read_input_tokens"], 64);
    assert_eq!(
        start["input_tokens"], delta["input_tokens"],
        "message_start and message_delta must report the same uncached input"
    );
    assert_eq!(
        start["cache_read_input_tokens"],
        delta["cache_read_input_tokens"]
    );
}

#[tokio::test]
async fn anthropic_streams_thinking_text_and_tool_use() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .reasoning(&["plan"])
        .content(&["checking"])
        .tool_call("engine", "get_weather", &["{\"city\":", "\"Beijing\"}"]);
    let body = json!({
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "stream": true,
    });
    let (status, bytes) = send(app_with(engine), "/v1/messages", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let events = sse_events(&bytes);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert!(names.contains(&"message_start"), "{names:?}");
    assert!(names.contains(&"content_block_start"), "{names:?}");
    assert!(
        events
            .iter()
            .any(|(_, data)| { data["delta"]["type"] == "input_json_delta" }),
        "{events:?}"
    );
    assert!(names.contains(&"message_delta"), "{names:?}");
    assert!(names.contains(&"message_stop"), "{names:?}");
    let delta = events
        .iter()
        .find(|(name, _)| name == "message_delta")
        .map(|(_, data)| data)
        .expect("message_delta");
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert_eq!(delta["usage"]["output_tokens"], 4);
}

#[tokio::test]
async fn anthropic_stream_honours_parallel_false() {
    let engine = MockEngine::new(MODEL)
        .with_capabilities(Capabilities::ALL)
        .tool_call("one", "get_weather", &["{}"])
        .tool_call("two", "get_weather", &["{}"]);
    let body = json!({
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
        "stream": true,
    });
    let (status, bytes) = send(app_with(engine), "/v1/messages", &body).await;
    assert_eq!(status, StatusCode::OK);
    let starts = sse_events(&bytes)
        .into_iter()
        .filter(|(name, data)| {
            name == "content_block_start" && data["content_block"]["type"] == "tool_use"
        })
        .count();
    assert_eq!(starts, 1, "only the first call is streamed");
}

#[tokio::test]
async fn anthropic_refuses_malformed_shapes_typed() {
    let cases = [
        (
            json!({"model": MODEL, "max_tokens": 8, "messages": []}),
            "messages",
        ),
        (
            json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            "max_tokens",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8,
                   "messages": [{"role": "user", "content": []}]}),
            "messages",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": [{"type": "document"}]}
            ]}]}),
            "input",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8,
                   "messages": [{"role": "wizard", "content": "hi"}]}),
            "messages",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8,
                   "messages": [{"role": "user", "content": [{"type": "document"}]}]}),
            "input",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8, "system": [{"type": "image"}],
                   "messages": [{"role": "user", "content": "hi"}]}),
            "",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8, "tools": "nope",
                   "messages": [{"role": "user", "content": "hi"}]}),
            "tools",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8, "tool_choice": {"type": "sometimes"},
                   "messages": [{"role": "user", "content": "hi"}]}),
            "tool_choice",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8, "thinking": {"type": "maybe"},
                   "messages": [{"role": "user", "content": "hi"}]}),
            "thinking",
        ),
        (
            json!({"model": MODEL, "max_tokens": 8, "messages": "nope"}),
            "",
        ),
    ];
    for (body, param) in cases {
        let (status, bytes) = send(app_with(MockEngine::new(MODEL)), "/v1/messages", &body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{body}: {}",
            String::from_utf8_lossy(&bytes)
        );
        if !param.is_empty() {
            assert_eq!(
                body_json(&bytes)["error"]["param"],
                param,
                "{body}: {}",
                String::from_utf8_lossy(&bytes)
            );
        }
    }
}

#[tokio::test]
async fn anthropic_thinking_disabled_reaches_the_prompt_and_top_k_reaches_the_engine() {
    let requests: Arc<Mutex<Vec<EngineRequest>>> = Arc::new(Mutex::new(Vec::new()));
    struct Recorder {
        requests: Arc<Mutex<Vec<EngineRequest>>>,
        inner: MockEngine,
    }
    impl InferenceEngine for Recorder {
        fn model_id(&self) -> &str {
            self.inner.model_id()
        }
        fn generate(
            &self,
            request: EngineRequest,
        ) -> futures::stream::BoxStream<'static, sparkflash_api::engine::EngineEvent> {
            self.requests
                .lock()
                .expect("request log")
                .push(request.clone());
            self.inner.generate(request)
        }
        fn capabilities(&self) -> Capabilities {
            self.inner.capabilities()
        }
        fn cluster_stats(&self) -> ClusterStatsSnapshot {
            self.inner.cluster_stats()
        }
    }
    let recorder = Recorder {
        requests: Arc::clone(&requests),
        inner: MockEngine::new(MODEL)
            .with_capabilities(Capabilities::ALL)
            .content(&["ok"]),
    };
    let codec = RecordingCodec::default();
    let codec_requests = Arc::clone(&codec.requests);
    let app = router(
        AppState::new(Arc::new(recorder))
            .with_fixed_time(FIXED_TIME)
            .with_tokenizer(Arc::new(codec)),
    );
    let body = json!({
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
        "top_k": 7,
    });
    let (status, bytes) = send(app, "/v1/messages", &body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let captured = codec_requests.lock().expect("codec log");
    assert_eq!(captured[0]["enable_thinking"], false);
    let engine_request = requests.lock().expect("engine log")[0].clone();
    assert_eq!(engine_request.top_k, 7);
    assert!(!engine_request.thinking);
}

fn dsml_answer() -> String {
    let bar = '\u{ff5c}';
    format!(
        "I will check.\n\n<{bar}DSML{bar} calls>\n\
         <{bar}DSML{bar} invoke name=\"get_weather\">\n\
         <{bar}DSML{bar} parameter name=\"city\" string=\"true\">Beijing</{bar}DSML{bar} parameter>\n\
         </{bar}DSML{bar} invoke>\n\
         </{bar}DSML{bar} calls>"
    )
}

#[tokio::test]
async fn responses_parses_dsml_tool_calls_buffered_and_streamed() {
    let answer = dsml_answer();
    let body = json!({
        "model": MODEL,
        "input": "weather?",
        "tools": [function_tool()],
    });
    let (status, bytes) = send(
        app_with(
            MockEngine::new(MODEL)
                .with_capabilities(Capabilities::ALL)
                .content(&["I will check."])
                .with_response(sparkflash_api::engine::MockResponse {
                    content: vec![answer.clone()],
                    ..Default::default()
                }),
        ),
        "/v1/responses",
        &body,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let json = body_json(&bytes);
    let output = json["output"].as_array().expect("output");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "I will check.");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["name"], "get_weather");
    assert_eq!(output[1]["arguments"], "{\"city\": \"Beijing\"}");

    let mut stream_body = body.clone();
    stream_body["stream"] = json!(true);
    let (status, bytes) = send(
        app_with(
            MockEngine::new(MODEL)
                .with_capabilities(Capabilities::ALL)
                .with_response(sparkflash_api::engine::MockResponse {
                    content: vec![answer],
                    ..Default::default()
                }),
        ),
        "/v1/responses",
        &stream_body,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let events = sse_events(&bytes);
    let completed = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, data)| data)
        .expect("response.completed");
    let output = completed["response"]["output"].as_array().expect("output");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["arguments"], "{\"city\": \"Beijing\"}");
}

#[tokio::test]
async fn responses_stream_refuses_a_malformed_dsml_block() {
    let bar = '\u{ff5c}';
    let answer = format!("<{bar}DSML{bar} calls>\n<{bar}DSML{bar} invoke name=\"x\">");
    let body = json!({
        "model": MODEL,
        "input": "weather?",
        "tools": [function_tool()],
        "stream": true,
    });
    let (status, bytes) = send(
        app_with(
            MockEngine::new(MODEL)
                .with_capabilities(Capabilities::ALL)
                .with_response(sparkflash_api::engine::MockResponse {
                    content: vec![answer],
                    ..Default::default()
                }),
        ),
        "/v1/responses",
        &body,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_events(&bytes);
    assert!(
        events.iter().any(|(name, _)| name == "response.failed"),
        "{events:?}"
    );
}

#[tokio::test]
async fn anthropic_parses_dsml_tool_calls_buffered_and_streamed() {
    let answer = dsml_answer();
    let body = json!({
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    });
    let (status, bytes) = send(
        app_with(
            MockEngine::new(MODEL)
                .with_capabilities(Capabilities::ALL)
                .with_response(sparkflash_api::engine::MockResponse {
                    content: vec![answer.clone()],
                    ..Default::default()
                }),
        ),
        "/v1/messages",
        &body,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let json = body_json(&bytes);
    assert_eq!(json["content"][0]["type"], "text");
    assert_eq!(json["content"][0]["text"], "I will check.");
    assert_eq!(json["content"][1]["type"], "tool_use");
    assert_eq!(json["content"][1]["name"], "get_weather");
    assert_eq!(json["content"][1]["input"]["city"], "Beijing");
    assert_eq!(json["stop_reason"], "tool_use");

    let mut stream_body = body.clone();
    stream_body["stream"] = json!(true);
    let (status, bytes) = send(
        app_with(
            MockEngine::new(MODEL)
                .with_capabilities(Capabilities::ALL)
                .with_response(sparkflash_api::engine::MockResponse {
                    content: vec![answer],
                    ..Default::default()
                }),
        ),
        "/v1/messages",
        &stream_body,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let events = sse_events(&bytes);
    assert!(
        events.iter().any(|(name, data)| {
            name == "content_block_start" && data["content_block"]["type"] == "tool_use"
        }),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|(_, data)| { data["delta"]["type"] == "input_json_delta" }),
        "{events:?}"
    );
}
