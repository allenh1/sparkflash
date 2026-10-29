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

//! The chat route's vision path (D59) against an engine that declares the vision tower: image parts
//! are prepared, each placeholder the template rendered is expanded to its image's span, and the
//! usage block counts the span positions. The prompt ids come from the vendored checkpoint
//! tokenizer, so the expansion is checked against the real placeholder id.

use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::MODEL_ID;
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, InferenceEngine, MockEngine,
};
use sparkflash_api::server::router;
use sparkflash_api::state::AppState;
use sparkflash_model_deepseek_v41::encoding::{TokenizerSeam, reference_tokenizer_path};
use sparkflash_vision::{IMAGE_TOKEN_ID, PrepOptions};
use tower::ServiceExt as _;

/// A 1x1 PNG, base64: planned up to the minimum pixel count, a 13x13 grid of 184 positions.
const DOT: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

fn seam() -> Arc<TokenizerSeam> {
    Arc::new(
        TokenizerSeam::from_path(&reference_tokenizer_path())
            .expect("the vendored reference tokenizer loads"),
    )
}

fn mock(vision: bool) -> MockEngine {
    MockEngine::new(MODEL_ID)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .with_capabilities(Capabilities {
            vision,
            ..Capabilities::ALL
        })
        .content(&["A", " dot"])
}

fn app(vision: bool) -> Router {
    router(AppState::new(Arc::new(mock(vision))).with_tokenizer(seam()))
}

/// A scripted engine that keeps the normalized request it served, so a test can assert where an
/// image's span actually sits in the prompt the model receives, not only how long the prompt is.
struct Recorder {
    served: Arc<Mutex<Vec<EngineRequest>>>,
    inner: MockEngine,
}

impl InferenceEngine for Recorder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> futures::stream::BoxStream<'static, EngineEvent> {
        self.served
            .lock()
            .expect("the request log is not poisoned")
            .push(request.clone());
        self.inner.generate(request)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.inner.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
}

fn recording(script: MockEngine) -> (Router, Arc<Mutex<Vec<EngineRequest>>>) {
    let served = Arc::new(Mutex::new(Vec::new()));
    let app = router(
        AppState::new(Arc::new(Recorder {
            served: Arc::clone(&served),
            inner: script,
        }))
        .with_tokenizer(seam()),
    );
    (app, served)
}

/// The single request the engine served.
fn served_one(served: &Mutex<Vec<EngineRequest>>) -> EngineRequest {
    let requests = served.lock().expect("the request log is not poisoned");
    assert_eq!(requests.len(), 1, "exactly one engine request");
    requests[0].clone()
}

/// Whether the recorder was entered at all.
fn served_none(served: &Mutex<Vec<EngineRequest>>) -> bool {
    served
        .lock()
        .expect("the request log is not poisoned")
        .is_empty()
}

/// The request's spans as `(start, len)` pairs, each checked to be a run of the image token id and
/// to account for every image token in the prompt.
fn spans(request: &EngineRequest) -> Vec<(usize, usize)> {
    let spans: Vec<(usize, usize)> = request
        .vision
        .iter()
        .map(|image| (image.span.start, image.span.len()))
        .collect();
    let mut end = 0;
    for &(start, len) in &spans {
        assert!(start >= end, "spans are in prompt order and do not overlap");
        assert_eq!(
            &request.prompt_tokens[start..start + len],
            vec![IMAGE_TOKEN_ID; len].as_slice(),
            "the span at {start} is {len} image tokens"
        );
        end = start + len;
    }
    assert_eq!(
        request
            .prompt_tokens
            .iter()
            .filter(|&&id| id == IMAGE_TOKEN_ID)
            .count(),
        spans.iter().map(|(_, len)| len).sum::<usize>(),
        "every image token in the prompt belongs to a span"
    );
    spans
}

/// A solid RGB PNG of the given size, so a test can exercise the size-dependent recipe without a
/// large fixture on disk.
fn solid_png(width: u32, height: u32) -> Vec<u8> {
    use flate2::write::ZlibEncoder;
    use flate2::{Compression, Crc};
    use std::io::Write as _;

    fn chunk(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(body.len() + 12);
        out.extend(
            u32::try_from(body.len())
                .expect("chunk length")
                .to_be_bytes(),
        );
        out.extend(kind);
        out.extend(body);
        let mut crc = Crc::new();
        crc.update(kind);
        crc.update(body);
        out.extend(crc.sum().to_be_bytes());
        out
    }

    let mut raw = Vec::with_capacity((width as usize * 3 + 1) * height as usize);
    for _ in 0..height {
        raw.push(0); // filter: none
        raw.extend(std::iter::repeat_n(0x80_u8, width as usize * 3));
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&raw).expect("the scanlines compress");
    let data = encoder.finish().expect("the zlib stream finishes");

    let mut header = Vec::new();
    header.extend(width.to_be_bytes());
    header.extend(height.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit truecolour RGB, deflate, adaptive filters, no interlace
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend(chunk(b"IHDR", &header));
    png.extend(chunk(b"IDAT", &data));
    png.extend(chunk(b"IEND", &[]));
    png
}

/// The vision crate's JPEG decode fixture.
fn jpeg_fixture() -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sparkflash-vision/tests/fixtures/decode/q75_420.jpg"),
    )
    .expect("the vision crate's JPEG fixture")
}

async fn send(app: Router, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
        .expect("request");
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn data_url(mime: &str, base64: &str) -> String {
    format!("data:{mime};base64,{base64}")
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0_u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[(n >> (18 - 6 * i) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn chat(content: Value) -> Value {
    json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
    })
}

/// Prompt ids before expansion: the template's one placeholder per image part.
fn template_ids(body: &Value) -> Vec<u32> {
    seam().encode_chat(body).expect("the chat body renders")
}

#[tokio::test]
async fn image_parts_become_spans_and_are_counted_in_usage() {
    let jpeg = jpeg_fixture();
    // Sent at `detail: "low"`: the recipe's 512-pixel long side.
    let low = PrepOptions {
        max_long_side: Some(512),
        ..PrepOptions::default()
    };
    let jpeg_tokens = sparkflash_vision::prepare(&jpeg, &low)
        .expect("the fixture prepares")
        .grid
        .tokens();
    let body = chat(json!([
        {"type": "text", "text": "What are these?"},
        {"type": "image_url", "image_url": {"url": data_url("image/png", DOT)}},
        {"type": "image_url", "image_url": {"url": data_url("image/jpeg", &base64(&jpeg)), "detail": "low"}},
    ]));
    let ids = template_ids(&body);
    assert_eq!(
        ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count(),
        2,
        "the template renders one placeholder token per image part"
    );
    let (app, served) = recording(mock(true));
    let (status, response) = send(app, "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["choices"][0]["message"]["content"], "A dot");
    let usage = &response["usage"];
    let image_tokens = 184 + jpeg_tokens;
    assert_eq!(usage["prompt_tokens_details"]["image_tokens"], image_tokens);
    assert_eq!(usage["prompt_tokens"], ids.len() - 2 + image_tokens);
    // The parts' order is the spans' order: the dot first (184 tokens), the rescaled JPEG second.
    let request = served_one(&served);
    let starts: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, id)| **id == IMAGE_TOKEN_ID)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        spans(&request),
        vec![(starts[0], 184), (starts[1] - 1 + 184, jpeg_tokens)]
    );
}

#[tokio::test]
async fn the_url_length_limit_bounds_fetched_urls_not_inline_images() {
    // The recipe's 8192-byte URL limit is for the URLs it fetches. An inline image is bounded by
    // its bytes: a JPEG carrying a 12 KB comment segment (the same pixels) is read.
    let jpeg = jpeg_fixture();
    let comment = vec![b'x'; 12_000];
    let mut padded = jpeg[..2].to_vec();
    padded.extend([0xFF, 0xFE]);
    padded.extend(u16::try_from(comment.len() + 2).unwrap().to_be_bytes());
    padded.extend(&comment);
    padded.extend(&jpeg[2..]);
    let url = data_url("image/jpeg", &base64(&padded));
    assert!(url.len() > sparkflash_api::vision::MAX_URL_LEN);
    let body = chat(json!([
        {"type": "image_url", "image_url": {"url": url}},
    ]));
    let (status, response) = send(app(true), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");

    let long = format!("https://example.invalid/{}.png", "a".repeat(8192));
    let body = chat(json!([
        {"type": "image_url", "image_url": {"url": long}},
    ]));
    let (status, response) = send(app(true), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("8192-byte limit"), "{message}");
}

#[tokio::test]
async fn a_placeholder_typed_into_the_text_is_text() {
    // The placeholder's spelling in a text part is not a second image: the one image part gets
    // the one span, and the spelling reaches the model as text.
    let body = chat(json!([
        {"type": "text", "text": "<｜deepseek_image｜>"},
        {"type": "image_url", "image_url": {"url": data_url("image/png", DOT)}},
    ]));
    let ids = template_ids(&body);
    assert_eq!(ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count(), 1);
    let (status, response) = send(app(true), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(
        response["usage"]["prompt_tokens_details"]["image_tokens"],
        184
    );
    assert_eq!(response["usage"]["prompt_tokens"], ids.len() - 1 + 184);
}

#[tokio::test]
async fn an_image_that_does_not_decode_is_refused() {
    let body = chat(json!([
        {"type": "image_url", "image_url": {"url": data_url("image/png", "aGVsbG8gd29ybGQ=")}},
    ]));
    let (status, response) = send(app(true), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["error"]["param"], "messages");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("image 1:"), "{message}");
}

#[tokio::test]
async fn an_image_shape_the_route_does_not_read_is_refused() {
    // The template renders an `image_pil` part as a placeholder, but there is no payload to read:
    // the carrier left after the readable parts are taken refuses the request (D27).
    let body = chat(json!([
        {"type": "text", "text": "hi"},
        {"type": "image_pil", "image": "opaque"},
    ]));
    let (status, response) = send(app(true), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("does not read"), "{message}");
}

#[tokio::test]
async fn a_text_engine_still_refuses_images() {
    let body = chat(json!([
        {"type": "image_url", "image_url": {"url": data_url("image/png", DOT)}},
    ]));
    let (status, response) = send(app(false), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("D18"), "{message}");
}

/// The DOT canvas's span, as the chat route already pins it.
const DOT_SPAN: usize = 184;

/// Prompt length of a chat body whose only content is `part`, before image expansion.
fn one_part_chat_len(part: Value) -> usize {
    let body = json!({"messages": [{"role": "user", "content": [part]}]});
    seam()
        .encode_chat(&body)
        .expect("the chat body renders")
        .len()
}

async fn send_text(app: Router, path: &str, body: Value) -> (StatusCode, String) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn responses_image_body(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "input": [{"type": "input_image", "image_url": data_url("image/png", DOT)}],
    });
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            body.insert(key.clone(), value.clone());
        }
    }
    body
}

fn anthropic_image_body(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": DOT,
            }},
        ]}],
    });
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            body.insert(key.clone(), value.clone());
        }
    }
    body
}

#[tokio::test]
async fn the_responses_bridge_expands_an_input_image_to_its_span() {
    // The bridge renders `input_image` as a chat image part, so the chat vision path expands its
    // placeholder to the canvas's span and the usage block counts the span (D59).
    let expected =
        one_part_chat_len(json!({"type": "image_url", "image_url": data_url("image/png", DOT)}))
            - 1
            + DOT_SPAN;
    let (status, response) = send(
        app(true),
        "POST",
        "/v1/responses",
        Some(responses_image_body(json!({}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["output_text"], "A dot");
    assert_eq!(response["usage"]["input_tokens"], expected);
}

#[tokio::test]
async fn the_anthropic_bridge_expands_an_image_block_to_its_span() {
    let expected = one_part_chat_len(
        json!({"type": "image_url", "image_url": {"url": data_url("image/png", DOT)}}),
    ) - 1
        + DOT_SPAN;
    let (status, response) = send(
        app(true),
        "POST",
        "/v1/messages",
        Some(anthropic_image_body(json!({}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["content"][0]["text"], "A dot");
    assert_eq!(response["usage"]["input_tokens"], expected);
}

#[tokio::test]
async fn both_bridges_stream_an_image_answer() {
    // The streamed paths build the same engine request, so the image span is counted there too.
    let responses_expected =
        one_part_chat_len(json!({"type": "image_url", "image_url": data_url("image/png", DOT)}))
            - 1
            + DOT_SPAN;
    let (status, events) = send_text(
        app(true),
        "/v1/responses",
        responses_image_body(json!({"stream": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{events}");
    assert!(events.contains("response.completed"), "{events}");
    assert!(
        events.contains(&format!("\"input_tokens\":{responses_expected}")),
        "{events}"
    );

    let anthropic_expected = one_part_chat_len(
        json!({"type": "image_url", "image_url": {"url": data_url("image/png", DOT)}}),
    ) - 1
        + DOT_SPAN;
    let (status, events) = send_text(
        app(true),
        "/v1/messages",
        anthropic_image_body(json!({"stream": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{events}");
    assert!(events.contains("message_start"), "{events}");
    assert!(
        events.contains(&format!("\"input_tokens\":{anthropic_expected}")),
        "{events}"
    );
}

#[tokio::test]
async fn a_text_engine_refuses_images_on_both_bridges() {
    for (uri, body) in [
        ("/v1/responses", responses_image_body(json!({}))),
        ("/v1/messages", anthropic_image_body(json!({}))),
    ] {
        let (status, response) = send(app(false), "POST", uri, Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {response}");
        let message = response["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("D18") && message.contains("263"),
            "{uri}: {message}"
        );
    }
}

#[tokio::test]
async fn a_carrier_the_bridge_does_not_read_is_refused_by_name() {
    // `find_image_carrier` scans the whole body, but the bridge renders only `input`/`messages`.
    // A carrier outside them would otherwise be answered from its text tokens alone, so it is
    // refused with the shape and its path.
    for (uri, body) in [
        (
            "/v1/responses",
            responses_image_body(json!({"metadata": {"image_url": data_url("image/png", DOT)}})),
        ),
        (
            "/v1/messages",
            anthropic_image_body(json!({"metadata": {"image_url": data_url("image/png", DOT)}})),
        ),
    ] {
        let (status, response) = send(app(true), "POST", uri, Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {response}");
        let message = response["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("does not read an image in this shape"),
            "{uri}: {message}"
        );
    }
}

#[tokio::test]
async fn an_anthropic_system_role_inside_messages_joins_the_system_prompt() {
    // The Messages API has no "system" role in `messages`, but Claude Code sends one; the text
    // joins the system prompt instead of failing with "invalid Anthropic message role".
    let body = json!({
        "model": MODEL_ID,
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ],
    });
    let (status, response) = send(app(true), "POST", "/v1/messages", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["content"][0]["text"], "A dot");
    // The system turn renders exactly where a top-level system prompt would: one system message,
    // before the conversation, so the prompt length matches the equivalent chat request.
    let expected = seam()
        .encode_chat(&json!({
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hi"},
            ],
            "parallel_tool_calls": true,
        }))
        .expect("renders")
        .len();
    assert_eq!(response["usage"]["input_tokens"], expected);
}

#[tokio::test]
async fn system_turns_join_in_order_after_the_top_level_system() {
    // Top-level `system` first, then each `system` turn in conversation order.
    let body = json!({
        "model": MODEL_ID,
        "max_tokens": 8,
        "system": "top",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "first"}]},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "second"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
    });
    let (status, response) = send(app(true), "POST", "/v1/messages", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let expected = seam()
        .encode_chat(&json!({
            "messages": [
                {"role": "system", "content": "top\n\nfirst\n\nsecond"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
            "parallel_tool_calls": true,
        }))
        .expect("renders")
        .len();
    assert_eq!(response["usage"]["input_tokens"], expected);
}

fn tool_parameters() -> Value {
    json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
}

#[tokio::test]
async fn many_images_on_both_bridges_take_one_span_each() {
    // Two images in one turn: each placeholder gets its own image's span, in part order, so the
    // second span starts (span_len - 1) tokens later than its placeholder did.
    let jpeg = jpeg_fixture();
    let photo =
        sparkflash_vision::prepare(&jpeg, &PrepOptions::default()).expect("the fixture prepares");
    let photo_tokens = photo.grid.tokens();
    let photo_pixels = photo.grid.best_height as usize * photo.grid.best_width as usize * 3;
    let dot_url = data_url("image/png", DOT);
    let photo_url = data_url("image/jpeg", &base64(&jpeg));
    let cases = [
        (
            "/v1/responses",
            json!({
                "model": MODEL_ID,
                "input": [
                    {"type": "input_image", "image_url": dot_url},
                    {"type": "input_text", "text": "between"},
                    {"type": "input_image", "image_url": photo_url},
                ],
            }),
            // Each input_image renders as a chat image part carrying the URL it was given.
            json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": dot_url},
                {"type": "text", "text": "between"},
                {"type": "image_url", "image_url": photo_url},
            ]}]}),
        ),
        (
            "/v1/messages",
            json!({
                "model": MODEL_ID,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": DOT,
                    }},
                    {"type": "text", "text": "between"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": base64(&jpeg),
                    }},
                ]}],
            }),
            // Each image block renders as a chat image part with a data URL.
            json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": dot_url}},
                {"type": "text", "text": "between"},
                {"type": "image_url", "image_url": {"url": photo_url}},
            ]}]}),
        ),
    ];
    for (uri, body, chat_body) in cases {
        let (app, served) = recording(mock(true));
        let (status, response) = send(app, "POST", uri, Some(body)).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {response}");
        let placeholders = template_ids(&chat_body);
        let starts: Vec<usize> = placeholders
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == IMAGE_TOKEN_ID)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(
            starts.len(),
            2,
            "{uri}: the template renders one placeholder per image part"
        );
        let expected = placeholders.len() - 2 + DOT_SPAN + photo_tokens;
        assert_eq!(response["usage"]["input_tokens"], expected, "{uri}");
        let request = served_one(&served);
        assert_eq!(
            request.prompt_tokens.len(),
            expected,
            "{uri}: the counted prompt is the one the engine reads"
        );
        assert_eq!(
            spans(&request),
            vec![
                (starts[0], DOT_SPAN),
                (starts[1] - 1 + DOT_SPAN, photo_tokens)
            ],
            "{uri}"
        );
        assert_eq!(
            request.vision[0].pixels.len(),
            546 * 546 * 3,
            "{uri}: the dot's canvas"
        );
        assert_eq!(
            request.vision[1].pixels.len(),
            photo_pixels,
            "{uri}: the photo's canvas"
        );
        assert!(
            request.images.is_empty(),
            "{uri}: every image part was prepared"
        );
    }
}

#[tokio::test]
async fn the_responses_bridge_keeps_detail_low() {
    // The Responses schema puts `detail` beside `image_url`; it has to reach the chat image part,
    // or a "low" image is prepared at full size while the chat route prepares it at 512 pixels.
    let pixels = solid_png(1024, 768);
    let low = PrepOptions {
        max_long_side: Some(512),
        ..PrepOptions::default()
    };
    let low_tokens = sparkflash_vision::prepare(&pixels, &low)
        .expect("the image prepares")
        .grid
        .tokens();
    let full_tokens = sparkflash_vision::prepare(&pixels, &PrepOptions::default())
        .expect("the image prepares")
        .grid
        .tokens();
    assert!(
        low_tokens < full_tokens,
        "the image is larger than the 512-pixel side"
    );
    let url = data_url("image/png", &base64(&pixels));
    let body = json!({
        "model": MODEL_ID,
        "input": [{"type": "input_image", "image_url": url, "detail": "low"}],
    });
    let expected = one_part_chat_len(json!({
        "type": "image_url",
        "image_url": {"url": url, "detail": "low"},
    })) - 1
        + low_tokens;
    let (app, served) = recording(mock(true));
    let (status, response) = send(app, "POST", "/v1/responses", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["usage"]["input_tokens"], expected);
    let request = served_one(&served);
    assert_eq!(
        spans(&request),
        vec![(
            request
                .prompt_tokens
                .iter()
                .position(|&id| id == IMAGE_TOKEN_ID)
                .expect("one span"),
            low_tokens
        )]
    );
}

#[tokio::test]
async fn an_image_and_a_tool_call_share_one_request_on_both_bridges() {
    // The tool list renders into the prompt ahead of the conversation, so the image span sits after
    // it while the user's text follows the span; the model's call still comes back as a
    // function_call / tool_use, and the image is counted in the same request.
    let arguments = ["{\"city\":", " \"SF\"}"];
    let dot = data_url("image/png", DOT);
    let cases = [
        (
            "/v1/responses",
            json!({
                "model": MODEL_ID,
                "input": [
                    {"type": "input_image", "image_url": dot.clone()},
                    {"type": "input_text", "text": "weather here?"},
                ],
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Report the weather.",
                    "parameters": tool_parameters(),
                }],
            }),
            json!({
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": dot.clone()},
                    {"type": "text", "text": "weather here?"},
                ]}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Report the weather.",
                        "parameters": tool_parameters(),
                    },
                }],
            }),
        ),
        (
            "/v1/messages",
            json!({
                "model": MODEL_ID,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": DOT,
                    }},
                    {"type": "text", "text": "weather here?"},
                ]}],
                "tools": [{
                    "name": "get_weather",
                    "description": "Report the weather.",
                    "input_schema": tool_parameters(),
                }],
            }),
            json!({
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": dot.clone()}},
                    {"type": "text", "text": "weather here?"},
                ]}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Report the weather.",
                        "parameters": tool_parameters(),
                    },
                }],
            }),
        ),
    ];
    for (uri, body, chat_body) in cases {
        let (app, served) = recording(mock(true).tool_call("engine", "get_weather", &arguments));
        let (status, response) = send(app, "POST", uri, Some(body)).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {response}");
        let placeholders = template_ids(&chat_body);
        let start = placeholders
            .iter()
            .position(|&id| id == IMAGE_TOKEN_ID)
            .expect("the template renders the image placeholder");
        let expected = placeholders.len() - 1 + DOT_SPAN;
        assert_eq!(response["usage"]["input_tokens"], expected, "{uri}");
        let request = served_one(&served);
        assert_eq!(
            request.prompt_tokens.len(),
            expected,
            "{uri}: the tool list and the span are both in the counted prompt"
        );
        assert_eq!(spans(&request), vec![(start, DOT_SPAN)], "{uri}");
        let before = seam()
            .decode(&request.prompt_tokens[..start])
            .expect("the prefix decodes");
        assert!(
            before.contains("get_weather"),
            "{uri}: the tool list renders before the image span: {before}"
        );
        let after = seam()
            .decode(&request.prompt_tokens[start + DOT_SPAN..])
            .expect("the suffix decodes");
        assert!(
            after.contains("weather here?"),
            "{uri}: the image's text follows its span: {after}"
        );
        let call = match uri {
            "/v1/responses" => response["output"].as_array().and_then(|items| {
                items
                    .iter()
                    .find(|item| item["type"] == "function_call")
                    .cloned()
            }),
            _ => response["content"].as_array().and_then(|blocks| {
                blocks
                    .iter()
                    .find(|block| block["type"] == "tool_use")
                    .cloned()
            }),
        }
        .unwrap_or_else(|| panic!("{uri}: a tool call in {response}"));
        assert_eq!(call["name"], "get_weather", "{uri}");
        if uri == "/v1/responses" {
            assert_eq!(call["arguments"], "{\"city\": \"SF\"}", "{uri}");
        } else {
            assert_eq!(call["input"], json!({"city": "SF"}), "{uri}");
        }
    }
}

#[tokio::test]
async fn a_streamed_image_answer_can_still_call_a_tool() {
    // The streamed paths build the same engine request, so a screenshot-and-tool turn streams its
    // call while the image span is in the prompt and counted in the terminal usage frame.
    let arguments = ["{\"city\":", " \"SF\"}"];
    let cases = [
        (
            "/v1/responses",
            responses_image_body(json!({
                "stream": true,
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "parameters": tool_parameters(),
                }],
            })),
            "function_call",
        ),
        (
            "/v1/messages",
            anthropic_image_body(json!({
                "stream": true,
                "tools": [{"name": "get_weather", "input_schema": tool_parameters()}],
            })),
            "tool_use",
        ),
    ];
    for (uri, body, item) in cases {
        let (app, served) = recording(mock(true).tool_call("engine", "get_weather", &arguments));
        let (status, events) = send_text(app, uri, body).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {events}");
        assert!(events.contains(item), "{uri}: the call in {events}");
        assert!(events.contains("get_weather"), "{uri}: {events}");
        let request = served_one(&served);
        assert_eq!(
            spans(&request),
            vec![(
                request
                    .prompt_tokens
                    .iter()
                    .position(|&id| id == IMAGE_TOKEN_ID)
                    .expect("the placeholder expands to a span"),
                DOT_SPAN
            )],
            "{uri}"
        );
        assert!(
            events.contains(&format!("\"input_tokens\":{}", request.prompt_tokens.len())),
            "{uri}: the streamed usage counts the span: {events}"
        );
    }
}

#[tokio::test]
async fn the_completions_route_takes_a_prompt_so_it_refuses_an_image_marker() {
    // A prompt, not messages: there is no image part to pair a marker with, so the marker is
    // refused by name with or without the vision tower instead of reaching the model untouched.
    let placeholder = "look <｜deepseek_image｜> here";
    for (prompt, marker) in [
        ("look <image>AAAA</image>", "<image>...</image> tag"),
        (placeholder, "image placeholder"),
    ] {
        for vision in [false, true] {
            let (app, served) = recording(mock(vision));
            let body = json!({"model": MODEL_ID, "prompt": prompt, "max_tokens": 4});
            let (status, response) = send(app, "POST", "/v1/completions", Some(body)).await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{prompt} (vision: {vision}): {response}"
            );
            let message = response["error"]["message"].as_str().expect("a message");
            assert!(message.contains("/v1/completions"), "{message}");
            assert!(message.contains(marker), "{message}");
            assert!(message.contains("prompt 1"), "{message}");
            assert!(!message.contains("  "), "no folded line breaks: {message}");
            assert!(
                served_none(&served),
                "{prompt} (vision: {vision}): refused before generation"
            );
        }
    }

    // A pre-tokenized prompt carries the image token with no image part to fill it. This body is
    // device-token eligible (a pre-tokenized prompt and no sampling fields), so the refusal has to
    // run before that fast path: a claimed envelope replays the window on the ranks, and the host
    // never reads the prompt at all.
    let ids = seam().encode_text(placeholder).expect("the prompt encodes");
    assert!(ids.contains(&IMAGE_TOKEN_ID));
    let (app, served) = recording(mock(true));
    let body = json!({"model": MODEL_ID, "prompt": ids});
    let (status, response) = send(app, "POST", "/v1/completions", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("image token id"),
        "{response}"
    );
    assert!(served_none(&served), "refused before generation");

    // A carrier in the body is refused by name too, in the same device-eligible shape (a
    // pre-tokenized prompt with no image token of its own, so only the carrier is the reason).
    let text_ids = seam().encode_text("hi").expect("the prompt encodes");
    for prompt in [json!("hi"), json!(text_ids)] {
        let (app, served) = recording(mock(true));
        let body = json!({
            "model": MODEL_ID,
            "prompt": prompt,
            "image_url": data_url("image/png", DOT),
        });
        let (status, response) = send(app, "POST", "/v1/completions", Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert!(
            response["error"]["message"]
                .as_str()
                .expect("a message")
                .contains("image_url"),
            "{response}"
        );
        assert!(served_none(&served), "refused before generation");
    }

    // Plain text, including other angle brackets and a data: URL in the prompt, is still served.
    let (app, _) = recording(mock(true));
    let body = json!({
        "model": MODEL_ID,
        "prompt": "plain <text> and a data:image/png;base64,aGVsbG8= URL",
        "max_tokens": 4,
    });
    let (status, response) = send(app, "POST", "/v1/completions", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");
}

#[tokio::test]
async fn models_advertise_images_exactly_when_the_engine_declares_vision() {
    for vision in [false, true] {
        let (status, body) = send(app(vision), "GET", "/v1/models", None).await;
        assert_eq!(status, StatusCode::OK);
        let caps = &body["data"][0]["capabilities"];
        assert_eq!(caps["multimodal"], vision);
        assert_eq!(caps["image_url"], vision);
    }
}
