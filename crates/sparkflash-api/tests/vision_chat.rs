//! The chat route's vision path (D59) against an engine that declares the vision tower: image parts
//! are prepared, each placeholder the template rendered is expanded to its image's span, and the
//! usage block counts the span positions. The prompt ids come from the vendored checkpoint
//! tokenizer, so the expansion is checked against the real placeholder id.

use std::path::Path;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::MODEL_ID;
use sparkflash_api::encoding::{TokenizerSeam, reference_tokenizer_path};
use sparkflash_api::engine::{Capabilities, ClusterStatsSnapshot, InferenceEngine, MockEngine};
use sparkflash_api::server::router;
use sparkflash_api::state::AppState;
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

fn engine(vision: bool) -> Arc<dyn InferenceEngine> {
    Arc::new(
        MockEngine::new(MODEL_ID)
            .with_cluster_stats(ClusterStatsSnapshot::default())
            .with_capabilities(Capabilities {
                vision,
                ..Capabilities::ALL
            })
            .content(&["A", " dot"]),
    )
}

fn app(vision: bool) -> Router {
    router(AppState::new(engine(vision)).with_tokenizer(seam()))
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
    let jpeg = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sparkflash-vision/tests/fixtures/decode/q75_420.jpg"),
    )
    .expect("the vision crate's JPEG fixture");
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
    let (status, response) = send(app(true), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["choices"][0]["message"]["content"], "A dot");
    let usage = &response["usage"];
    let image_tokens = 184 + jpeg_tokens;
    assert_eq!(usage["prompt_tokens_details"]["image_tokens"], image_tokens);
    assert_eq!(usage["prompt_tokens"], ids.len() - 2 + image_tokens);
}

#[tokio::test]
async fn the_url_length_limit_bounds_fetched_urls_not_inline_images() {
    // The recipe's 8192-byte URL limit is for the URLs it fetches. An inline image is bounded by
    // its bytes: a JPEG carrying a 12 KB comment segment (the same pixels) is read.
    let jpeg = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sparkflash-vision/tests/fixtures/decode/q75_420.jpg"),
    )
    .expect("the vision crate's JPEG fixture");
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
async fn a_placeholder_without_its_image_is_refused() {
    // A literal placeholder typed into the text is one more placeholder token than there are
    // images: the spans cannot be placed, so the request is refused rather than misaligned.
    let body = chat(json!([
        {"type": "text", "text": "<｜deepseek_image｜>"},
        {"type": "image_url", "image_url": {"url": data_url("image/png", DOT)}},
    ]));
    let (status, response) = send(app(true), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["error"]["param"], "messages");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("placeholder"), "{message}");
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

#[tokio::test]
async fn routes_without_the_vision_path_say_so() {
    let body = json!({
        "model": MODEL_ID,
        "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": data_url("image/png", DOT)},
        ]}],
    });
    let (status, response) = send(app(true), "POST", "/v1/responses", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("/v1/chat/completions"), "{message}");
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
