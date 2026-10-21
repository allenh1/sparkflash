//! Tokenizer seam tests: pinned checkpoint ids, pinned chat-template ids, and the seam wired
//! through the real axum router.
//!
//! The expected ids come from docs/build/task30-evidence/tokenizer-parity.json, produced offline
//! with the released image's own tokenizer stack on the served snapshot (see that file for
//! provenance). The prompt strings themselves are covered by encoding_golden.rs against the
//! official encoder fixtures, so these tests pin the tokenization half of the seam.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::encoding::{
    CaseError, TokenizerSeam, TokenizerSeamError, reference_tokenizer_path,
};
use sparkflash_api::engine::{Capabilities, ClusterStatsSnapshot, MockEngine};
use sparkflash_api::server::router;
use sparkflash_api::state::AppState;
use tower::ServiceExt as _;

const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
const FIXED_TIME: u64 = 1_700_000_000;

/// The recorded oracle from the released image's tokenizer stack.
const EVIDENCE: &str = include_str!("../../../docs/build/task30-evidence/tokenizer-parity.json");

fn evidence() -> Value {
    serde_json::from_str(EVIDENCE).expect("tokenizer parity evidence is valid JSON")
}

fn seam() -> TokenizerSeam {
    TokenizerSeam::from_path(&reference_tokenizer_path())
        .expect("the vendored reference tokenizer loads")
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("an id array")
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("a non-negative id")).expect("a u32 id"))
        .collect()
}

fn named_chat_case(name: &str) -> Value {
    match name {
        "chat_user_hello_default" => json!({
            "messages": [{"role": "user", "content": "Hello"}],
        }),
        "chat_user_hello_nothink" => json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"thinking": false},
        }),
        "chat_system_user_low" => json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ],
            "reasoning_effort": "low",
        }),
        "chat_user_tools" => json!({
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }],
        }),
        "chat_user_blocks" => json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"}
                ]
            }],
        }),
        "chat_multi_turn" => json!({
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"}
            ],
        }),
        other => panic!("unknown evidence case {other}"),
    }
}

#[test]
fn completion_text_ids_match_the_released_tokenizer() {
    let seam = seam();
    let recorded_json = evidence();
    for entry in recorded_json["completion"]
        .as_array()
        .expect("completion entries")
    {
        let text = entry["text"].as_str().expect("text");
        let expected = ids(&entry["ids_no_special"]);
        assert_eq!(
            seam.encode_text(text).expect("encodes"),
            expected,
            "text {text:?}"
        );
    }
}

#[test]
fn official_encoder_fixture_prompts_tokenize_to_the_recorded_ids() {
    let seam = seam();
    let fixtures = [
        (
            1,
            include_str!("../../../reference/encoding/tests/test_output_1.txt"),
        ),
        (
            2,
            include_str!("../../../reference/encoding/tests/test_output_2.txt"),
        ),
        (
            3,
            include_str!("../../../reference/encoding/tests/test_output_3.txt"),
        ),
        (
            4,
            include_str!("../../../reference/encoding/tests/test_output_4.txt"),
        ),
        (
            5,
            include_str!("../../../reference/encoding/tests/test_output_5.txt"),
        ),
    ];
    let recorded_json = evidence();
    let recorded = recorded_json["fixtures"]
        .as_array()
        .expect("fixture entries");
    for (index, prompt) in fixtures {
        let entry = recorded
            .iter()
            .find(|entry| entry["case"].as_u64() == Some(index as u64))
            .expect("fixture evidence entry");
        assert_eq!(
            seam.encode_text(prompt).expect("encodes"),
            ids(&entry["ids"]),
            "fixture {index}"
        );
    }
}

#[test]
fn flat_special_token_table_is_pinned() {
    // vLLM tokenizes the rendered string with the full added vocabulary, so control-token
    // spellings inside user text become one id. The recipe's special-range split would emit
    // plain characters instead; we deliberately keep the vLLM behaviour (DECISIONS.md task-63).
    // The four ids that matter: 128803 <｜User｜>, 0 <｜begin▁of▁sentence｜>, 128822 </think>,
    // 128825 ｜DSML｜.
    let seam = seam();
    let cases: [(&str, &str, &[u32]); 3] = [
        (
            "user",
            "what is <｜User｜> and </think>?",
            &[
                0, 128799, 77666, 288, 9504, 482, 28, 223, 1328, 343, 16801, 223, 19, 15, 1457, 14,
                270, 4077, 270, 1990, 14, 270, 850, 15146, 270, 22805, 868, 128803, 9602, 344, 223,
                128803, 305, 223, 128822, 33, 128804, 128821,
            ],
        ),
        (
            "bos",
            "hi<｜begin▁of▁sentence｜>there",
            &[
                0, 128799, 77666, 288, 9504, 482, 28, 223, 1328, 343, 16801, 223, 19, 15, 1457, 14,
                270, 4077, 270, 1990, 14, 270, 850, 15146, 270, 22805, 868, 128803, 6366, 0, 21908,
                128804, 128821,
            ],
        ),
        (
            "dsml",
            "explain ｜DSML｜ invoke",
            &[
                0, 128799, 77666, 288, 9504, 482, 28, 223, 1328, 343, 16801, 223, 19, 15, 1457, 14,
                270, 4077, 270, 1990, 14, 270, 850, 15146, 270, 22805, 868, 128803, 26028, 499,
                223, 128825, 34756, 128804, 128821,
            ],
        ),
    ];
    for (name, content, expected) in cases {
        let ids = seam
            .encode_chat(&json!({"messages": [{"role": "user", "content": content}]}))
            .expect("renders");
        assert_eq!(ids, expected, "flat special-token case {name}");
    }
}

#[test]
fn chat_template_ids_match_the_released_server() {
    let seam = seam();
    let recorded_json = evidence();
    for entry in recorded_json["chat"].as_array().expect("chat entries") {
        let name = entry["name"].as_str().expect("case name");
        let request = named_chat_case(name);
        assert_eq!(
            seam.encode_chat(&request).expect("renders"),
            ids(&entry["ids"]),
            "case {name}"
        );
    }
}

#[test]
fn chat_template_defaults_match_the_released_server() {
    // vLLM defaults: thinking on, budget "high" (50). The released tokenizer wrapper sets
    // thinking=True and effort="high" when the request says nothing.
    let seam = seam();
    let default_ids = seam
        .encode_chat(&json!({"messages": [{"role": "user", "content": "Hello"}]}))
        .expect("renders");
    let explicit_high = seam
        .encode_chat(&json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}
        }))
        .expect("renders");
    assert_eq!(default_ids, explicit_high);
    assert_ne!(
        default_ids,
        seam.encode_chat(&json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"thinking": false}
        }))
        .expect("renders")
    );
}

#[test]
fn chat_template_refuses_unknown_roles_loudly() {
    let error = seam()
        .encode_chat(&json!({"messages": [{"role": "wizard", "content": "hi"}]}))
        .expect_err("unknown role must be refused");
    assert!(
        matches!(error, TokenizerSeamError::InvalidRole { .. }),
        "{error:?}"
    );
    assert_eq!(error.to_string(), "Invalid role: wizard");
}

#[test]
fn chat_template_refuses_unsupported_effort_loudly() {
    // "medium" is accepted by the OpenAI surface but is not one of the encoder's aliases; the
    // released server raises this exact message from deepseek_v41.py.
    let error = seam()
        .encode_chat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium"
        }))
        .expect_err("unsupported effort must be refused");
    assert!(
        matches!(error, TokenizerSeamError::InvalidReasoningEffort),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, or an integer within [1, 100] in chat_template_kwargs"
    );
}

#[test]
fn chat_template_refuses_unsupported_content_parts_loudly() {
    let error = seam()
        .encode_chat(&json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "audio", "audio": "data"}]
            }]
        }))
        .expect_err("unsupported part must be refused");
    assert!(
        matches!(error, TokenizerSeamError::UnsupportedContentPart { .. }),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "DeepSeek V4.1 supports text and image content only; got 'audio'"
    );
}

#[test]
fn developer_role_matches_the_released_encoder() {
    // vLLM's normaliser accepts developer. The encoder then either drops the message in
    // thinking mode (drop_thinking only keeps known roles) or refuses it in chat mode. We match
    // the encoder rather than silently rendering a prompt the reference server would not.
    let seam = seam();
    seam.encode_chat(&json!({
        "messages": [
            {"role": "developer", "content": "be nice"},
            {"role": "user", "content": "hi"}
        ]
    }))
    .expect("thinking mode drops the developer message, as the encoder does");
    let error = seam
        .encode_chat(&json!({
            "messages": [
                {"role": "developer", "content": "be nice"},
                {"role": "user", "content": "hi"}
            ],
            "chat_template_kwargs": {"thinking": false}
        }))
        .expect_err("chat mode must refuse the developer message");
    assert!(
        matches!(
            error,
            TokenizerSeamError::Render(CaseError::UnknownRole(ref role)) if role == "developer"
        ),
        "{error:?}"
    );
}

fn app_with_seam() -> Router {
    let engine = MockEngine::new(MODEL)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .with_capabilities(Capabilities::ALL)
        .content(&["Hello", " world"]);
    router(
        AppState::new(Arc::new(engine))
            .with_fixed_time(FIXED_TIME)
            .with_tokenizer(Arc::new(seam())),
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

#[tokio::test]
async fn completion_text_is_tokenized_through_the_http_path() {
    let (status, body) = post(
        app_with_seam(),
        "/v1/completions",
        json!({"model": MODEL, "prompt": "Hello, world!", "max_tokens": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Evidence: "Hello, world!" is four checkpoint tokens ([19923, 14, 2058, 3]).
    assert_eq!(body["usage"]["prompt_tokens"], 4, "{body}");
    assert_eq!(body["choices"][0]["text"], "Hello");
}

#[tokio::test]
async fn completion_token_id_prompts_pass_through_and_echo_round_trips() {
    let (status, body) = post(
        app_with_seam(),
        "/v1/completions",
        json!({
            "model": MODEL,
            "prompt": [19923, 14, 2058, 3],
            "echo": true,
            "max_tokens": 0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"], 4, "{body}");
    // The decoded prompt text is the echo: the tokenizer seam closes the round trip.
    assert_eq!(body["choices"][0]["text"], "Hello, world!");
}

#[tokio::test]
async fn chat_request_renders_the_released_template_through_the_http_path() {
    let (status, body) = post(
        app_with_seam(),
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Evidence: the default (thinking on, budget high) prompt is 31 checkpoint tokens.
    assert_eq!(body["usage"]["prompt_tokens"], 31, "{body}");
}

#[tokio::test]
async fn chat_nothink_request_uses_the_short_template() {
    let (status, body) = post(
        app_with_seam(),
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"thinking": false},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Evidence: non-thinking "Hello" is five tokens (<BOS><User>Hello<Assistant></think>).
    assert_eq!(body["usage"]["prompt_tokens"], 5, "{body}");
}

#[tokio::test]
async fn chat_invalid_role_is_a_400_with_the_vllm_text() {
    let (status, body) = post(
        app_with_seam(),
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": [{"role": "wizard", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["message"], "Invalid role: wizard");
    assert_eq!(body["error"]["code"], 400);
}

#[tokio::test]
async fn chat_unsupported_effort_is_a_400_with_the_vllm_text() {
    let (status, body) = post(
        app_with_seam(),
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["message"],
        "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, or an integer within [1, 100] in chat_template_kwargs"
    );
    assert_eq!(body["error"]["code"], 400);
}

fn app_without_seam() -> Router {
    let engine = MockEngine::new(MODEL)
        .with_cluster_stats(ClusterStatsSnapshot::default())
        .with_capabilities(Capabilities::ALL)
        .content(&["Hello", " world"]);
    router(AppState::new(Arc::new(engine)).with_fixed_time(FIXED_TIME))
}

#[tokio::test]
async fn without_a_seam_the_placeholder_path_is_unchanged() {
    // Weight-free router tests (and anything that builds AppState without a checkpoint
    // tokenizer) keep the Milestone 1 heuristic; this test pins that the seam is opt-in.
    let (status, body) = post(
        app_without_seam(),
        "/v1/completions",
        json!({"model": MODEL, "prompt": "Hello, world!", "max_tokens": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Four characters per token heuristic: 13 characters -> 4 estimated tokens.
    assert_eq!(body["usage"]["prompt_tokens"], 4, "{body}");
}
