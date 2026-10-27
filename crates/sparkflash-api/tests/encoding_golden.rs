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

//! Golden prompt-vector tests against the official Python encoder outputs.
//!
//! For each of the five vectors the JSON input is parsed with the reference
//! schema (#[EncodingCase]) and rendered; the rendered prompt must equal the
//! checked-in .txt golden byte for byte.

use std::path::{Path, PathBuf};

use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::{ImageDetail, ImageSource};
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::dsv41::DeepseekV41Encoding;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sparkflash_model_deepseek_v41::encoding::{
    EncodingCase, HfTokenizerEncoder, IMAGE_PLACEHOLDER, MessageContent, OpenAiTool,
    SparkV41Encoding, ThinkingMode, TokenizerSeam, load_first_case, load_hf_tokenizer,
    reference_tokenizer_path,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/encoding/tests")
}

fn read_fixture(case_id: u32, suffix: &str, extension: &str) -> String {
    let path = fixtures_dir().join(format!("{suffix}_{case_id}.{extension}"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn load_golden_case(case_id: u32) -> EncodingCase {
    load_first_case(&read_fixture(case_id, "test_input", "json"))
        .unwrap_or_else(|error| panic!("failed to parse vector {case_id}: {error}"))
}

fn golden_text(case_id: u32) -> String {
    read_fixture(case_id, "test_output", "txt")
}

fn first_100(text: &str) -> String {
    text.chars().take(100).collect()
}

fn assert_prompt_eq(case_id: u32, actual: &str, expected: &str) {
    assert!(
        !expected.is_empty(),
        "vector {case_id}: the golden fixture is empty, refusing to compare against it"
    );
    assert_eq!(
        actual.len(),
        expected.len(),
        "vector {case_id}: rendered {} bytes but the golden prompt is {} bytes\n  actual  [{:?}]\n  expect  [{:?}]\n  actual chars {} / expected chars {}",
        actual.len(),
        expected.len(),
        first_100(actual),
        first_100(expected),
        actual.chars().count(),
        expected.chars().count()
    );
    assert_eq!(
        actual,
        expected,
        "vector {case_id} mismatch\n  actual  [{:?}]\n  expect  [{:?}]\n  actual chars {} / expected chars {}",
        first_100(actual),
        first_100(expected),
        actual.chars().count(),
        expected.chars().count()
    );
}

fn assert_golden(case_id: u32) {
    let case = load_golden_case(case_id);
    let rendered = SparkV41Encoding::new()
        .render_case(&case, None)
        .unwrap_or_else(|error| panic!("vector {case_id} failed to render: {error}"));
    assert_prompt_eq(case_id, &rendered.prompt, &golden_text(case_id));
}

#[test]
fn golden_vector_1_tool_call_thinking() {
    assert_golden(1);
}

#[test]
fn golden_vector_2_multiturn_chat() {
    assert_golden(2);
}

#[test]
fn golden_vector_3_chinese_tools_mid_system() {
    assert_golden(3);
}

#[test]
fn golden_vector_4_quick_instruction_task() {
    assert_golden(4);
}

#[test]
fn golden_vector_5_vision_numeric_effort() {
    assert_golden(5);
}

// ---------------------------------------------------------------------------
// Conversation-path cross checks
//
// The reference case schema is richer than deepseek-recipe-core Conversation:
// it can attach tools to a mid-conversation system message and can carry
// quick-instruction task tokens. Vectors 3 and 4 therefore only have a
// case-path rendering; the representable vectors are also rendered through
// Conversation to prove the two paths agree.
// ---------------------------------------------------------------------------

fn message_text(content: &Option<MessageContent>) -> String {
    match content {
        Some(MessageContent::Text(text)) => text.clone(),
        Some(MessageContent::Blocks(_)) | None => String::new(),
    }
}

fn user_content(content: &Option<MessageContent>) -> (String, Vec<ImageSource>) {
    match content {
        Some(MessageContent::Text(text)) => (text.clone(), Vec::new()),
        Some(MessageContent::Blocks(blocks)) => {
            let mut parts = Vec::new();
            let mut images = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => parts.push(
                        block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    Some("image_url" | "image") => {
                        parts.push(IMAGE_PLACEHOLDER.to_owned());
                        let url = block
                            .pointer("/image_url/url")
                            .or_else(|| block.get("url"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        images.push(ImageSource::DataUrl {
                            data_url: url,
                            detail: ImageDetail::High,
                        });
                    }
                    _ => {}
                }
            }
            (parts.join("\n\n"), images)
        }
        None => (String::new(), Vec::new()),
    }
}

fn tool_definition(tool: &OpenAiTool) -> Option<ToolDefinition> {
    let function = tool.function.as_object()?;
    Some(ToolDefinition {
        name: function.get("name")?.as_str()?.to_owned(),
        description: function
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        parameters: function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| Value::Object(serde_json::Map::new())),
        strict: function.get("strict").and_then(Value::as_bool),
    })
}

fn parse_reasoning_effort(value: Option<&Value>) -> Option<ReasoningEffort> {
    match value.and_then(Value::as_str) {
        Some("low") => Some(ReasoningEffort::Low),
        Some("high") => Some(ReasoningEffort::High),
        Some("xhigh") => Some(ReasoningEffort::Xhigh),
        Some("max") => Some(ReasoningEffort::Max),
        _ => None,
    }
}

/// Build a Conversation for the representable subset of the reference schema.
///
/// Returns None when a message carries a task token or tools on a
/// non-leading message, neither of which Conversation can express.
fn case_to_conversation(case: &EncodingCase) -> Option<Conversation> {
    let thinking = case.thinking_mode().ok()?;
    let mut tools = Vec::new();
    let mut messages = Vec::new();
    for (index, message) in case.messages.iter().enumerate() {
        if message.task.is_some() {
            return None;
        }
        match message.role.as_str() {
            "system" => {
                if let Some(message_tools) = &message.tools {
                    if index != 0 {
                        return None;
                    }
                    for tool in message_tools {
                        tools.push(tool_definition(tool)?);
                    }
                }
                messages.push(InputMessage::System {
                    content: message_text(&message.content),
                });
            }
            "user" => {
                let (content, image_sources) = user_content(&message.content);
                messages.push(InputMessage::User {
                    content,
                    image_sources,
                });
            }
            "assistant" => {
                messages.push(InputMessage::Assistant {
                    content: message_text(&message.content),
                    reasoning_content: message.reasoning_content.clone(),
                    tool_calls: message.tool_calls.as_ref().map(|calls| {
                        calls
                            .iter()
                            .map(|call| ToolCall {
                                id: call.id.clone().unwrap_or_default(),
                                name: call.function.name.clone(),
                                arguments: call
                                    .function
                                    .arguments
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_owned(),
                            })
                            .collect()
                    }),
                });
            }
            "tool" => {
                let (content, image_sources) = user_content(&message.content);
                messages.push(InputMessage::Tool {
                    content,
                    image_sources,
                    tool_call_id: message.tool_call_id.clone().unwrap_or_default(),
                });
            }
            _ => return None,
        }
    }
    Some(Conversation {
        messages,
        thinking_mode: thinking == ThinkingMode::Thinking,
        tools,
        tool_choice: ToolChoice::Auto,
        reasoning_effort: parse_reasoning_effort(case.reasoning_effort.as_ref()),
        response_format: ResponseFormat::Text,
    })
}

#[test]
fn conversation_path_matches_golden_vector_1() {
    let case = load_golden_case(1);
    let conversation = case_to_conversation(&case).expect("vector 1 is representable");
    let rendered = SparkV41Encoding::new().render_conversation(&conversation, Some(50));
    assert_prompt_eq(1, &rendered.prompt, &golden_text(1));
}

#[test]
fn conversation_path_matches_golden_vector_2() {
    let case = load_golden_case(2);
    let conversation = case_to_conversation(&case).expect("vector 2 is representable");
    let rendered = SparkV41Encoding::new().render_conversation(&conversation, None);
    assert_prompt_eq(2, &rendered.prompt, &golden_text(2));
}

#[test]
fn conversation_path_matches_golden_vector_5() {
    let case = load_golden_case(5);
    let conversation = case_to_conversation(&case).expect("vector 5 is representable");
    let rendered = SparkV41Encoding::new().render_conversation(&conversation, None);
    assert_prompt_eq(5, &rendered.prompt, &golden_text(5));
    assert_eq!(rendered.image_sources.len(), 2);
}

// ---------------------------------------------------------------------------
// Upstream parity check
// ---------------------------------------------------------------------------

#[test]
fn upstream_recipe_and_port_agree_on_simple_chat() {
    let conversation = Conversation {
        messages: vec![
            InputMessage::System {
                content: "You are a helpful assistant.".to_owned(),
            },
            InputMessage::User {
                content: "Hello".to_owned(),
                image_sources: Vec::new(),
            },
            InputMessage::Assistant {
                content: "Hi there!".to_owned(),
                reasoning_content: None,
                tool_calls: None,
            },
            InputMessage::User {
                content: "How are you?".to_owned(),
                image_sources: Vec::new(),
            },
        ],
        thinking_mode: false,
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        reasoning_effort: None,
        response_format: ResponseFormat::Text,
    };

    let upstream = DeepseekV41Encoding::new().render_conversation(&conversation);
    let port = SparkV41Encoding::new().render_conversation(&conversation, None);
    assert_eq!(port.prompt, upstream.prompt);
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[test]
fn hf_tokenizer_encodes_when_vendored_file_is_present() {
    let path = reference_tokenizer_path();
    if !path.exists() {
        eprintln!("skipping: {} is not present", path.display());
        return;
    }
    let tokenizer = load_hf_tokenizer(&path).expect("load vendored tokenizer");
    let ids = <HfTokenizerEncoder as deepseek_recipe_encoding::TokenizerEncoder>::encode_ids(
        &tokenizer,
        "<｜begin▁of▁sentence｜><｜User｜>Hello",
    )
    .expect("tokenize");
    assert!(!ids.is_empty(), "tokenizer returned no ids");

    let conversation = Conversation {
        messages: vec![InputMessage::User {
            content: "Hello".to_owned(),
            image_sources: Vec::new(),
        }],
        thinking_mode: false,
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        reasoning_effort: None,
        response_format: ResponseFormat::Text,
    };
    let encoding = SparkV41Encoding::new().with_tokenizer(tokenizer);
    let ids = encoding.encode(&conversation, None).expect("encode");
    assert!(!ids.is_empty(), "encoding returned no ids");
}

#[test]
fn encode_without_tokenizer_reports_missing_tokenizer() {
    let conversation = Conversation {
        messages: vec![InputMessage::User {
            content: "Hello".to_owned(),
            image_sources: Vec::new(),
        }],
        thinking_mode: false,
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        reasoning_effort: None,
        response_format: ResponseFormat::Text,
    };
    let error = SparkV41Encoding::new()
        .encode(&conversation, None)
        .expect_err("no tokenizer attached");
    assert!(matches!(
        error,
        deepseek_recipe_encoding::EncodingError::MissingTokenizer
    ));
}

#[test]
fn golden_vectors_parse_and_render_without_network_or_weights() {
    // The library must not require a tokenizer to render prompts.
    for case_id in 1..=5 {
        let case = load_golden_case(case_id);
        let encoding: SparkV41Encoding = SparkV41Encoding::default();
        let rendered = encoding
            .render_case(&case, None)
            .unwrap_or_else(|error| panic!("vector {case_id}: {error}"));
        assert!(!rendered.prompt.is_empty());
    }
}

#[test]
fn mid_conversation_system_and_tool_placement_are_case_path_only() {
    // Vector 3 attaches tools to the third message; that is not expressible as
    // a Conversation, so the conversation conversion must refuse it.
    let case = load_golden_case(3);
    assert!(case_to_conversation(&case).is_none());
    // Vector 4 carries a task token; also not expressible.
    let case = load_golden_case(4);
    assert!(case_to_conversation(&case).is_none());
    // Both still render through the case path.
    let encoding = SparkV41Encoding::new();
    for case_id in [3, 4] {
        let case = load_golden_case(case_id);
        let rendered = encoding.render_case(&case, None).expect("case render");
        assert_prompt_eq(case_id, &rendered.prompt, &golden_text(case_id));
    }
}

// ---------------------------------------------------------------------------
// The vLLM id pin: every fixture prompt hash and id array must stay put.
// ---------------------------------------------------------------------------

fn task30_oracle() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/build/task30-evidence/tokenizer-parity.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn fixture_prompt_hashes_and_ids_match_the_recorded_vllm_oracle() {
    let oracle = task30_oracle();
    let fixtures = oracle["fixtures"].as_array().expect("fixtures array");
    let tokenizer =
        load_hf_tokenizer(&reference_tokenizer_path()).expect("load reference tokenizer");
    for case_id in 1..=5u32 {
        let case = load_golden_case(case_id);
        let rendered = SparkV41Encoding::new()
            .render_case(&case, None)
            .unwrap_or_else(|error| panic!("vector {case_id} failed to render: {error}"));
        let expected = fixtures
            .iter()
            .find(|entry| entry["case"].as_u64() == Some(u64::from(case_id)))
            .unwrap_or_else(|| panic!("oracle has no case {case_id}"));
        assert_eq!(
            sha256_hex(rendered.prompt.as_bytes()),
            expected["prompt_sha256"].as_str().expect("prompt hash"),
            "case {case_id}: rendered prompt hash moved"
        );
        let ids = <HfTokenizerEncoder as deepseek_recipe_encoding::TokenizerEncoder>::encode_ids(
            &tokenizer,
            &rendered.prompt,
        )
        .expect("tokenize rendered prompt");
        let expected_ids: Vec<u32> = expected["ids"]
            .as_array()
            .expect("ids array")
            .iter()
            .map(|value| u32::try_from(value.as_u64().expect("id")).expect("u32 id"))
            .collect();
        assert_eq!(ids, expected_ids, "case {case_id}: prompt ids moved");
        assert_eq!(
            ids.len(),
            expected["n_ids"].as_u64().expect("n_ids") as usize
        );
    }
}

#[test]
fn pinned_user_hello_vector_is_exactly_31_ids() {
    let oracle = task30_oracle();
    let expected = oracle["chat"]
        .as_array()
        .expect("chat array")
        .iter()
        .find(|entry| entry["name"] == "chat_user_hello_default")
        .expect("hello oracle");
    let seam = TokenizerSeam::from_path(&reference_tokenizer_path()).expect("load seam tokenizer");
    let request = json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let ids = seam.encode_chat(&request).expect("encode hello");
    let expected_ids: Vec<u32> = expected["ids"]
        .as_array()
        .expect("ids array")
        .iter()
        .map(|value| u32::try_from(value.as_u64().expect("id")).expect("u32 id"))
        .collect();
    assert_eq!(
        ids, expected_ids,
        "the pinned 31-id user:Hello vector moved"
    );
    assert_eq!(ids.len(), 31);
}
