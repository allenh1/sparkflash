//! Regression tests for the reference encoder's top-level "context" field.
//!
//! encoding.py merges the context (line 729) and then uses the *merged* length
//! both to suppress the default BOS token and as the index offset into
//! full_messages (lines 734, 747, 751). The context messages themselves are
//! never rendered: only the current-turn messages are. Every expected string in
//! this file was produced by running reference/encoding/encoding.py directly.

use serde_json::{Value, json};
use sparkflash_api::encoding::{SparkV41Encoding, parse_cases};

const BOS: &str = "<｜begin▁of▁sentence｜>";
const SYSTEM: &str = "<｜System｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const THINK_START: &str = "<think>";

const EFFORT_50: &str =
    "Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

/// P1: chat turn with a two-message context.
const P1: &str = "<｜User｜>second question<｜Assistant｜></think>";
/// P2: thinking turn with a two-message context.
const P2: &str = "<｜User｜>second question<｜Assistant｜><think>";
/// P4: no context, thinking, default effort.
fn p4() -> String {
    format!("{BOS}{SYSTEM}{EFFORT_50}{USER}second question{ASSISTANT}{THINK_START}")
}

fn render(case_json: Value) -> String {
    let cases = parse_cases(&case_json).expect("case parses");
    assert_eq!(cases.len(), 1, "expected exactly one case");
    SparkV41Encoding::new()
        .render_case(&cases[0], None)
        .expect("case renders")
        .prompt
}

fn context_two_messages() -> Value {
    json!([
        { "role": "user", "content": "first question" },
        { "role": "assistant", "content": "first answer" }
    ])
}

fn assert_bytes(name: &str, actual: &str, expected: &str) {
    assert_eq!(
        actual,
        expected,
        "{name} mismatch\n  actual  [{actual:?}]\n  expect  [{expected:?}]\n  actual chars {} / expected chars {}",
        actual.chars().count(),
        expected.chars().count()
    );
}

#[test]
fn context_offsets_indexing_without_rendering_context_text() {
    let chat = render(json!({
        "thinking_mode": "chat",
        "context": context_two_messages(),
        "messages": [{ "role": "user", "content": "second question" }]
    }));
    assert_bytes("chat context", &chat, P1);

    let thinking = render(json!({
        "thinking_mode": "thinking",
        "context": context_two_messages(),
        "messages": [{ "role": "user", "content": "second question" }]
    }));
    assert_bytes("thinking context", &thinking, P2);

    // The context also suppresses the default BOS token.
    assert!(
        !chat.starts_with(BOS),
        "context must suppress the default BOS"
    );
    assert!(!thinking.starts_with(BOS));
}

#[test]
fn context_length_is_measured_after_tool_merging() {
    // Raw context has 3 messages; merging the two tool results leaves 2.
    // Using the raw length made the thinking path render 0 bytes.
    let context = json!([
        {
            "role": "assistant",
            "content": "a",
            "tool_calls": [
                { "type": "function", "function": { "name": "f", "arguments": "{}" } }
            ]
        },
        { "role": "tool", "content": "r1", "tool_call_id": "1" },
        { "role": "tool", "content": "r2", "tool_call_id": "2" }
    ]);

    let thinking = render(json!({
        "thinking_mode": "thinking",
        "context": context.clone(),
        "messages": [{ "role": "user", "content": "second question" }]
    }));
    assert!(!thinking.is_empty(), "thinking render must not be empty");
    assert_bytes("thinking tool-context", &thinking, P2);

    let chat = render(json!({
        "thinking_mode": "chat",
        "context": context,
        "messages": [{ "role": "user", "content": "second question" }]
    }));
    assert_bytes("chat tool-context", &chat, P1);
}

#[test]
fn reasoning_effort_line_is_not_emitted_when_context_offsets_index_zero() {
    let rendered = render(json!({
        "thinking_mode": "thinking",
        "reasoning_effort": 42,
        "context": context_two_messages(),
        "messages": [{ "role": "user", "content": "second question" }]
    }));
    assert_bytes("effort with context", &rendered, P2);
    assert!(
        !rendered.contains("Reasoning Effort:"),
        "the effort line belongs to index 0, which is inside the context"
    );
}

#[test]
fn empty_context_is_indistinguishable_from_absent_context() {
    let absent = render(json!({
        "thinking_mode": "thinking",
        "messages": [{ "role": "user", "content": "second question" }]
    }));
    assert_bytes("absent context", &absent, &p4());

    let empty = render(json!({
        "thinking_mode": "thinking",
        "context": [],
        "messages": [{ "role": "user", "content": "second question" }]
    }));
    assert_bytes("empty context", &empty, &p4());
    assert_eq!(empty, absent);
}

#[test]
fn a_context_without_messages_renders_nothing() {
    let chat = render(json!({
        "thinking_mode": "chat",
        "context": context_two_messages(),
        "messages": []
    }));
    assert_eq!(chat, "");

    let thinking = render(json!({
        "thinking_mode": "thinking",
        "context": context_two_messages(),
        "messages": []
    }));
    assert_eq!(thinking, "");
}
