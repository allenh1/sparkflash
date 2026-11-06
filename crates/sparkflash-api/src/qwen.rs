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

//! Qwen3.8-Flash-Next's output dialect: `<think>` reasoning and `<tool_call>` XML.
//!
//! The prompt codec renders the released ChatML template, which ends a thinking request with
//! `<|im_start|>assistant\n<think>\n` and teaches the model to answer in
//! `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` XML (see
//! `sparkflash-model-qwen4-exp::codec`). The raw engine stream therefore carries both channels as
//! one content stream. [`QwenResponseEngine`] is the adapter that splits them back into the
//! engine's reasoning/content/tool-call events, so every API surface (Chat Completions, Responses,
//! Anthropic Messages) sees the same shape it sees from the V4.1 engine.
//!
//! The streaming parser holds back only the bytes that could still become a `</think>` or
//! `<tool_call>` marker. A tool block is parsed once its `</tool_call>` arrives; the resulting
//! arguments object is emitted as one fragment, so concatenating fragments still yields exactly
//! the buffered parser's OpenAI `arguments` string.

use std::sync::Arc;

use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::Value;

use crate::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, InferenceEngine,
    MissingCapability,
};
use crate::telemetry_types::{DeepSeekSparkRequestStats, OpenAiUsageWithCache};
use crate::tool_block::{Block, BlockSplitter, CALL_CLOSE, CALL_OPEN};

const THINK_CLOSE: &str = "</think>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// One parsed Qwen answer fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenEvent {
    /// A fragment of the reasoning channel.
    Reasoning(String),
    /// Visible answer text.
    Content(String),
    /// A complete function call.
    ArgumentFragment {
        /// Position in the emitted tool-call array.
        index: usize,
        /// Function name.
        name: String,
        /// The complete OpenAI `arguments` JSON string.
        arguments: String,
    },
}

/// A buffered Qwen answer split into the channels the API exposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedQwenCompletion {
    /// Text before `</think>`, when the request enabled thinking.
    pub reasoning: String,
    /// Visible content outside tool-call blocks.
    pub content: String,
    /// Tool calls in emission order.
    pub tool_calls: Vec<ParsedQwenToolCall>,
}

/// One tool call lifted out of a Qwen completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedQwenToolCall {
    /// Function name from `<function=...>`.
    pub name: String,
    /// OpenAI `arguments` JSON string.
    pub arguments: String,
}

/// Parses a complete Qwen answer.
#[must_use]
pub fn parse_completion(text: &str, thinking: bool) -> ParsedQwenCompletion {
    let mut parser = QwenStreamParser::new(thinking);
    let mut events = parser.feed(text);
    events.extend(parser.finish());
    let mut parsed = ParsedQwenCompletion {
        reasoning: String::new(),
        content: String::new(),
        tool_calls: Vec::new(),
    };
    for event in events {
        match event {
            QwenEvent::Reasoning(text) => parsed.reasoning.push_str(&text),
            QwenEvent::Content(text) => parsed.content.push_str(&text),
            QwenEvent::ArgumentFragment {
                name, arguments, ..
            } => parsed
                .tool_calls
                .push(ParsedQwenToolCall { name, arguments }),
        }
    }
    parsed
}

/// The incremental Qwen answer parser.
#[derive(Debug)]
pub struct QwenStreamParser {
    thinking: bool,
    reasoning_done: bool,
    newlines_after_reasoning: usize,
    buffer: String,
    /// The content stream, split into text and tool-call bodies by the shared framer.
    blocks: BlockSplitter,
    calls: usize,
}

impl QwenStreamParser {
    /// A parser for one answer. `thinking` comes from the resolved request and says whether the
    /// prompt opened a `<think>` block, so a non-thinking answer is never mistaken for reasoning.
    #[must_use]
    pub fn new(thinking: bool) -> Self {
        Self {
            thinking,
            reasoning_done: !thinking,
            newlines_after_reasoning: 0,
            buffer: String::new(),
            blocks: BlockSplitter::new(),
            calls: 0,
        }
    }

    /// How many complete tool calls the parser has emitted.
    #[must_use]
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Consumes one content delta.
    pub fn feed(&mut self, text: &str) -> Vec<QwenEvent> {
        if text.is_empty() {
            return Vec::new();
        }
        self.buffer.push_str(text);
        let mut events = Vec::new();
        if self.thinking && !self.reasoning_done {
            let Some(index) = self.buffer.find(THINK_CLOSE) else {
                let cut = hold_back(&self.buffer, &[THINK_CLOSE]);
                if cut > 0 {
                    let reasoning = self.buffer[..cut].to_owned();
                    self.buffer.drain(..cut);
                    push_reasoning(&mut events, reasoning);
                }
                return events;
            };
            let mut reasoning = self.buffer[..index].to_owned();
            if let Some(rest) = reasoning.strip_prefix("<think>") {
                reasoning = rest.strip_prefix('\n').unwrap_or(rest).to_owned();
            }
            self.buffer.drain(..index + THINK_CLOSE.len());
            self.newlines_after_reasoning = 2;
            self.skip_after_reasoning();
            push_reasoning(&mut events, reasoning);
            self.reasoning_done = true;
        }
        self.skip_after_reasoning();
        events.extend(self.advance_content());
        events
    }

    /// Flushes the end of the answer.
    pub fn finish(&mut self) -> Vec<QwenEvent> {
        let mut events = Vec::new();
        if self.thinking && !self.reasoning_done {
            let mut reasoning = std::mem::take(&mut self.buffer);
            if let Some(rest) = reasoning.strip_prefix("<think>") {
                reasoning = rest.strip_prefix('\n').unwrap_or(rest).to_owned();
            }
            push_reasoning(&mut events, reasoning);
            self.reasoning_done = true;
        }
        self.skip_after_reasoning();
        events.extend(self.advance_content());
        // An unterminated or malformed call stays visible: dropping it would lose the model's
        // answer, and inventing a call from a partial block would be worse.
        let tail = self.blocks.finish();
        events.extend(self.emit(tail));
        events
    }

    /// Consumes the template's blank line after `</think>`. The marker and the newlines can be
    /// split across deltas, so the count has to survive a feed boundary.
    fn skip_after_reasoning(&mut self) {
        while self.newlines_after_reasoning > 0 && self.buffer.starts_with('\n') {
            self.buffer.drain(..1);
            self.newlines_after_reasoning -= 1;
        }
        if !self.buffer.is_empty() && !self.buffer.starts_with('\n') {
            self.newlines_after_reasoning = 0;
        }
    }

    /// Drains the held-back content into the shared framer.
    fn advance_content(&mut self) -> Vec<QwenEvent> {
        let text = std::mem::take(&mut self.buffer);
        let blocks = self.blocks.feed(&text);
        self.emit(blocks)
    }

    /// Turns framed blocks into events: text stays visible, a body this arm can spell becomes a
    /// call, and one it cannot keeps its markers.
    fn emit(&mut self, blocks: Vec<Block>) -> Vec<QwenEvent> {
        let mut events = Vec::new();
        for block in blocks {
            match block {
                Block::Text(content) => events.push(QwenEvent::Content(content)),
                Block::Call(body) => match parse_tool_block(&body) {
                    Some((name, arguments)) => {
                        let index = self.calls;
                        self.calls += 1;
                        events.push(QwenEvent::ArgumentFragment {
                            index,
                            name,
                            arguments,
                        });
                    }
                    None => {
                        events.push(QwenEvent::Content(format!("{CALL_OPEN}{body}{CALL_CLOSE}")))
                    }
                },
            }
        }
        events
    }
}

fn push_reasoning(events: &mut Vec<QwenEvent>, reasoning: String) {
    if !reasoning.is_empty() {
        events.push(QwenEvent::Reasoning(reasoning));
    }
}

/// Parses the body of one `<tool_call>` block.
fn parse_tool_block(body: &str) -> Option<(String, String)> {
    let body = body.trim();
    if body.starts_with(FUNCTION_OPEN) {
        parse_xml_function(body)
    } else if body.starts_with('{') {
        parse_json_function(body)
    } else {
        None
    }
}

fn parse_xml_function(body: &str) -> Option<(String, String)> {
    let rest = body.strip_prefix(FUNCTION_OPEN)?;
    let (name, mut rest) = rest.split_once('>')?;
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let mut arguments = serde_json::Map::new();
    loop {
        rest = rest.trim_start();
        if let Some(after) = rest.strip_prefix(FUNCTION_CLOSE) {
            return after
                .trim()
                .is_empty()
                .then(|| (name.to_owned(), Value::Object(arguments).to_string()));
        }
        let after = rest.strip_prefix(PARAMETER_OPEN)?;
        let (parameter, after) = after.split_once('>')?;
        let parameter = parameter.trim();
        if parameter.is_empty() {
            return None;
        }
        let end = after.find(PARAMETER_CLOSE)?;
        let mut value = &after[..end];
        if let Some(stripped) = value.strip_prefix('\n') {
            value = stripped;
        }
        if let Some(stripped) = value.strip_suffix('\n') {
            value = stripped;
        }
        let value = serde_json::from_str::<Value>(value)
            .unwrap_or_else(|_| Value::String(value.to_owned()));
        arguments.insert(parameter.to_owned(), value);
        rest = &after[end + PARAMETER_CLOSE.len()..];
    }
}

fn parse_json_function(body: &str) -> Option<(String, String)> {
    let value: Value = serde_json::from_str(body).ok()?;
    let object = value.as_object()?;
    let function = object.get("function").unwrap_or(&value);
    let function = function.as_object()?;
    let name = function.get("name")?.as_str()?;
    let arguments = function.get("arguments")?;
    let arguments = match arguments {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).ok()?,
    };
    Some((name.to_owned(), arguments))
}

/// The longest suffix of `buffer` that could still become one of `markers`.
fn hold_back(buffer: &str, markers: &[&str]) -> usize {
    let longest = markers.iter().map(|marker| marker.len()).max().unwrap_or(0);
    let mut cut = buffer.len();
    for (offset, _) in buffer.char_indices().rev() {
        let suffix = &buffer[offset..];
        if suffix.len() >= longest {
            break;
        }
        if markers.iter().any(|marker| marker.starts_with(suffix)) {
            cut = offset;
        }
    }
    cut
}

/// An engine adapter that turns Qwen's raw content stream into normalized engine events.
///
/// The wrapped engine's capabilities, telemetry and health are forwarded unchanged; only the
/// decode-event shape is rewritten.
pub struct QwenResponseEngine {
    inner: Arc<dyn InferenceEngine>,
}

impl QwenResponseEngine {
    /// Wraps an engine serving Qwen3.8-Flash-Next.
    #[must_use]
    pub fn new(inner: Arc<dyn InferenceEngine>) -> Self {
        Self { inner }
    }

    /// Wraps an engine as a shared trait object.
    #[must_use]
    pub fn layer(inner: Arc<dyn InferenceEngine>) -> Arc<dyn InferenceEngine> {
        Arc::new(Self::new(inner))
    }
}

impl InferenceEngine for QwenResponseEngine {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        let thinking = request.thinking;
        let inner = Arc::clone(&self.inner);
        Box::pin(async_stream::stream! {
            let mut parser = QwenStreamParser::new(thinking);
            let mut events = inner.generate(request);
            let mut finished: Option<(
                FinishReason,
                OpenAiUsageWithCache,
                Box<DeepSeekSparkRequestStats>,
            )> = None;
            while let Some(event) = events.next().await {
                match event {
                    EngineEvent::ContentDelta(text) => {
                        for parsed in parser.feed(&text) {
                            yield qwen_engine_event(parsed);
                        }
                    }
                    EngineEvent::Finished {
                        finish_reason,
                        usage,
                        stats,
                    } => finished = Some((finish_reason, usage, stats)),
                    EngineEvent::Error { .. } => {
                        yield event;
                        return;
                    }
                    other => yield other,
                }
            }
            for parsed in parser.finish() {
                yield qwen_engine_event(parsed);
            }
            if let Some((finish_reason, usage, stats)) = finished {
                let finish_reason = if parser.calls() > 0 {
                    FinishReason::ToolCalls
                } else {
                    finish_reason
                };
                yield EngineEvent::Finished {
                    finish_reason,
                    usage,
                    stats,
                };
            }
        })
    }

    fn vocab_size(&self) -> Option<usize> {
        self.inner.vocab_size()
    }

    fn independent_choices(&self) -> bool {
        self.inner.independent_choices()
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.inner.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    fn health(&self) -> Result<(), String> {
        self.inner.health()
    }

    fn validate_request(&self, request: &EngineRequest) -> Result<(), MissingCapability> {
        self.inner.validate_request(request)
    }
}

fn qwen_engine_event(event: QwenEvent) -> EngineEvent {
    match event {
        QwenEvent::Reasoning(text) => EngineEvent::ReasoningDelta(text),
        QwenEvent::Content(text) => EngineEvent::ContentDelta(text),
        QwenEvent::ArgumentFragment {
            index,
            name,
            arguments,
        } => EngineEvent::ToolCallDelta {
            index,
            id: Some(format!("call_{index}")),
            name: Some(name),
            arguments_delta: arguments,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineRequest, MockEngine};

    fn request(thinking: bool) -> EngineRequest {
        crate::engine::test_request("qwen", thinking)
    }

    fn call(name: &str, arguments: &str) -> String {
        format!("<tool_call>\n<function={name}>\n{arguments}</function>\n</tool_call>")
    }

    fn parameter(name: &str, value: &str) -> String {
        format!("<parameter={name}>\n{value}\n</parameter>\n")
    }

    #[test]
    fn the_adapter_emits_reasoning_content_and_native_tool_calls() {
        let engine = MockEngine::new("qwen")
            .content(&[
                "I should inspect the repository.",
                "\n</think>\n\n",
                "I'll look.\n\n",
                "<tool_call>\n<function=read>\n<parameter=path>\nsrc/lib.rs\n</parameter>\n",
                "</function>\n</tool_call>",
            ])
            .with_finish_reason(FinishReason::Stop);
        let engine = QwenResponseEngine::new(Arc::new(engine));
        let events =
            futures::executor::block_on(engine.generate(request(true)).collect::<Vec<_>>());
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut finish = None;
        for event in events {
            match event {
                EngineEvent::ReasoningDelta(text) => reasoning.push_str(&text),
                EngineEvent::ContentDelta(text) => content.push_str(&text),
                EngineEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                } => tool_calls.push((index, id, name, arguments_delta)),
                EngineEvent::Finished { finish_reason, .. } => finish = Some(finish_reason),
                _ => {}
            }
        }
        assert_eq!(reasoning, "I should inspect the repository.\n");
        assert_eq!(content, "I'll look.\n\n");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].0, 0);
        assert_eq!(tool_calls[0].2.as_deref(), Some("read"));
        assert_eq!(tool_calls[0].3, r#"{"path":"src/lib.rs"}"#);
        assert_eq!(finish, Some(FinishReason::ToolCalls));
    }

    #[test]
    fn the_adapter_leaves_a_non_thinking_answer_alone() {
        let engine = MockEngine::new("qwen")
            .content(&["The answer is 4."])
            .with_finish_reason(FinishReason::Stop);
        let engine = QwenResponseEngine::new(Arc::new(engine));
        let events =
            futures::executor::block_on(engine.generate(request(false)).collect::<Vec<_>>());
        let mut content = String::new();
        for event in events {
            if let EngineEvent::ContentDelta(text) = event {
                content.push_str(&text);
            }
        }
        assert_eq!(content, "The answer is 4.");
    }

    #[test]
    fn reasoning_is_split_before_the_visible_answer() {
        let text = "I should inspect the repository.\n</think>\n\nLet me look.";
        let parsed = parse_completion(text, true);
        assert_eq!(parsed.reasoning, "I should inspect the repository.\n");
        assert_eq!(parsed.content, "Let me look.");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn a_non_thinking_answer_is_all_content() {
        let parsed = parse_completion("The answer is 4.", false);
        assert!(parsed.reasoning.is_empty());
        assert_eq!(parsed.content, "The answer is 4.");
    }

    #[test]
    fn xml_tool_calls_are_lifted_out_of_the_content() {
        let block = call(
            "get_weather",
            &format!("{}{}", parameter("city", "Beijing"), parameter("days", "3")),
        );
        let text = format!("Let me check.\n\n{block}");
        let parsed = parse_completion(&text, false);
        assert_eq!(parsed.content, "Let me check.\n\n");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            r#"{"city":"Beijing","days":3}"#
        );
    }

    #[test]
    fn several_xml_calls_keep_their_order() {
        let text = format!(
            "{}\n{}",
            call("first", &parameter("a", "x")),
            call("second", &parameter("b", "true"))
        );
        let parsed = parse_completion(&text, false);
        let names: Vec<&str> = parsed
            .tool_calls
            .iter()
            .map(|call| call.name.as_str())
            .collect();
        assert_eq!(names, ["first", "second"]);
        assert_eq!(parsed.tool_calls[1].arguments, r#"{"b":true}"#);
    }

    #[test]
    fn a_json_style_call_is_also_accepted() {
        let text = r#"<tool_call>
{"name": "read", "arguments": {"path": "src/lib.rs"}}
</tool_call>"#;
        let parsed = parse_completion(text, false);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read");
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"path":"src/lib.rs"}"#);
    }

    #[test]
    fn an_unterminated_block_stays_visible() {
        let text = "<tool_call>\n<function=broken>\n";
        let parsed = parse_completion(text, false);
        assert_eq!(parsed.content, text);
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn every_split_boundary_reassembles_the_same_answer() {
        let text = format!(
            "reasoning first\n</think>\n\nI will check.\n\n{}",
            call(
                "search",
                &format!(
                    "{}{}",
                    parameter("query", "a \"quoted\" value"),
                    parameter("limit", "{\"nested\": [1, 2]}")
                )
            )
        );
        let want = parse_completion(&text, true);
        let mut boundaries: Vec<usize> = text.char_indices().map(|(index, _)| index).collect();
        boundaries.push(text.len());
        for split in boundaries {
            let mut parser = QwenStreamParser::new(true);
            let mut events = parser.feed(&text[..split]);
            events.extend(parser.feed(&text[split..]));
            events.extend(parser.finish());
            let mut got = ParsedQwenCompletion {
                reasoning: String::new(),
                content: String::new(),
                tool_calls: Vec::new(),
            };
            for event in events {
                match event {
                    QwenEvent::Reasoning(text) => got.reasoning.push_str(&text),
                    QwenEvent::Content(text) => got.content.push_str(&text),
                    QwenEvent::ArgumentFragment {
                        index,
                        name,
                        arguments,
                    } => {
                        assert_eq!(index, got.tool_calls.len());
                        got.tool_calls.push(ParsedQwenToolCall { name, arguments });
                    }
                }
            }
            assert_eq!(got, want, "split at byte {split}");
        }
    }
}
