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

//! GLM-5.3-Flash's output dialect: the `<think>` reasoning block.
//!
//! The released template opens the generation prompt with `<think>`, so the raw engine stream
//! carries the model's reasoning and its answer as one content stream, separated only by the
//! closing `</think>` marker. [`GlmResponseEngine`] splits that stream into
//! [`EngineEvent::ReasoningDelta`] and [`EngineEvent::ContentDelta`], which is the shape every
//! API surface expects (and the shape the Qwen arm already produces). The parser holds back only
//! the bytes that could still become the closing marker, so a streamed answer is never truncated
//! by a marker split across two deltas.
//!
//! The released template wraps a call in `<tool_call>` with a `{name}<arg_key>k</arg_key>
//! <arg_value>v</arg_value>...</tool_call>` body, which this layer lifts out of the content stream
//! into [EngineEvent::ToolCallDelta]. The block framing is the shared
//! [crate::tool_block::BlockSplitter]'s; only the body is GLM's.

use std::sync::Arc;

use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Map, Value};

use crate::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, InferenceEngine,
    MissingCapability,
};
use crate::telemetry_types::{DeepSeekSparkRequestStats, OpenAiUsageWithCache};
use crate::tool_block::{Block, BlockSplitter, CALL_CLOSE, CALL_OPEN};

/// The marker that ends the GLM reasoning block.
const THINK_CLOSE: &str = "</think>";

/// The marker that opens one argument's key.
const ARG_KEY_OPEN: &str = "<arg_key>";
/// The marker that closes one argument's key.
const ARG_KEY_CLOSE: &str = "</arg_key>";
/// The marker that opens one argument's value.
const ARG_VALUE_OPEN: &str = "<arg_value>";
/// The marker that closes one argument's value.
const ARG_VALUE_CLOSE: &str = "</arg_value>";

/// One parsed GLM answer fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlmEvent {
    /// A fragment of the reasoning channel, before `</think>`.
    Reasoning(String),
    /// Visible answer text, after `</think>`.
    Content(String),
    /// A complete tool call, with its arguments as the OpenAI JSON string.
    ToolCall {
        /// Position of this call among the answer's calls.
        index: usize,
        /// The function the model named.
        name: String,
        /// The arguments object, serialised as a JSON string.
        arguments: String,
    },
}

/// One tool call's name and arguments, parsed out of the template's body.
///
/// The body is the function's name followed by `<arg_key>`/`<arg_value>` pairs, which is what
/// the released template renders (`render_tool_call` in the model pack). A value is read as JSON
/// when it is one and as a string when it is not, which is the reading the released parser uses;
/// a bare `true`, `5` or `{"a":1}` therefore keeps its type, and a bare word does not.
///
/// `None` means the body is not a call this arm can spell: an empty name, or a pair whose key or
/// value markers do not close.
#[must_use]
pub fn parse_tool_call(body: &str) -> Option<(String, String)> {
    let body = body.trim();
    let (name, rest) = match body.find(ARG_KEY_OPEN) {
        Some(index) => (&body[..index], &body[index..]),
        None => (body, ""),
    };
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let mut arguments = Map::new();
    let mut rest = rest;
    while !rest.is_empty() {
        let rest_after_key = rest.strip_prefix(ARG_KEY_OPEN)?;
        let (key, rest_after_key) = rest_after_key.split_once(ARG_VALUE_OPEN)?;
        let key = key.strip_suffix(ARG_KEY_CLOSE)?;
        let (value, rest_after_value) = rest_after_key.split_once(ARG_VALUE_CLOSE)?;
        let value = match serde_json::from_str::<Value>(value) {
            Ok(Value::String(text)) => Value::String(text),
            Ok(json) => json,
            Err(_) => Value::String(value.to_owned()),
        };
        arguments.insert(key.to_owned(), value);
        rest = rest_after_value.trim_start();
    }
    let arguments = serde_json::to_string(&Value::Object(arguments)).ok()?;
    Some((name.to_owned(), arguments))
}

/// The incremental GLM answer parser.
///
/// A parser created with `thinking = false` treats the whole stream as visible content: the
/// request asked the API to close the thinking block in the prompt, so anything the model emits is
/// the answer. With `thinking = true` everything before `</think>` is reasoning and everything
/// after it is content; a stream that ends before the marker (a length stop mid-thought) is
/// reported as reasoning.
#[derive(Debug)]
pub struct GlmStreamParser {
    reasoning_done: bool,
    buffer: String,
    /// The content stream after the reasoning block, split into text and tool-call bodies.
    blocks: BlockSplitter,
    /// How many complete calls the answer has emitted.
    calls: usize,
}

impl GlmStreamParser {
    /// A parser for one answer.
    #[must_use]
    pub fn new(thinking: bool) -> Self {
        Self {
            reasoning_done: !thinking,
            buffer: String::new(),
            blocks: BlockSplitter::new(),
            calls: 0,
        }
    }

    /// How many complete tool calls the answer has emitted.
    #[must_use]
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Splits content into text and tool calls, and counts the calls.
    fn advance(&mut self, text: &str) -> Vec<GlmEvent> {
        let mut events = Vec::new();
        for block in self.blocks.feed(text) {
            match block {
                Block::Text(text) => events.push(GlmEvent::Content(text)),
                Block::Call(body) => match parse_tool_call(&body) {
                    Some((name, arguments)) => {
                        events.push(GlmEvent::ToolCall {
                            index: self.calls,
                            name,
                            arguments,
                        });
                        self.calls += 1;
                    }
                    // A block this arm cannot spell stays visible: dropping it would lose the
                    // model's answer, and inventing a call is worse.
                    None => {
                        events.push(GlmEvent::Content(format!("{CALL_OPEN}{body}{CALL_CLOSE}")))
                    }
                },
            }
        }
        events
    }

    /// Feeds one engine delta, returning the fragments that are now unambiguous.
    pub fn feed(&mut self, text: &str) -> Vec<GlmEvent> {
        if self.reasoning_done {
            return self.advance(text);
        }
        self.buffer.push_str(text);
        let Some(index) = self.buffer.find(THINK_CLOSE) else {
            let keep = longest_marker_suffix(&self.buffer, THINK_CLOSE);
            let split = self.buffer.len() - keep;
            if split == 0 {
                return Vec::new();
            }
            let reasoning = self.buffer[..split].to_owned();
            self.buffer.drain(..split);
            return vec![GlmEvent::Reasoning(reasoning)];
        };
        let mut events = Vec::new();
        if index > 0 {
            events.push(GlmEvent::Reasoning(self.buffer[..index].to_owned()));
        }
        let content = self.buffer[index + THINK_CLOSE.len()..].to_owned();
        self.buffer.clear();
        self.reasoning_done = true;
        events.extend(self.advance(&content));
        events
    }

    /// Flushes the held-back tail at end of stream.
    pub fn finish(&mut self) -> Vec<GlmEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let text = std::mem::take(&mut self.buffer);
            if self.reasoning_done {
                events.extend(self.advance(&text));
            } else {
                events.push(GlmEvent::Reasoning(text));
            }
        }
        // A block that never closed stays visible, as the shared splitter reads it.
        for block in self.blocks.finish() {
            match block {
                Block::Text(text) => events.push(GlmEvent::Content(text)),
                Block::Call(body) => {
                    events.push(GlmEvent::Content(format!("{CALL_OPEN}{body}{CALL_CLOSE}")))
                }
            }
        }
        events
    }

    /// Whether the closing marker has been seen.
    #[must_use]
    pub fn closed(&self) -> bool {
        self.reasoning_done
    }
}

/// The length of the longest suffix of `text` that is a proper prefix of `marker`.
fn longest_marker_suffix(text: &str, marker: &str) -> usize {
    let limit = text.len().min(marker.len() - 1);
    (1..=limit)
        .rev()
        .find(|&length| text.ends_with(&marker[..length]))
        .unwrap_or(0)
}

/// Wraps an engine that serves GLM-5.3-Flash and splits its `<think>` block out of the content
/// stream.
pub struct GlmResponseEngine {
    inner: Arc<dyn InferenceEngine>,
}

impl GlmResponseEngine {
    /// Wraps an engine serving GLM-5.3-Flash.
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

impl InferenceEngine for GlmResponseEngine {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        let thinking = request.thinking;
        let inner = Arc::clone(&self.inner);
        Box::pin(async_stream::stream! {
            let mut parser = GlmStreamParser::new(thinking);
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
                            yield glm_engine_event(parsed);
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
                yield glm_engine_event(parsed);
            }
            if let Some((finish_reason, usage, stats)) = finished {
                // A finish reason of `stop` beside a parsed call is what every client reads as
                // "the answer ended"; the call is what actually ended it.
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

fn glm_engine_event(event: GlmEvent) -> EngineEvent {
    match event {
        GlmEvent::Reasoning(text) => EngineEvent::ReasoningDelta(text),
        GlmEvent::Content(text) => EngineEvent::ContentDelta(text),
        GlmEvent::ToolCall {
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

    fn collect(events: Vec<GlmEvent>) -> (String, String) {
        let (reasoning, content, _) = collect_calls(events);
        (reasoning, content)
    }

    /// The reasoning, the visible text and each call's `(index, name, arguments)`.
    fn collect_calls(events: Vec<GlmEvent>) -> (String, String, Vec<(usize, String, String)>) {
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut calls = Vec::new();
        for event in events {
            match event {
                GlmEvent::Reasoning(text) => reasoning.push_str(&text),
                GlmEvent::Content(text) => content.push_str(&text),
                GlmEvent::ToolCall {
                    index,
                    name,
                    arguments,
                } => calls.push((index, name, arguments)),
            }
        }
        (reasoning, content, calls)
    }

    #[test]
    fn a_tool_call_is_lifted_out_of_the_content() {
        let mut parser = GlmStreamParser::new(true);
        let mut events = parser.feed("thinking</think>looking it up");
        events.extend(parser.feed(
            "<tool_call>lookup<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        ));
        events.extend(parser.finish());
        let (reasoning, content, calls) = collect_calls(events);
        assert_eq!(reasoning, "thinking");
        assert_eq!(content, "looking it up");
        assert_eq!(
            calls,
            [(0, "lookup".to_owned(), r#"{"city":"Paris"}"#.to_owned())]
        );
        assert_eq!(parser.calls(), 1);
    }

    #[test]
    fn several_calls_keep_their_order_and_their_types() {
        let mut parser = GlmStreamParser::new(false);
        let mut events = parser.feed(
            "<tool_call>a<arg_key>n</arg_key><arg_value>5</arg_value>\
             <arg_key>on</arg_key><arg_value>true</arg_value>\
             <arg_key>rest</arg_key><arg_value>{\"k\":1}</arg_value></tool_call>",
        );
        events.extend(
            parser.feed(
                "<tool_call>b<arg_key>text</arg_key><arg_value>a word</arg_value></tool_call>",
            ),
        );
        let (_, content, calls) = collect_calls(events);
        assert_eq!(content, "");
        assert_eq!(
            calls,
            [
                (
                    0,
                    "a".to_owned(),
                    r#"{"n":5,"on":true,"rest":{"k":1}}"#.to_owned()
                ),
                (1, "b".to_owned(), r#"{"text":"a word"}"#.to_owned()),
            ]
        );
    }

    #[test]
    fn a_call_named_by_the_template_parses_to_its_name_and_no_arguments() {
        assert_eq!(
            parse_tool_call("ping"),
            Some(("ping".to_owned(), "{}".to_owned()))
        );
        assert_eq!(parse_tool_call("   "), None);
        assert_eq!(parse_tool_call("<arg_key>k</arg_key>"), None);
        // A pair whose value never closes is not a call.
        assert_eq!(parse_tool_call("f<arg_key>k</arg_key><arg_value>v"), None);
    }

    #[test]
    fn the_adapter_emits_a_native_tool_call_and_reports_the_cause() {
        use crate::engine::MockEngine;

        let engine = MockEngine::new("glm")
            // The released template writes `thinking{reasoning}</think>{content}` with nothing
            // between the marker and the answer, so the mock does too.
            .content(&[
                "I should look.",
                "</think>",
                "Looking.\n\n<tool_call>read<arg_key>path</arg_key>",
                "<arg_value>src/lib.rs</arg_value></tool_call>",
            ])
            .with_finish_reason(FinishReason::Stop);
        let engine = GlmResponseEngine::new(Arc::new(engine));
        let request = crate::engine::test_request("glm", true);
        let events = futures::executor::block_on(engine.generate(request).collect::<Vec<_>>());
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut calls = Vec::new();
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
                } => calls.push((index, id, name, arguments_delta)),
                EngineEvent::Finished { finish_reason, .. } => finish = Some(finish_reason),
                _ => {}
            }
        }
        assert_eq!(reasoning, "I should look.");
        assert_eq!(content, "Looking.\n\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[0].1.as_deref(), Some("call_0"));
        assert_eq!(calls[0].2.as_deref(), Some("read"));
        assert_eq!(calls[0].3, r#"{"path":"src/lib.rs"}"#);
        assert_eq!(finish, Some(FinishReason::ToolCalls));
    }

    #[test]
    fn an_unterminated_call_stays_visible() {
        let mut parser = GlmStreamParser::new(false);
        let mut events = parser.feed("answer<tool_call>g(");
        events.extend(parser.finish());
        let (_, content, calls) = collect_calls(events);
        assert_eq!(content, "answer<tool_call>g(");
        assert!(calls.is_empty());
    }

    #[test]
    fn a_marker_split_across_deltas_is_not_emitted_as_content() {
        let mut parser = GlmStreamParser::new(true);
        let mut events = parser.feed("The answer is 4.");
        events.extend(parser.feed("</thi"));
        events.extend(parser.feed("nk>"));
        events.extend(parser.feed("4"));
        events.extend(parser.finish());
        assert_eq!(
            collect(events),
            ("The answer is 4.".to_owned(), "4".to_owned())
        );
    }

    #[test]
    fn a_stream_that_ends_inside_the_reasoning_stays_reasoning() {
        let mut parser = GlmStreamParser::new(true);
        let mut events = parser.feed("still thinking");
        events.extend(parser.finish());
        assert_eq!(
            collect(events),
            ("still thinking".to_owned(), String::new())
        );
    }

    #[test]
    fn a_non_thinking_request_is_all_content() {
        let mut parser = GlmStreamParser::new(false);
        let mut events = parser.feed("</think>the answer");
        events.extend(parser.finish());
        assert_eq!(
            collect(events),
            (String::new(), "</think>the answer".to_owned())
        );
    }

    #[test]
    fn the_content_after_the_marker_is_one_fragment() {
        let mut parser = GlmStreamParser::new(true);
        events_are(&mut parser, "reason</think>answer");
        assert!(parser.closed());
    }

    fn events_are(parser: &mut GlmStreamParser, text: &str) {
        let (reasoning, content) = collect(parser.feed(text));
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "answer");
    }

    #[test]
    fn a_partial_marker_at_the_end_is_held_until_the_next_delta() {
        let mut parser = GlmStreamParser::new(true);
        assert_eq!(
            collect(parser.feed("abc</thi")),
            ("abc".to_owned(), String::new())
        );
        assert_eq!(
            collect(parser.feed("nk>def")),
            (String::new(), "def".to_owned())
        );
    }
}
