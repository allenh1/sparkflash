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

//! Inkling-2's output dialect: a model turn is a run of typed messages.
//!
//! The released template has no `<think>` block and no tool-call wrapper. A model turn is one or
//! more messages, each a role header, a content-type marker, a body and `<|end_message|>`:
//!
//! ```text
//! <|message_model|><|content_thinking|>reasoning<|end_message|>
//! <|message_model|><|content_text|>answer<|end_message|>
//! <|message_model|>lookup<|content_invoke_tool_json|>{"name":"lookup","args":{...}}<|end_message|>
//! <|content_model_end_sampling|>
//! ```
//!
//! The header may carry a name (a tool call's function), and the turn ends at
//! `<|content_model_end_sampling|>`, which the arm stops on before it reaches this layer. Every
//! marker is a special token, so the raw engine stream must keep them (the Inkling codec decodes
//! with special tokens) for this layer to see the structure at all; with them stripped a thinking
//! message and the answer arrive as one undifferentiated string.
//!
//! [`InklingStreamParser`] turns that stream into reasoning, content and tool calls; text outside
//! any message (a raw completion continuing plain text) is content. The parser holds back only
//! the bytes that could still become a marker, so a marker split across two deltas is never
//! emitted as text. A request with `skip_special_tokens: false` asked for the raw text, markers
//! included, and gets it unparsed.

use std::sync::Arc;

use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::Value;

use crate::engine::{
    next_tool_call_id, Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest,
    FinishReason, InferenceEngine, MissingCapability,
};
use crate::telemetry_types::{DeepSeekSparkRequestStats, OpenAiUsageWithCache};

const MESSAGE_MODEL: &str = "<|message_model|>";
const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_INVOKE_JSON: &str = "<|content_invoke_tool_json|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";

/// The content markers that open a body this layer shows as visible text.
const OTHER_CONTENT: &[&str] = &[
    "<|content_invoke_tool_text|>",
    "<|content_xml|>",
    "<|content_tool_error|>",
    "<|content_image|>",
];

/// Every marker the parser acts on, for the held-back tail.
const MARKERS: &[&str] = &[
    MESSAGE_MODEL,
    MESSAGE_USER,
    MESSAGE_SYSTEM,
    MESSAGE_TOOL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_JSON,
    END_MESSAGE,
    END_SAMPLING,
    "<|content_invoke_tool_text|>",
    "<|content_xml|>",
    "<|content_tool_error|>",
    "<|content_image|>",
];

/// One parsed Inkling answer fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InklingEvent {
    /// A fragment of a thinking message.
    Reasoning(String),
    /// Visible answer text.
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

/// Where in a turn the stream is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    /// Outside any message: a raw continuation, shown as content.
    Outside,
    /// After a role marker and before its content marker: the header's name.
    Header,
    /// A thinking body.
    Thinking,
    /// A text body.
    Text,
    /// A JSON tool invocation body, held until the message closes.
    Invoke,
    /// Any other body, shown as content.
    Other,
}

/// Parses one invocation body, `{"name": ..., "args": {...}}`, into the call's name and its
/// arguments as a JSON string.
///
/// The name in the body wins over the header's; a body that is not that object is not a call.
#[must_use]
pub fn parse_invocation(header: &str, body: &str) -> Option<(String, String)> {
    let value: Value = serde_json::from_str(body.trim()).ok()?;
    let object = value.as_object()?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_else(|| header.trim());
    if name.is_empty() {
        return None;
    }
    let arguments = match object.get("args") {
        None | Some(Value::Null) => "{}".to_owned(),
        Some(args @ Value::Object(_)) => serde_json::to_string(args).ok()?,
        Some(_) => return None,
    };
    Some((name.to_owned(), arguments))
}

/// The incremental Inkling answer parser.
#[derive(Debug)]
pub struct InklingStreamParser {
    section: Section,
    /// Bytes not yet attributed: at most a tail that could still become a marker.
    buffer: String,
    /// The current message's header name.
    header: String,
    /// The current invocation's body.
    invoke: String,
    calls: usize,
}

impl Default for InklingStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl InklingStreamParser {
    /// A parser for one answer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            section: Section::Outside,
            buffer: String::new(),
            header: String::new(),
            invoke: String::new(),
            calls: 0,
        }
    }

    /// How many complete tool calls the answer has emitted.
    #[must_use]
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Attributes plain text to the current section.
    fn text(&mut self, text: &str, events: &mut Vec<InklingEvent>) {
        if text.is_empty() {
            return;
        }
        match self.section {
            Section::Thinking => push(events, InklingEvent::Reasoning(text.to_owned())),
            Section::Outside | Section::Text | Section::Other => {
                push(events, InklingEvent::Content(text.to_owned()));
            }
            Section::Header => self.header.push_str(text),
            Section::Invoke => self.invoke.push_str(text),
        }
    }

    /// Applies one complete marker.
    fn marker(&mut self, marker: &str, events: &mut Vec<InklingEvent>) {
        match marker {
            MESSAGE_MODEL | MESSAGE_USER | MESSAGE_SYSTEM | MESSAGE_TOOL => {
                self.close_invoke(events);
                self.section = Section::Header;
                self.header.clear();
            }
            CONTENT_THINKING => self.section = Section::Thinking,
            CONTENT_TEXT => self.section = Section::Text,
            CONTENT_INVOKE_JSON => {
                self.section = Section::Invoke;
                self.invoke.clear();
            }
            END_MESSAGE | END_SAMPLING => {
                self.close_invoke(events);
                self.section = Section::Outside;
            }
            _ if OTHER_CONTENT.contains(&marker) => self.section = Section::Other,
            _ => {}
        }
    }

    /// Ends an invocation body: a call when it parses, visible text when it does not.
    fn close_invoke(&mut self, events: &mut Vec<InklingEvent>) {
        if self.section != Section::Invoke {
            return;
        }
        let body = std::mem::take(&mut self.invoke);
        match parse_invocation(&self.header, &body) {
            Some((name, arguments)) => {
                events.push(InklingEvent::ToolCall {
                    index: self.calls,
                    name,
                    arguments,
                });
                self.calls += 1;
            }
            // A body this arm cannot read stays visible: dropping it would lose the model's
            // answer, and inventing a call is worse.
            None => push(
                events,
                InklingEvent::Content(format!("{}{CONTENT_INVOKE_JSON}{body}", self.header)),
            ),
        }
        self.section = Section::Outside;
    }

    /// Feeds one engine delta, returning the fragments that are now unambiguous.
    pub fn feed(&mut self, text: &str) -> Vec<InklingEvent> {
        self.buffer.push_str(text);
        let mut events = Vec::new();
        loop {
            let Some(start) = self.buffer.find("<|") else {
                let pending = std::mem::take(&mut self.buffer);
                self.text(&pending, &mut events);
                break;
            };
            let before = self.buffer[..start].to_owned();
            self.text(&before, &mut events);
            self.buffer.drain(..start);
            if let Some(marker) = MARKERS
                .iter()
                .find(|marker| self.buffer.starts_with(**marker))
            {
                self.buffer.drain(..marker.len());
                self.marker(marker, &mut events);
                continue;
            }
            if MARKERS
                .iter()
                .any(|marker| marker.starts_with(self.buffer.as_str()))
            {
                // A marker may still be arriving: hold the tail for the next delta.
                break;
            }
            // `<|` that is not a marker is the model's text.
            self.buffer.drain(..2);
            self.text("<|", &mut events);
        }
        events
    }

    /// Flushes the held-back tail at end of stream.
    ///
    /// An invocation body that arrived whole is a call even when the closing `<|end_message|>`
    /// never reached this layer: the stream can end on the sampling stop right after the JSON.
    /// A body that does not parse stays visible, exactly as it does before a closing marker.
    pub fn finish(&mut self) -> Vec<InklingEvent> {
        let mut events = Vec::new();
        let pending = std::mem::take(&mut self.buffer);
        self.text(&pending, &mut events);
        self.close_invoke(&mut events);
        events
    }
}

/// Appends an event, merging it into the previous one of the same kind.
fn push(events: &mut Vec<InklingEvent>, event: InklingEvent) {
    match (events.last_mut(), event) {
        (Some(InklingEvent::Reasoning(last)), InklingEvent::Reasoning(text))
        | (Some(InklingEvent::Content(last)), InklingEvent::Content(text)) => last.push_str(&text),
        (_, event) => events.push(event),
    }
}

/// Wraps an engine that serves Inkling-2 and splits its typed messages into the API's channels.
pub struct InklingResponseEngine {
    inner: Arc<dyn InferenceEngine>,
}

impl InklingResponseEngine {
    /// Wraps an engine serving Inkling-2.
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

impl InferenceEngine for InklingResponseEngine {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        let raw = !request.skip_special_tokens;
        let inner = Arc::clone(&self.inner);
        if raw {
            return inner.generate(request);
        }
        Box::pin(async_stream::stream! {
            let mut parser = InklingStreamParser::new();
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
                            yield inkling_engine_event(parsed);
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
                yield inkling_engine_event(parsed);
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

fn inkling_engine_event(event: InklingEvent) -> EngineEvent {
    match event {
        InklingEvent::Reasoning(text) => EngineEvent::ReasoningDelta(text),
        InklingEvent::Content(text) => EngineEvent::ContentDelta(text),
        InklingEvent::ToolCall {
            index,
            name,
            arguments,
        } => EngineEvent::ToolCallDelta {
            index,
            id: Some(next_tool_call_id()),
            name: Some(name),
            arguments_delta: arguments,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(events: Vec<InklingEvent>) -> (String, String, Vec<(usize, String, String)>) {
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut calls = Vec::new();
        for event in events {
            match event {
                InklingEvent::Reasoning(text) => reasoning.push_str(&text),
                InklingEvent::Content(text) => content.push_str(&text),
                InklingEvent::ToolCall {
                    index,
                    name,
                    arguments,
                } => calls.push((index, name, arguments)),
            }
        }
        (reasoning, content, calls)
    }

    fn parse_all(pieces: &[&str]) -> (String, String, Vec<(usize, String, String)>) {
        let mut parser = InklingStreamParser::new();
        let mut events = Vec::new();
        for piece in pieces {
            events.extend(parser.feed(piece));
        }
        events.extend(parser.finish());
        collect(events)
    }

    #[test]
    fn a_thinking_message_and_a_text_message_split_into_their_channels() {
        let (reasoning, content, calls) = parse_all(&[
            "<|message_model|><|content_thinking|>The user asks for the capital.<|end_message|>",
            "<|message_model|><|content_text|>Paris<|end_message|>",
        ]);
        assert_eq!(reasoning, "The user asks for the capital.");
        assert_eq!(content, "Paris");
        assert!(calls.is_empty());
    }

    #[test]
    fn a_marker_split_across_deltas_is_never_text() {
        let (reasoning, content, _) = parse_all(&[
            "<|message_mo",
            "del|><|content_thi",
            "nking|>hm<|end_mes",
            "sage|><|message_model|><|content_text|>4<|end_message|>",
        ]);
        assert_eq!(reasoning, "hm");
        assert_eq!(content, "4");
    }

    #[test]
    fn an_invocation_becomes_a_tool_call() {
        let (_, content, calls) = parse_all(&[
            "<|message_model|><|content_text|>Looking.<|end_message|>",
            "<|message_model|>lookup<|content_invoke_tool_json|>",
            r#"{"name":"lookup","args":{"city":"Paris","n":2}}<|end_message|>"#,
        ]);
        assert_eq!(content, "Looking.");
        assert_eq!(
            calls,
            [(
                0,
                "lookup".to_owned(),
                r#"{"city":"Paris","n":2}"#.to_owned()
            )]
        );
    }

    #[test]
    fn an_invocation_complete_at_stream_end_is_still_a_call() {
        let (_, content, calls) = parse_all(&[
            "Noting the call first. run_code<|content_invoke_tool_json|>",
            r#"{"name":"run_code","args":{"code":"inspect(1)"}}"#,
        ]);
        assert_eq!(content, "Noting the call first. run_code");
        assert_eq!(
            calls,
            [(
                0,
                "run_code".to_owned(),
                r#"{"code":"inspect(1)"}"#.to_owned()
            )]
        );
    }

    #[test]
    fn an_unreadable_or_unterminated_invocation_stays_visible() {
        let (_, content, calls) =
            parse_all(&["<|message_model|>f<|content_invoke_tool_json|>not json<|end_message|>"]);
        assert!(calls.is_empty());
        assert_eq!(content, "f<|content_invoke_tool_json|>not json");
        let (_, content, calls) =
            parse_all(&["<|message_model|>f<|content_invoke_tool_json|>{\"name\":"]);
        assert!(calls.is_empty());
        assert_eq!(content, "f<|content_invoke_tool_json|>{\"name\":");
    }

    #[test]
    fn plain_continuation_text_is_content_and_a_literal_bracket_survives() {
        let (reasoning, content, _) = parse_all(&[" Paris. It is <|", "x|> a city"]);
        assert_eq!(reasoning, "");
        assert_eq!(content, " Paris. It is <|x|> a city");
    }

    #[test]
    fn a_stream_cut_inside_the_thinking_stays_reasoning() {
        let (reasoning, content, _) =
            parse_all(&["<|message_model|><|content_thinking|>still thinking"]);
        assert_eq!(reasoning, "still thinking");
        assert_eq!(content, "");
    }

    #[test]
    fn the_adapter_reports_a_call_as_the_finish_reason_and_raw_requests_pass_through() {
        use crate::engine::MockEngine;

        let pieces = [
            "<|message_model|><|content_thinking|>I should look.<|end_message|>",
            "<|message_model|>read<|content_invoke_tool_json|>",
            r#"{"name":"read","args":{"path":"src/lib.rs"}}<|end_message|>"#,
        ];
        let engine = MockEngine::new("inkling")
            .content(&pieces)
            .with_finish_reason(FinishReason::Stop);
        let engine = InklingResponseEngine::new(Arc::new(engine));
        let request = crate::engine::test_request("inkling", true);
        let events = futures::executor::block_on(engine.generate(request).collect::<Vec<_>>());
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        let mut finish = None;
        for event in events {
            match event {
                EngineEvent::ReasoningDelta(text) => reasoning.push_str(&text),
                EngineEvent::ToolCallDelta {
                    name,
                    arguments_delta,
                    ..
                } => calls.push((name, arguments_delta)),
                EngineEvent::Finished { finish_reason, .. } => finish = Some(finish_reason),
                _ => {}
            }
        }
        assert_eq!(reasoning, "I should look.");
        assert_eq!(
            calls,
            [(
                Some("read".to_owned()),
                r#"{"path":"src/lib.rs"}"#.to_owned()
            )]
        );
        assert_eq!(finish, Some(FinishReason::ToolCalls));

        let engine = MockEngine::new("inkling").content(&pieces);
        let engine = InklingResponseEngine::new(Arc::new(engine));
        let mut request = crate::engine::test_request("inkling", true);
        request.skip_special_tokens = false;
        let events = futures::executor::block_on(engine.generate(request).collect::<Vec<_>>());
        let raw: String = events
            .into_iter()
            .filter_map(|event| match event {
                EngineEvent::ContentDelta(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(raw, pieces.concat());
    }

    #[test]
    fn every_response_gets_a_fresh_tool_call_id() {
        use crate::engine::MockEngine;

        let pieces = [
            "<|message_model|>read<|content_invoke_tool_json|>",
            r#"{"name":"read","args":{"path":"src/lib.rs"}}<|end_message|>"#,
        ];
        let id = || {
            let engine = MockEngine::new("inkling")
                .content(&pieces)
                .with_finish_reason(FinishReason::Stop);
            let engine = InklingResponseEngine::new(Arc::new(engine));
            let request = crate::engine::test_request("inkling", true);
            let events = futures::executor::block_on(engine.generate(request).collect::<Vec<_>>());
            events
                .into_iter()
                .find_map(|event| match event {
                    EngineEvent::ToolCallDelta { id, .. } => id,
                    _ => None,
                })
                .expect("the adapter names its tool call")
        };
        let first = id();
        let second = id();
        assert!(
            first.starts_with("call_"),
            "ids keep the OpenAI prefix: {first}"
        );
        assert_ne!(
            first, second,
            "a later response must not reuse an earlier call id"
        );
    }
}
