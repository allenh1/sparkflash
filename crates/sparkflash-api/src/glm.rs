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
//! The GLM tool-call XML is not parsed here yet; it is streamed as content, as the model page says.

use std::sync::Arc;

use futures::StreamExt;
use futures::stream::BoxStream;

use crate::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, InferenceEngine,
    MissingCapability,
};
use crate::telemetry_types::{DeepSeekSparkRequestStats, OpenAiUsageWithCache};

/// The marker that ends the GLM reasoning block.
const THINK_CLOSE: &str = "</think>";

/// One parsed GLM answer fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlmEvent {
    /// A fragment of the reasoning channel, before `</think>`.
    Reasoning(String),
    /// Visible answer text, after `</think>`.
    Content(String),
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
}

impl GlmStreamParser {
    /// A parser for one answer.
    #[must_use]
    pub fn new(thinking: bool) -> Self {
        Self {
            reasoning_done: !thinking,
            buffer: String::new(),
        }
    }

    /// Feeds one engine delta, returning the fragments that are now unambiguous.
    pub fn feed(&mut self, text: &str) -> Vec<GlmEvent> {
        if self.reasoning_done {
            return vec![GlmEvent::Content(text.to_owned())];
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
        if !content.is_empty() {
            events.push(GlmEvent::Content(content));
        }
        events
    }

    /// Flushes the held-back tail at end of stream.
    pub fn finish(&mut self) -> Vec<GlmEvent> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let text = std::mem::take(&mut self.buffer);
        if self.reasoning_done {
            vec![GlmEvent::Content(text)]
        } else {
            vec![GlmEvent::Reasoning(text)]
        }
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(events: Vec<GlmEvent>) -> (String, String) {
        let mut reasoning = String::new();
        let mut content = String::new();
        for event in events {
            match event {
                GlmEvent::Reasoning(text) => reasoning.push_str(&text),
                GlmEvent::Content(text) => content.push_str(&text),
            }
        }
        (reasoning, content)
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
