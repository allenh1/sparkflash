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

//! The tool-call block framing every family's template shares.
//!
//! A GLM-5.3-Flash answer and a Qwen3.8-Flash-Next answer both wrap a call in
//! `<tool_call>...</tool_call>`; only the body differs (GLM writes a name and
//! `<arg_key>`/`<arg_value>` pairs, Qwen a `<function=...>` with `<parameter=...>` entries). The
//! framing is therefore shared - the markers, the hold-back that keeps a marker split across two
//! deltas from leaking as text, and the rule that an unterminated block stays visible - and each
//! family parses its own body.
//!
//! Nothing here knows a family's name or its argument syntax, and no arithmetic is involved: this
//! is the streaming seam between a raw content stream and the OpenAI `tool_calls` array.

/// The marker that opens a tool-call block.
pub const CALL_OPEN: &str = "<tool_call>";

/// The marker that closes one.
pub const CALL_CLOSE: &str = "</tool_call>";

/// A fragment of the content stream: visible text, or a complete tool-call body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    /// Visible answer text that is not inside a tool-call block.
    Text(String),
    /// The body of one call, between the markers and without them.
    Call(String),
}

/// Splits a streamed answer into visible text and complete tool-call bodies.
///
/// The parser holds back the bytes that could still become an opening marker, so a streamed answer
/// is never truncated by a marker split across two deltas, and a block that never closes (a length
/// stop) is handed back as text rather than dropped: the model's answer is worth more than the
/// call it did not finish.
#[derive(Debug)]
pub struct BlockSplitter {
    buffer: String,
}

impl Default for BlockSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockSplitter {
    /// A splitter for one answer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Feeds one content delta, returning the fragments that are now unambiguous.
    pub fn feed(&mut self, text: &str) -> Vec<Block> {
        if text.is_empty() {
            return Vec::new();
        }
        self.buffer.push_str(text);
        let mut blocks = Vec::new();
        loop {
            match self.buffer.find(CALL_OPEN) {
                None => {
                    let keep = longest_marker_suffix(&self.buffer, CALL_OPEN);
                    let split = self.buffer.len() - keep;
                    if split > 0 {
                        let text = self.buffer[..split].to_owned();
                        self.buffer.drain(..split);
                        blocks.push(Block::Text(text));
                    }
                    return blocks;
                }
                Some(index) => {
                    if index > 0 {
                        let text = self.buffer[..index].to_owned();
                        self.buffer.drain(..index);
                        blocks.push(Block::Text(text));
                    }
                    let body_start = CALL_OPEN.len();
                    let Some(relative_end) = self.buffer[body_start..].find(CALL_CLOSE) else {
                        // The block is still arriving; hold it so no partial body is emitted.
                        return blocks;
                    };
                    let body_end = body_start + relative_end;
                    let body = self.buffer[body_start..body_end].to_owned();
                    self.buffer.drain(..body_end + CALL_CLOSE.len());
                    blocks.push(Block::Call(body));
                }
            }
        }
    }

    /// Flushes the held-back tail at end of stream.
    ///
    /// A complete block cannot be waiting here - `feed` consumes every one it sees - so the tail is
    /// an unterminated block or the end of a marker, and both stay visible.
    pub fn finish(&mut self) -> Vec<Block> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        vec![Block::Text(std::mem::take(&mut self.buffer))]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(blocks: &[Block]) -> String {
        blocks
            .iter()
            .filter_map(|block| match block {
                Block::Text(text) => Some(text.as_str()),
                Block::Call(_) => None,
            })
            .collect()
    }

    fn calls(blocks: &[Block]) -> Vec<&str> {
        blocks
            .iter()
            .filter_map(|block| match block {
                Block::Call(body) => Some(body.as_str()),
                Block::Text(_) => None,
            })
            .collect()
    }

    #[test]
    fn a_plain_answer_is_all_text() {
        let mut splitter = BlockSplitter::new();
        let mut blocks = splitter.feed("The answer is 4.");
        blocks.extend(splitter.finish());
        assert_eq!(texts(&blocks), "The answer is 4.");
        assert!(calls(&blocks).is_empty());
    }

    #[test]
    fn a_block_is_lifted_out_and_the_text_around_it_kept() {
        let mut splitter = BlockSplitter::new();
        let mut blocks = splitter.feed("before<tool_call>g(1)</tool_call>after");
        blocks.extend(splitter.finish());
        assert_eq!(texts(&blocks), "beforeafter");
        assert_eq!(calls(&blocks), ["g(1)"]);
    }

    #[test]
    fn an_opening_marker_split_across_deltas_is_held() {
        let mut splitter = BlockSplitter::new();
        assert_eq!(texts(&splitter.feed("abc<tool_")), "abc");
        let blocks = splitter.feed("call>g()</tool_call>");
        assert_eq!(calls(&blocks), ["g()"]);
    }

    #[test]
    fn a_body_split_across_deltas_is_emitted_once() {
        let mut splitter = BlockSplitter::new();
        assert!(splitter.feed("<tool_call>g(").is_empty());
        assert!(splitter.feed("1, 2)").is_empty());
        let blocks = splitter.feed("</tool_call>");
        assert_eq!(calls(&blocks), ["g(1, 2)"]);
    }

    #[test]
    fn two_blocks_keep_their_order() {
        let mut splitter = BlockSplitter::new();
        let blocks = splitter.feed("<tool_call>a</tool_call>x<tool_call>b</tool_call>");
        assert_eq!(texts(&blocks), "x");
        assert_eq!(calls(&blocks), ["a", "b"]);
    }

    #[test]
    fn an_unterminated_block_stays_visible() {
        let mut splitter = BlockSplitter::new();
        let mut blocks = splitter.feed("answer<tool_call>g(");
        blocks.extend(splitter.finish());
        assert_eq!(texts(&blocks), "answer<tool_call>g(");
        assert!(calls(&blocks).is_empty());
    }

    #[test]
    fn every_split_boundary_reassembles_the_same_answer() {
        let answer = "a<tool_call>f(1)</tool_call>b<tool_call>g(2)</tool_call>c";
        for cut in 1..answer.len() {
            let (left, right) = answer.split_at(cut);
            let mut splitter = BlockSplitter::new();
            let mut blocks = splitter.feed(left);
            blocks.extend(splitter.feed(right));
            blocks.extend(splitter.finish());
            assert_eq!(texts(&blocks), "abc", "at {cut}");
            assert_eq!(calls(&blocks), ["f(1)", "g(2)"], "at {cut}");
        }
    }
}
