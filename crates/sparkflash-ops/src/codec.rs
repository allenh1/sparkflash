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

//! The byte-level prompt codec the tiny fixture arms serve through.
//!
//! It is not a tokenizer and says so in its name: it maps text bytes and token ids onto a
//! fixture vocabulary so the router, queue, streaming and sampling path can be exercised without
//! a released checkpoint.

use serde_json::Value;
use sparkflash_model::{PromptCodec, PromptError};

/// A codec for a tiny fixture vocabulary.
#[derive(Debug, Clone, Copy)]
pub struct ByteCodec {
    vocabulary: u32,
}

impl ByteCodec {
    /// A codec for a vocabulary of this size (at least 3).
    ///
    /// # Panics
    /// When the vocabulary is smaller than three tokens.
    #[must_use]
    pub fn new(vocabulary: usize) -> Self {
        assert!(vocabulary >= 3, "a byte codec needs at least three tokens");
        Self {
            vocabulary: vocabulary as u32,
        }
    }

    fn id_for(byte: u8, vocabulary: u32) -> u32 {
        let span = vocabulary - 3;
        3 + u32::from(byte) % span
    }
}

impl PromptCodec for ByteCodec {
    fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, PromptError> {
        let messages = request
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                PromptError::invalid(
                    "messages must be an array",
                    Some("messages"),
                    Some("invalid_request_error"),
                )
            })?;
        let mut text = String::new();
        for message in messages {
            if let Some(content) = message.get("content").and_then(Value::as_str) {
                text.push_str(content);
            }
        }
        self.encode_text(&text, true)
    }

    fn encode_text(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        let mut ids: Vec<u32> = text
            .bytes()
            .map(|byte| Self::id_for(byte, self.vocabulary))
            .collect();
        if ids.is_empty() {
            ids.push(3);
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Option<String> {
        Some(
            ids.iter()
                .map(|id| char::from((*id % 26) as u8 + b'a'))
                .collect(),
        )
    }

    fn token_str(&self, id: u32) -> Option<String> {
        Some(char::from((id % 26) as u8 + b'a').to_string())
    }
}
