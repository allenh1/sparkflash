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

//! The prompt-codec contract.
//!
//! A model pack owns its tokenizer and chat template. The protocol crate only needs four
//! operations: render a chat request to token ids, tokenize text, decode ids to text, and name one
//! token. This trait keeps the concrete tokenizer and the V4.1 rendering rules in the pack while
//! letting the HTTP surface stay model-agnostic.

use std::fmt;
use std::sync::Arc;

use serde_json::Value;

/// A prompt-codec refusal.
///
/// The optional parameter and code are the OpenAI-shaped fields the HTTP layer reports. They are
/// data rather than an HTTP dependency so the contract stays usable by a non-HTTP caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptError {
    message: String,
    param: Option<&'static str>,
    code: Option<&'static str>,
    internal: bool,
}

impl PromptError {
    /// A request-shaped refusal.
    #[must_use]
    pub fn invalid(
        message: impl Into<String>,
        param: Option<&'static str>,
        code: Option<&'static str>,
    ) -> Self {
        Self {
            message: message.into(),
            param,
            code,
            internal: false,
        }
    }

    /// A server-side failure, such as a tokenizer that could not encode the prompt.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: None,
            code: None,
            internal: true,
        }
    }

    /// The human-readable refusal.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The request field the refusal is about, when one applies.
    #[must_use]
    pub fn param(&self) -> Option<&'static str> {
        self.param
    }

    /// The stable machine-readable code, when one applies.
    #[must_use]
    pub fn code(&self) -> Option<&'static str> {
        self.code
    }

    /// Whether the failure is on the server rather than in the request.
    #[must_use]
    pub fn is_internal(&self) -> bool {
        self.internal
    }
}

impl fmt::Display for PromptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PromptError {}

/// What the HTTP surface needs from a model's prompt handling.
pub trait PromptCodec: Send + Sync {
    /// Render a chat-completion request body and tokenize it.
    ///
    /// The argument is the raw JSON body because a chat template may depend on fields the typed
    /// request view discards, such as arbitrary chat_template_kwargs.
    ///
    /// # Errors
    /// A PromptError when the request does not match the model's template or the tokenizer
    /// fails.
    fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, PromptError>;

    /// Tokenize text with the caller's add_special_tokens choice.
    ///
    /// # Errors
    /// A PromptError when the tokenizer fails.
    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError>;

    /// Decode ids back to text, or None when the tokenizer cannot.
    fn decode(&self, ids: &[u32]) -> Option<String>;

    /// The string spelling of one vocabulary id, or None when it is outside the vocabulary.
    fn token_str(&self, id: u32) -> Option<String>;

    /// The model's image front end, when it has vision.
    ///
    /// This lives on the prompt codec because both are model-owned front ends selected together
    /// at startup; a model without vision returns None.
    fn vision_codec(&self) -> Option<Arc<dyn crate::vision::VisionCodec>> {
        None
    }
}
