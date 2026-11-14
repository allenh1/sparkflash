// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Inkling's prompt seam.
//!
//! The released template is a message/content pair, not ChatML:
//! `<|message_user|><|content_text|>...<|end_message|>`. It emits a thinking-effort system
//! message before the first non-system turn, and an assistant turn ends with
//! `<|content_model_end_sampling|>`.

use std::path::Path;

use serde_json::Value;
use sparkflash_model::{PromptCodec, PromptError};
use tokenizers::Tokenizer;

const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_IMAGE: &str = "<|content_image|>";
const CONTENT_INVOKE: &str = "<|content_invoke_tool_json|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";
const IMAGE_PLACEHOLDER: &str = "<|unused_200054|>";

/// A codec over the released Inkling tokenizer and template.
pub struct InklingCodec {
    tokenizer: Tokenizer,
    /// The beginning-of-sequence id, when the checkpoint names one.
    pub bos: Option<u32>,
    /// Every id that ends a generation.
    pub eos: Vec<u32>,
}

impl std::fmt::Debug for InklingCodec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InklingCodec")
            .field("vocabulary", &self.tokenizer.get_vocab_size(true))
            .finish()
    }
}

impl InklingCodec {
    /// Open the tokenizer at `path`.
    ///
    /// # Errors
    /// A missing or malformed tokenizer, or one that carries none of Inkling's message tokens.
    pub fn from_path(
        path: &Path,
        bos: Option<u32>,
        eos: Vec<u32>,
    ) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let codec = Self {
            tokenizer,
            bos,
            eos,
        };
        if codec.tokenizer.token_to_id(MESSAGE_USER).is_none()
            || codec.tokenizer.token_to_id(END_MESSAGE).is_none()
        {
            return Err(format!(
                "{} is not an Inkling tokenizer: no {MESSAGE_USER}/{END_MESSAGE} tokens",
                path.display()
            ));
        }
        Ok(codec)
    }

    /// Render a request through Inkling's released template.
    ///
    /// # Errors
    /// A request whose `messages` is not an array, or whose content is neither text nor a
    /// content-part array.
    pub fn render_chat(request: &Value) -> Result<String, PromptError> {
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
        let mut effort_emitted = false;
        if let Some(tools) = request.get("tools").and_then(Value::as_array)
            && !tools.is_empty() {
                text.push_str(MESSAGE_SYSTEM);
                text.push_str("tool_declare");
                text.push_str("<|content_xml|>");
                text.push_str(
                    &serde_json::to_string(tools)
                        .map_err(|error| PromptError::invalid(error.to_string(), None, None))?,
                );
                text.push_str(END_MESSAGE);
            }
        for message in messages {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    PromptError::invalid(
                        "every message needs a role",
                        Some("messages"),
                        Some("invalid_request_error"),
                    )
                })?;
            if !effort_emitted && role != "system" {
                text.push_str(MESSAGE_SYSTEM);
                text.push_str(CONTENT_TEXT);
                text.push_str("Thinking effort level: ");
                text.push_str(&effort(request.get("reasoning_effort"))?);
                text.push_str(END_MESSAGE);
                effort_emitted = true;
            }
            let role_token = match role {
                "user" => MESSAGE_USER,
                "assistant" => MESSAGE_MODEL,
                "system" => MESSAGE_SYSTEM,
                "tool" => MESSAGE_TOOL,
                other => {
                    return Err(PromptError::invalid(
                        format!("Unknown message role: {other}"),
                        Some("messages"),
                        Some("invalid_request_error"),
                    ));
                }
            };
            let mut header = role_token.to_owned();
            if matches!(role, "tool" | "assistant")
                && let Some(name) = message.get("name").and_then(Value::as_str)
            {
                header.push_str(name);
            }
            if role == "assistant"
                && let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str)
                && !reasoning.is_empty()
            {
                text.push_str(&header);
                text.push_str(CONTENT_THINKING);
                text.push_str(reasoning);
                text.push_str(END_MESSAGE);
            }
            if let Some(content) = message.get("content") {
                text.push_str(&header);
                match content {
                    Value::String(value) => {
                        text.push_str(CONTENT_TEXT);
                        text.push_str(value);
                        text.push_str(END_MESSAGE);
                    }
                    Value::Array(parts) => {
                        for part in parts {
                            match part.get("type").and_then(Value::as_str) {
                                Some("text" | "input_text") | None => {
                                    text.push_str(CONTENT_TEXT);
                                    text.push_str(
                                        part.get("text").and_then(Value::as_str).unwrap_or(""),
                                    );
                                    text.push_str(END_MESSAGE);
                                }
                                Some("image" | "input_image" | "image_url") => {
                                    text.push_str(CONTENT_IMAGE);
                                    text.push_str(IMAGE_PLACEHOLDER);
                                    text.push_str(END_MESSAGE);
                                }
                                Some(other) => {
                                    return Err(PromptError::invalid(
                                        format!("Unsupported content part type: {other}"),
                                        Some("messages"),
                                        Some("invalid_request_error"),
                                    ));
                                }
                            }
                        }
                    }
                    Value::Null => {}
                    _ => {
                        return Err(PromptError::invalid(
                            "message content must be text or an array of parts",
                            Some("messages"),
                            Some("invalid_request_error"),
                        ));
                    }
                }
            }
            if role == "assistant" {
                if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        let function = call.get("function").unwrap_or(&Value::Null);
                        let name = function.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = function.get("arguments").cloned().unwrap_or(Value::Null);
                        text.push_str(MESSAGE_MODEL);
                        text.push_str(name);
                        text.push_str(CONTENT_INVOKE);
                        text.push_str("{\"name\":");
                        text.push_str(&serde_json::to_string(name).map_err(|error| {
                            PromptError::invalid(error.to_string(), None, None)
                        })?);
                        text.push_str(",\"args\":");
                        text.push_str(&serde_json::to_string(&args).map_err(|error| {
                            PromptError::invalid(error.to_string(), None, None)
                        })?);
                        text.push('}');
                        text.push_str(END_MESSAGE);
                    }
                }
                text.push_str(END_SAMPLING);
            }
        }
        if !effort_emitted {
            text.push_str(MESSAGE_SYSTEM);
            text.push_str(CONTENT_TEXT);
            text.push_str("Thinking effort level: ");
            text.push_str(&effort(request.get("reasoning_effort"))?);
            text.push_str(END_MESSAGE);
        }
        Ok(text)
    }

    fn encode_ids(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|error| PromptError::invalid(error.to_string(), None, None))?;
        Ok(encoding.get_ids().to_vec())
    }
}

impl PromptCodec for InklingCodec {
    fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, PromptError> {
        let text = Self::render_chat(request)?;
        self.encode_ids(&text, false)
    }

    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        self.encode_ids(text, add_special_tokens)
    }

    fn decode(&self, ids: &[u32]) -> Option<String> {
        self.tokenizer.decode(ids, true).ok()
    }

    fn token_str(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

fn effort(value: Option<&Value>) -> Result<String, PromptError> {
    let value = value.unwrap_or(&Value::Null);
    let number = match value {
        Value::Null => 0.9,
        Value::String(name) => match name.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            other => {
                return Err(PromptError::invalid(
                    format!("Unknown reasoning_effort: {other}"),
                    Some("reasoning_effort"),
                    Some("invalid_request_error"),
                ));
            }
        },
        Value::Number(number) => number.as_f64().unwrap_or(0.9),
        _ => {
            return Err(PromptError::invalid(
                "reasoning_effort must be a string or a number",
                Some("reasoning_effort"),
                Some("invalid_request_error"),
            ));
        }
    };
    if !(0.0..=0.99).contains(&number) {
        return Err(PromptError::invalid(
            "reasoning_effort must be in [0.0, 0.99]",
            Some("reasoning_effort"),
            Some("invalid_request_error"),
        ));
    }
    Ok(if number == 0.0 {
        "0".to_owned()
    } else {
        number.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_template_emits_effort_and_message_pairs() {
        let text = InklingCodec::render_chat(&json!({
            "messages": [{"role":"user","content":"hello"}]
        }))
        .expect("the template renders");
        assert_eq!(
            text,
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>hello<|end_message|>"
        );
    }

    #[test]
    fn an_assistant_turn_closes_sampling() {
        let text = InklingCodec::render_chat(&json!({
            "messages": [
                {"role":"user","content":"hi"},
                {"role":"assistant","content":"there"}
            ]
        }))
        .expect("the template renders");
        assert!(text.ends_with(
            "<|message_model|><|content_text|>there<|end_message|><|content_model_end_sampling|>"
        ));
    }
}
