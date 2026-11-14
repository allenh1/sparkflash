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

use std::sync::Arc;

use serde_json::Value;
use sparkflash_model::{PromptCodec, PromptError};
use tokenizers::Tokenizer;

use crate::config::InklingVisionConfig;
use crate::vision::InklingVisionCodec;

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
    /// The image front end, when the checkpoint has a tower ([Self::with_vision]).
    vision: Option<InklingVisionCodec>,
}

impl std::fmt::Debug for InklingCodec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InklingCodec")
            .field("vocabulary", &self.tokenizer.get_vocab_size(true))
            .field("vision", &self.vision.is_some())
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
            vision: None,
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

    /// The codec with the image front end for `config`'s tower: the template's
    /// `<|content_image|>`, placeholder and `<|end_message|>` ids come from this tokenizer.
    ///
    /// # Errors
    /// A tokenizer without one of the three, or whose placeholder is not the config's image token.
    pub fn with_vision(mut self, config: InklingVisionConfig) -> Result<Self, String> {
        let id = |token: &str| {
            self.tokenizer
                .token_to_id(token)
                .ok_or_else(|| format!("the tokenizer has no {token}, which images need"))
        };
        let (open, placeholder, close) = (id(CONTENT_IMAGE)?, id(IMAGE_PLACEHOLDER)?, id(END_MESSAGE)?);
        if placeholder != crate::vision::IMAGE_TOKEN_ID {
            return Err(format!(
                "the tokenizer's {IMAGE_PLACEHOLDER} is id {placeholder}, not the image token {}",
                crate::vision::IMAGE_TOKEN_ID
            ));
        }
        self.vision = Some(InklingVisionCodec::new(config, placeholder, open, close));
        Ok(self)
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
                text.push_str(&tool_specs(tools)?);
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
                && let Some(name) = message
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
            {
                header.push_str(name);
            } else if role == "tool"
                && let Some(id) = message.get("tool_call_id").and_then(Value::as_str)
                && let Some(name) = answered_call(messages, id)
            {
                // A tool result without a name is headed by the call it answers, as the
                // released template looks it up.
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
            // The released template writes an empty assistant text only when the turn has nothing
            // else: a turn that is all reasoning or all tool calls carries no text message.
            let carries_more = role == "assistant"
                && (message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .is_some_and(|calls| !calls.is_empty())
                    || message
                        .get("reasoning_content")
                        .and_then(Value::as_str)
                        .is_some_and(|reasoning| !reasoning.is_empty()));
            if let Some(content) = message.get("content") {
                match content {
                    Value::String(value) if value.is_empty() && carries_more => {}
                    Value::String(value) => {
                        text.push_str(&header);
                        text.push_str(CONTENT_TEXT);
                        text.push_str(value);
                        text.push_str(END_MESSAGE);
                    }
                    Value::Array(parts) => {
                        // Every part is its own message, each with the turn's header.
                        for part in parts {
                            match part.get("type").and_then(Value::as_str) {
                                Some("text" | "input_text") | None => {
                                    text.push_str(&header);
                                    text.push_str(CONTENT_TEXT);
                                    text.push_str(
                                        part.get("text").and_then(Value::as_str).unwrap_or(""),
                                    );
                                    text.push_str(END_MESSAGE);
                                }
                                Some("image" | "input_image" | "image_url") => {
                                    if !matches!(role, "user" | "tool") {
                                        return Err(PromptError::invalid(
                                            "Image content is only supported for user and tool \
                                             messages",
                                            Some("messages"),
                                            Some("invalid_request_error"),
                                        ));
                                    }
                                    text.push_str(&header);
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
                        text.push_str(&invocation(call)?);
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

    /// The generated text with its message markers kept.
    ///
    /// Every structural marker of the dialect is a special token, so a decode that skipped them
    /// would hand the server one string in which the thinking message and the answer are
    /// indistinguishable. The served path's Inkling response layer
    /// (`sparkflash_api::inkling`) reads the markers and removes them.
    fn decode(&self, ids: &[u32]) -> Option<String> {
        self.tokenizer.decode(ids, false).ok()
    }

    fn token_str(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    fn vision_codec(&self) -> Option<Arc<dyn sparkflash_model::vision::VisionCodec>> {
        self.vision
            .clone()
            .map(|vision| Arc::new(vision) as Arc<dyn sparkflash_model::vision::VisionCodec>)
    }
}

/// The function name of the assistant tool call whose id is `id`, the last one when several
/// match (the template's loop overwrites its header for each).
fn answered_call<'a>(messages: &'a [Value], id: &str) -> Option<&'a str> {
    messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
        .filter_map(|message| message.get("tool_calls").and_then(Value::as_array))
        .flatten()
        .filter(|call| call.get("id").and_then(Value::as_str) == Some(id))
        .filter_map(|call| call.get("function")?.get("name")?.as_str())
        .next_back()
}

fn invalid(message: impl Into<String>) -> PromptError {
    PromptError::invalid(message.into(), Some("messages"), Some("invalid_request_error"))
}

fn json(value: &Value) -> Result<String, PromptError> {
    serde_json::to_string(value).map_err(|error| PromptError::invalid(error.to_string(), None, None))
}

/// `value` with every object's keys in sorted order, which is what the template's
/// `tojson(sort_keys=true)` writes. The workspace's `serde_json` keeps insertion order, so the
/// sort is explicit.
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|key| (key.clone(), sorted(&map[key]))).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// The tool declaration body: one `{description, name, parameters, type}` spec a tool, keys
/// sorted at every depth, compact separators - the released template's `tool_declare` message.
fn tool_specs(tools: &[Value]) -> Result<String, PromptError> {
    let specs: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            let truthy = |value: Option<&Value>| {
                value
                    .filter(|value| match value {
                        Value::Null | Value::Bool(false) => false,
                        Value::String(text) => !text.is_empty(),
                        Value::Object(map) => !map.is_empty(),
                        Value::Array(items) => !items.is_empty(),
                        _ => true,
                    })
                    .cloned()
            };
            let mut spec = serde_json::Map::new();
            spec.insert(
                "description".to_owned(),
                truthy(function.get("description")).unwrap_or_else(|| Value::String(String::new())),
            );
            spec.insert(
                "name".to_owned(),
                function.get("name").cloned().unwrap_or(Value::Null),
            );
            spec.insert(
                "parameters".to_owned(),
                truthy(function.get("parameters"))
                    .unwrap_or_else(|| Value::Object(serde_json::Map::new())),
            );
            spec.insert(
                "type".to_owned(),
                truthy(tool.get("type")).unwrap_or_else(|| Value::String("function".to_owned())),
            );
            sorted(&Value::Object(spec))
        })
        .collect();
    json(&Value::Array(specs))
}

/// One assistant tool call as the template writes it:
/// `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>`,
/// the arguments' top-level keys sorted.
///
/// The released template refuses arguments given as a JSON string and asks for them to be
/// canonicalized upstream; the OpenAI wire format sends exactly that string, so it is parsed
/// here. A string that is not a JSON object is refused rather than written as a string.
fn invocation(call: &Value) -> Result<String, PromptError> {
    let function = call.get("function").unwrap_or(&Value::Null);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("tool call function name must be a string"))?;
    let args = match function.get("arguments") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::String(text)) if text.trim().is_empty() => serde_json::Map::new(),
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(map)) => map,
            _ => return Err(invalid("tool call arguments must be a JSON object")),
        },
        Some(Value::Object(map)) => map.clone(),
        Some(_) => return Err(invalid("tool call arguments must be an object")),
    };
    let mut keys: Vec<&String> = args.keys().collect();
    keys.sort();
    let mut text = format!("{MESSAGE_MODEL}{name}{CONTENT_INVOKE}{{\"name\":");
    text.push_str(&json(&Value::String(name.to_owned()))?);
    text.push_str(",\"args\":{");
    for (index, key) in keys.into_iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        text.push_str(&json(&Value::String(key.clone()))?);
        text.push(':');
        text.push_str(&json(&args[key])?);
    }
    text.push_str("}}");
    text.push_str(END_MESSAGE);
    Ok(text)
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
    fn tools_and_calls_render_as_the_released_template_writes_them() {
        let text = InklingCodec::render_chat(&json!({
            "tools": [{"type":"function","function":{
                "name":"lookup",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
            }}],
            "messages": [
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function",
                    "function":{"name":"lookup","arguments":"{\"units\":\"c\",\"city\":\"Paris\"}"}}]},
                {"role":"tool","tool_call_id":"c1","content":"12C"}
            ]
        }))
        .expect("the template renders");
        assert!(text.starts_with(
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"\",\"name\":\"lookup\",\
             \"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"type\":\"object\"},\
             \"type\":\"function\"}]<|end_message|>"
        ));
        // No empty text message beside the call; the arguments' keys sorted; the tool result
        // headed by the call it answers.
        assert!(text.contains(
            "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":\
             {\"city\":\"Paris\",\"units\":\"c\"}}<|end_message|><|content_model_end_sampling|>\
             <|message_tool|>lookup<|content_text|>12C<|end_message|>"
        ));
        assert!(!text.contains("<|message_model|><|content_text|><|end_message|>"));
        assert!(InklingCodec::render_chat(&json!({"messages":[
            {"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"[1]"}}]}
        ]}))
        .is_err());
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
