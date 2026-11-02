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

//! The prompt codec the serving surface tokenizes with.
//!
//! GLM-5.3-Flash ships a Jinja template over a tokenizer whose special tokens are
//! `[gMASK]`, `<sop>`, `<|system|>`, `<|user|>`, `<|assistant|>` and
//! `<|observation|>`. This module renders the text, reasoning and function-call surface of that
//! template:
//!
//! * the `[gMASK]<sop>` prefix and the reasoning-effort system turn the template always writes;
//! * the optional tool catalogue block and the `<tool_call>` XML form;
//! * assistant turns carrying `reasoning_content` or a `<think>...</think>` block, with the
//!   template's `clear_thinking` rule for history;
//! * a run of `tool` messages merged into one `<|observation|>` turn;
//! * the generation prompt `<|assistant|><think>`.
//!
//! The vision placeholders are not rendered: this arm does not load the vision tower, so an image
//! part is refused rather than encoded as tokens the model cannot see. The template's tool-result
//! reordering (matching a response's id to its call) is not reproduced; tool messages keep the
//! caller's order, which is the order the protocol returns them in. A tool signature's and a tool
//! call's argument keys serialize in sorted order (the parsed body's map), where the template writes
//! the caller's insertion order; the XML carries the same JSON either way.

use std::path::Path;

use serde_json::Value;
use sparkflash_model::{PromptCodec, PromptError};

use crate::config::Glm5NextConfig;

/// The serialized special tokens the released template writes.
pub const GMASK: &str = "[gMASK]";
/// The start-of-prompt token.
pub const SOP: &str = "<sop>";
/// The system-turn opener.
pub const SYSTEM: &str = "<|system|>";
/// The user-turn opener.
pub const USER: &str = "<|user|>";
/// The assistant-turn opener.
pub const ASSISTANT: &str = "<|assistant|>";
/// The tool-result turn opener.
pub const OBSERVATION: &str = "<|observation|>";
/// The reasoning block's delimiters.
pub const THINK_OPEN: &str = "<think>";
/// The reasoning block's closing delimiter.
pub const THINK_CLOSE: &str = "</think>";

/// A GLM-5.3-Flash prompt codec over a HuggingFace tokenizer.
pub struct GlmCodec {
    tokenizer: tokenizers::Tokenizer,
    gmask: u32,
    sop: u32,
    system: u32,
    user: u32,
    assistant: u32,
    observation: u32,
    endoftext: u32,
}

impl std::fmt::Debug for GlmCodec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GlmCodec")
            .field("vocabulary", &self.tokenizer.get_vocab_size(true))
            .field("gmask", &self.gmask)
            .field("sop", &self.sop)
            .field("system", &self.system)
            .field("user", &self.user)
            .field("assistant", &self.assistant)
            .field("observation", &self.observation)
            .field("endoftext", &self.endoftext)
            .finish()
    }
}

impl GlmCodec {
    /// Load a `tokenizer.json` from disk.
    ///
    /// # Errors
    /// When the file is missing, is not a tokenizer, or does not carry the released special tokens
    /// at ids inside the checkpoint's vocabulary.
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|error| format!("cannot load tokenizer {}: {error}", path.display()))?;
        Self::new(tokenizer)
    }

    /// Wrap an already loaded tokenizer.
    ///
    /// # Errors
    /// When the tokenizer does not spell one of the released special tokens, when a spelling's id
    /// is outside the checkpoint's vocabulary, or when the tokenizer's vocabulary is wider than the
    /// model's. A silently substituted id would produce a prompt the model reads as ordinary text,
    /// which is worse than refusing to serve.
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Result<Self, String> {
        let config = Glm5NextConfig::glm53_flash()
            .map_err(|error| format!("cannot read the carried configuration: {error}"))?;
        if tokenizer.get_vocab_size(true) > config.vocab_size {
            return Err(format!(
                "the tokenizer holds {} ids, past the checkpoint's vocabulary {}",
                tokenizer.get_vocab_size(true),
                config.vocab_size
            ));
        }
        let special = |spelling: &str| {
            let id = special_id(&tokenizer, spelling)?;
            if id as usize >= config.vocab_size {
                return Err(format!(
                    "the tokenizer's {spelling} is {id}, outside the checkpoint vocabulary {}",
                    config.vocab_size
                ));
            }
            Ok(id)
        };
        let endoftext = special("<|endoftext|>")?;
        if !config.eos_token_ids.contains(&endoftext) {
            return Err(format!(
                "the tokenizer's <|endoftext|> is {endoftext}, but the checkpoint configuration ends at {:?}",
                config.eos_token_ids
            ));
        }
        Ok(Self {
            gmask: special(GMASK)?,
            sop: special(SOP)?,
            system: special(SYSTEM)?,
            user: special(USER)?,
            assistant: special(ASSISTANT)?,
            observation: special(OBSERVATION)?,
            endoftext,
            tokenizer,
        })
    }

    /// The vocabulary size, added tokens included.
    #[must_use]
    pub fn vocabulary(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// The end-of-turn id the template's generation prompt is followed by.
    #[must_use]
    pub fn endoftext(&self) -> u32 {
        self.endoftext
    }

    /// Render the template for a request body without tokenizing it.
    ///
    /// This is the released `chat_template.jinja` reduced to the text/function-call surface the
    /// server implements. See the module documentation for the pieces it does not reproduce.
    ///
    /// # Errors
    /// When the request has no `messages` array or a message asks for something this template
    /// does not render, such as an image part.
    pub fn render(&self, request: &Value) -> Result<String, PromptError> {
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
        if messages.is_empty() {
            return Err(PromptError::invalid(
                "messages must carry at least one turn",
                Some("messages"),
                Some("invalid_request_error"),
            ));
        }
        let kwargs = request.get("chat_template_kwargs");
        let effort = kwargs
            .and_then(|value| value.get("reasoning_effort"))
            .or_else(|| request.get("reasoning_effort"))
            .and_then(Value::as_str)
            .filter(|effort| matches!(*effort, "low" | "high"))
            .unwrap_or("max");
        let clear_thinking = kwargs
            .and_then(|value| value.get("clear_thinking"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let add_generation_prompt = kwargs
            .and_then(|value| value.get("add_generation_prompt"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let mut text = String::from(GMASK);
        text.push_str(SOP);
        // The released template defaults the effort to `max` and always writes the turn.
        text.push_str(SYSTEM);
        text.push_str("Reasoning Effort: ");
        let mut letters = effort.chars();
        if let Some(first) = letters.next() {
            text.extend(first.to_uppercase());
            text.push_str(letters.as_str());
        }
        if let Some(tools) = request.get("tools").and_then(Value::as_array)
            && !tools.is_empty()
        {
            text.push_str(&self.render_tools(tools)?);
        }
        let last_user = messages
            .iter()
            .rposition(|message| message.get("role").and_then(Value::as_str) == Some("user"));
        let mut index = 0;
        while index < messages.len() {
            let message = &messages[index];
            let role = message.get("role").and_then(Value::as_str).ok_or_else(|| {
                PromptError::invalid("a message has no role", Some("messages"), None)
            })?;
            match role {
                "system" => {
                    text.push_str(SYSTEM);
                    text.push_str(&self.visible_text(message.get("content"))?);
                }
                "user" => {
                    text.push_str(USER);
                    text.push_str(&self.visible_text(message.get("content"))?);
                }
                "assistant" => {
                    text.push_str(ASSISTANT);
                    let content = self.visible_text(message.get("content"))?;
                    let (content, reasoning) = split_reasoning(message, &content);
                    // The template keeps a history turn's reasoning only when it is defined,
                    // and under `clear_thinking` only for the turn after the last user message.
                    let keep = reasoning.is_some()
                        && (!clear_thinking || last_user.is_some_and(|last| index > last));
                    match (keep, reasoning) {
                        (true, Some(reasoning)) => {
                            text.push_str(THINK_OPEN);
                            text.push_str(&reasoning);
                            text.push_str(THINK_CLOSE);
                        }
                        _ => {
                            text.push_str(THINK_OPEN);
                            text.push_str(THINK_CLOSE);
                        }
                    }
                    let content = content.trim();
                    if !content.is_empty() {
                        text.push_str(content);
                    }
                    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                        // The template leaves the newline its `{% if m.tool_calls %}` and
                        // `{% endfor %}` lines carry, so a call block is its own line.
                        if !calls.is_empty() {
                            text.push('\n');
                            for call in calls {
                                text.push_str(&render_tool_call(call)?);
                            }
                            text.push('\n');
                        }
                    }
                }
                "tool" => {
                    // The template opens one observation turn for the whole run of tool
                    // messages, so the first of a run writes it and the rest only their response.
                    if index == 0
                        || messages[index - 1].get("role").and_then(Value::as_str) != Some("tool")
                    {
                        text.push_str(OBSERVATION);
                    }
                    let content = self.visible_text(message.get("content"))?;
                    text.push_str("<tool_response>");
                    text.push_str(&content);
                    text.push_str("</tool_response>");
                }
                other => {
                    return Err(PromptError::invalid(
                        format!(
                            "messages[{index}].role {other:?} is not one this template renders"
                        ),
                        Some("messages"),
                        Some("invalid_request_error"),
                    ));
                }
            }
            index += 1;
        }
        if add_generation_prompt {
            text.push_str(ASSISTANT);
            text.push_str(THINK_OPEN);
        }
        Ok(text)
    }

    /// The tool catalogue block the template writes before the messages.
    fn render_tools(&self, tools: &[Value]) -> Result<String, PromptError> {
        let mut text = String::from(SYSTEM);
        text.push_str(
            "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        );
        for tool in tools {
            let tool = tool.get("function").unwrap_or(tool);
            if tool.get("defer_loading").and_then(Value::as_bool) == Some(true) {
                continue;
            }
            // The template's loop leaves a blank line before each signature and two after it.
            text.push('\n');
            text.push_str(&tool_json(tool)?);
            text.push_str("\n\n");
        }
        text.push_str(
            "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>",
        );
        Ok(text)
    }

    /// The template's `visible_text`: a string, or the text parts of a content array.
    fn visible_text(&self, content: Option<&Value>) -> Result<String, PromptError> {
        match content {
            None | Some(Value::Null) => Ok(String::new()),
            Some(Value::String(text)) => Ok(text.clone()),
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text") | None => {
                            if let Some(part) = part.get("text").and_then(Value::as_str) {
                                text.push_str(part);
                            }
                        }
                        Some(
                            "image" | "image_url" | "video" | "video_url" | "audio" | "audio_url"
                            | "input_audio",
                        ) => {
                            return Err(PromptError::invalid(
                                "this arm serves text only: the vision and audio towers are not loaded",
                                Some("messages"),
                                Some("invalid_request_error"),
                            ));
                        }
                        Some(other) => {
                            return Err(PromptError::invalid(
                                format!("content part {other:?} is not one this template renders"),
                                Some("messages"),
                                Some("invalid_request_error"),
                            ));
                        }
                    }
                }
                Ok(text)
            }
            Some(_) => Err(PromptError::invalid(
                "message content must be a string or an array of parts",
                Some("messages"),
                Some("invalid_request_error"),
            )),
        }
    }
}

/// The template's `tool_to_json`: the signature object without `defer_loading` and `strict`.
fn tool_json(tool: &Value) -> Result<String, PromptError> {
    let object = tool.as_object().ok_or_else(|| {
        PromptError::invalid(
            "a tool must be an object with a function signature",
            Some("tools"),
            Some("invalid_request_error"),
        )
    })?;
    // The template writes `"key": <compact json>", ` by hand, so the separators are not the
    // serializer's compact form. Keys come out of the parsed body sorted rather than in wire order;
    // the signature the model reads is the same JSON object either way.
    let mut text = String::from("{");
    let mut first = true;
    for (key, value) in object {
        if key == "defer_loading" || key == "strict" {
            continue;
        }
        if !first {
            text.push_str(", ");
        }
        first = false;
        text.push('"');
        text.push_str(&key.replace('"', "\\\""));
        text.push_str("\": ");
        text.push_str(
            &serde_json::to_string(value)
                .map_err(|error| PromptError::internal(format!("tool signature: {error}")))?,
        );
    }
    text.push('}');
    Ok(text)
}

/// One assistant tool call in the template's XML form.
fn render_tool_call(call: &Value) -> Result<String, PromptError> {
    let call = call.get("function").unwrap_or(call);
    let name = call.get("name").and_then(Value::as_str).ok_or_else(|| {
        PromptError::invalid(
            "a tool call has no function name",
            Some("messages"),
            Some("invalid_request_error"),
        )
    })?;
    let arguments = match call.get("arguments") {
        Some(Value::String(text)) => serde_json::from_str::<Value>(text).map_err(|error| {
            PromptError::invalid(
                format!("tool call arguments are not JSON: {error}"),
                Some("messages"),
                Some("invalid_request_error"),
            )
        })?,
        Some(value) => value.clone(),
        None => Value::Object(serde_json::Map::new()),
    };
    let arguments = arguments.as_object().ok_or_else(|| {
        PromptError::invalid(
            "tool call arguments must be a JSON object",
            Some("messages"),
            Some("invalid_request_error"),
        )
    })?;
    let mut text = String::from("<tool_call>");
    text.push_str(name);
    for (key, value) in arguments {
        text.push_str("<arg_key>");
        text.push_str(key);
        text.push_str("</arg_key><arg_value>");
        match value {
            Value::String(value) => text.push_str(value),
            value => text.push_str(
                &serde_json::to_string(value)
                    .map_err(|error| PromptError::internal(format!("tool argument: {error}")))?,
            ),
        }
        text.push_str("</arg_value>");
    }
    text.push_str("</tool_call>");
    Ok(text)
}

/// An assistant turn's visible content and reasoning, from `reasoning_content` or a `<think>`
/// block in the content, as the template splits them.
fn split_reasoning(message: &Value, content: &str) -> (String, Option<String>) {
    if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
        return (content.to_owned(), Some(reasoning.to_owned()));
    }
    if let Some((reasoning, visible)) = content.split_once(THINK_CLOSE) {
        let reasoning = reasoning
            .rsplit_once(THINK_OPEN)
            .map_or(reasoning, |(_, reasoning)| reasoning);
        return (visible.to_owned(), Some(reasoning.to_owned()));
    }
    (content.to_owned(), None)
}

/// The id of a special token the tokenizer must spell.
fn special_id(tokenizer: &tokenizers::Tokenizer, spelling: &str) -> Result<u32, String> {
    tokenizer.token_to_id(spelling).ok_or_else(|| {
        format!("the tokenizer does not spell {spelling}; is this GLM-5.3-Flash's tokenizer.json?")
    })
}

impl PromptCodec for GlmCodec {
    fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, PromptError> {
        let text = self.render(request)?;
        self.encode_text(&text, false)
    }

    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| PromptError::internal(format!("tokenizer: {error}")))
    }

    fn decode(&self, ids: &[u32]) -> Option<String> {
        self.tokenizer.decode(ids, true).ok()
    }

    fn token_str(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A tiny word-level tokenizer carrying the released specials at the checkpoint's ids.
    ///
    /// The end-of-text id is the released one, because that is the id the codec validates against
    /// the checkpoint configuration; the other specials sit just below it so the sparse vocabulary
    /// stays a valid WordLevel model.
    fn tiny_tokenizer() -> tokenizers::Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (token, id) in [
            ("<unk>", 0),
            ("hello", 1),
            ("hi", 2),
            ("there", 3),
            ("<|endoftext|>", 154_820),
            ("<think>", 154_816),
            ("</think>", 154_817),
            (GMASK, 154_810),
            (SOP, 154_811),
            (SYSTEM, 154_812),
            (USER, 154_813),
            (ASSISTANT, 154_814),
            (OBSERVATION, 154_815),
        ] {
            vocab.insert(token.to_owned(), json!(id));
        }
        let mut added = Vec::new();
        for (token, id) in [
            ("<|endoftext|>", 154_820),
            ("<think>", 154_816),
            ("</think>", 154_817),
            (GMASK, 154_810),
            (SOP, 154_811),
            (SYSTEM, 154_812),
            (USER, 154_813),
            (ASSISTANT, 154_814),
            (OBSERVATION, 154_815),
        ] {
            added.push(json!({
                "id": id,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            }));
        }
        let document = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": Value::Object(vocab),
                "unk_token": "<unk>",
            },
        });
        tokenizers::Tokenizer::from_bytes(document.to_string().as_bytes())
            .expect("the toy tokenizer document parses")
    }

    fn codec() -> GlmCodec {
        GlmCodec::new(tiny_tokenizer()).expect("the toy tokenizer carries the specials")
    }

    #[test]
    fn the_template_renders_the_released_prefix_and_generation_prompt() {
        let codec = codec();
        let request = json!({"messages": [{"role": "user", "content": "hello"}]});
        let text = codec.render(&request).expect("the template renders");
        assert_eq!(
            text,
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hello<|assistant|><think>"
        );
        let ids = codec.encode_chat(&request).expect("the template tokenizes");
        assert_eq!(&ids[..3], &[154_810, 154_811, 154_812]);
        assert!(
            ids.contains(&154_813),
            "the user turn's id is in the prompt"
        );
        assert_eq!(ids.last(), Some(&154_816), "the prompt ends in <think>");
        assert_eq!(codec.endoftext(), 154_820);
    }

    #[test]
    fn a_tool_catalogue_and_a_reasoning_effort_are_rendered_before_the_turns() {
        let codec = codec();
        let request = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"reasoning_effort": "high"},
            "tools": [{
                "type": "function",
                "function": {
                    "description": "look something up",
                    "name": "lookup",
                    "strict": true,
                    "parameters": {"type": "object"},
                },
            }],
        });
        let text = codec.render(&request).expect("the template renders");
        assert_eq!(
            text,
            "[gMASK]<sop><|system|>Reasoning Effort: High<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\n{\"description\": \"look something up\", \"name\": \"lookup\", \"parameters\": {\"type\":\"object\"}}\n\n\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call><|user|>hello<|assistant|><think>"
        );
        assert!(
            !text.contains("\"strict\""),
            "strict is not part of the signature"
        );
    }

    #[test]
    fn clear_thinking_keeps_only_the_last_turns_reasoning() {
        let codec = codec();
        let messages = json!([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": "one"},
            {"role": "user", "content": "there"},
            {"role": "assistant", "content": "hi", "reasoning_content": "two"},
        ]);
        let request = json!({"messages": messages});
        let text = codec.render(&request).expect("the template renders");
        assert!(text.contains("<|assistant|><think>one</think>hi"));
        assert!(text.contains("<|assistant|><think>two</think>hi<|assistant|><think>"));

        let cleared = json!({
            "messages": messages,
            "chat_template_kwargs": {"clear_thinking": true},
        });
        let text = codec.render(&cleared).expect("the template renders");
        assert!(
            text.contains("<|assistant|><think></think>hi"),
            "a history turn clears its reasoning"
        );
        assert!(text.contains("<|assistant|><think>two</think>hi"));
    }

    #[test]
    fn a_tool_call_and_its_result_render_in_the_template_xml() {
        let codec = codec();
        let request = json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"city\": \"Paris\"}"},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                {"role": "tool", "tool_call_id": "call_2", "content": "warm"},
            ],
        });
        let text = codec.render(&request).expect("the template renders");
        assert!(text.ends_with(
            "<|assistant|><think></think>\n<tool_call>lookup<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>\n<|observation|><tool_response>sunny</tool_response><tool_response>warm</tool_response><|assistant|><think>"
        ));
        assert_eq!(
            text.matches("<|observation|>").count(),
            1,
            "a run of tool messages is one observation turn"
        );
    }

    #[test]
    fn an_image_part_is_refused_rather_than_rendered_unseen() {
        let codec = codec();
        let request = json!({"messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        }]});
        let error = codec.render(&request).expect_err("vision is not loaded");
        assert!(error.message().contains("text only"), "{}", error.message());
    }

    #[test]
    fn a_request_without_messages_is_refused() {
        let codec = codec();
        let error = codec
            .render(&json!({}))
            .expect_err("no messages is a request error");
        assert!(error.message().contains("messages"));
    }
}
