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
//! Qwen3.8-Flash-Next uses the ChatML template its `tokenizer_config.json` carries. The codec
//! renders the text/function-call surface of that template: the tool catalogue and its
//! `<tool_call>` XML instructions, assistant turns carrying `reasoning_content` and `tool_calls`,
//! the merged `<tool_response>` turn, and the thinking-mode generation prompt
//! (`<|im_start|>assistant\n<think>\n`, or the already-closed block for
//! `enable_thinking: false`). An image part renders as the template's single `<|image_pad|>`
//! placeholder, which the vision codec expands into the packed span. The device arm still refuses
//! prepared images because its prefill cannot write the merged rows yet.

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use sparkflash_model::{PromptCodec, PromptError};

use crate::config::{ENDOFTEXT_TOKEN, IM_END_TOKEN};
use crate::vision::QwenVisionCodec;

/// The system turn the template inserts when a request has none.
pub const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

/// The special-token spellings the template renders.
pub const IM_START: &str = "<|im_start|>";
/// The end-of-turn spelling.
pub const IM_END: &str = "<|im_end|>";
/// The image placeholder the released template writes for one image part.
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// The delimiter the vision codec writes before an image's merged rows.
pub const VISION_START: &str = "<|vision_start|>";
/// The delimiter the vision codec writes after an image's merged rows.
pub const VISION_END: &str = "<|vision_end|>";

/// A ChatML prompt codec over a HuggingFace tokenizer.
pub struct QwenCodec {
    tokenizer: tokenizers::Tokenizer,
    im_start: u32,
    im_end: u32,
    endoftext: u32,
    image_token: u32,
    vision: Option<QwenVisionCodec>,
}

impl std::fmt::Debug for QwenCodec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QwenCodec")
            .field("vocabulary", &self.tokenizer.get_vocab_size(true))
            .field("im_start", &self.im_start)
            .field("im_end", &self.im_end)
            .field("endoftext", &self.endoftext)
            .field("image_token", &self.image_token)
            .field("vision", &self.vision.is_some())
            .finish()
    }
}

impl QwenCodec {
    /// Load a `tokenizer.json` from disk.
    ///
    /// # Errors
    /// When the file is missing, is not a tokenizer, or does not carry the ChatML special tokens
    /// at the ids the checkpoint's configuration expects.
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|error| format!("cannot load tokenizer {}: {error}", path.display()))?;
        Self::new(tokenizer)
    }

    /// Wrap an already loaded tokenizer.
    ///
    /// # Errors
    /// When the tokenizer does not spell `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`,
    /// `<|image_pad|>`, `<|vision_start|>` or `<|vision_end|>`, or when their ids disagree with
    /// the checkpoint configuration. A silently substituted id would produce a prompt the model
    /// reads as ordinary text, which is worse than refusing to serve.
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Result<Self, String> {
        let im_start = special_id(&tokenizer, IM_START)?;
        let im_end = special_id(&tokenizer, IM_END)?;
        let endoftext = special_id(&tokenizer, "<|endoftext|>")?;
        let config = crate::Qwen4ExpConfig::qwen38_flash_next_nvfp4()
            .map_err(|error| format!("cannot read the carried configuration: {error}"))?;
        if endoftext != ENDOFTEXT_TOKEN {
            return Err(format!(
                "the tokenizer's <|endoftext|> is {endoftext}, but the checkpoint configuration ends at {ENDOFTEXT_TOKEN}"
            ));
        }
        if im_end != IM_END_TOKEN {
            return Err(format!(
                "the tokenizer's <|im_end|> is {im_end}, but the checkpoint configuration stops at {IM_END_TOKEN}"
            ));
        }
        if im_start >= config.vocab_size as u32 {
            return Err(format!(
                "the tokenizer's <|im_start|> is {im_start}, outside the checkpoint vocabulary {}",
                config.vocab_size
            ));
        }
        let image_token = special_id(&tokenizer, IMAGE_PAD)?;
        if image_token != config.image_token_id {
            return Err(format!(
                "the tokenizer's <|image_pad|> is {image_token}, but the checkpoint configuration names {}",
                config.image_token_id
            ));
        }
        // The vision codec inserts these two ids directly from the configuration. Validate the
        // spellings as well, or a tokenizer whose delimiters sit at different ids would receive
        // prompts that look right as text but decode to the wrong special tokens.
        for (spelling, configured) in [
            (VISION_START, config.vision_start_token_id),
            (VISION_END, config.vision_end_token_id),
        ] {
            let actual = special_id(&tokenizer, spelling)?;
            if actual != configured {
                return Err(format!(
                    "the tokenizer's {spelling} is {actual}, but the checkpoint configuration names {configured}"
                ));
            }
        }
        let vision = QwenVisionCodec::from_config(&config);
        Ok(Self {
            tokenizer,
            im_start,
            im_end,
            endoftext,
            image_token,
            vision,
        })
    }

    /// The vocabulary size, added tokens included.
    #[must_use]
    pub fn vocabulary(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Render the template for a request body without tokenizing it.
    ///
    /// This is the released `chat_template.jinja` reduced to the text/function-call surface the
    /// server implements: the tool block and its XML call format, assistant `reasoning_content`
    /// and `tool_calls` continuation, the merged `<tool_response>` turn, and the thinking-mode
    /// generation prompt.
    ///
    /// # Errors
    /// When the request has no `messages` array, a message that is not a text turn, or a
    /// `messages` entry that asks for something this template does not render.
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
        let thinking = thinking_enabled(request, kwargs);
        let reasoning_instructions = if thinking {
            reasoning_instructions(reasoning_effort_level(request, kwargs))
        } else {
            ""
        };
        let tools = request
            .get("tools")
            .and_then(Value::as_array)
            .filter(|tools| !tools.is_empty());
        let preserve_thinking = kwargs
            .and_then(|value| value.get("preserve_thinking"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let last_query_index = last_query_index(messages)?;

        let mut text = String::new();
        let mut first = 0usize;
        let first_is_system = matches!(
            messages[0].get("role").and_then(Value::as_str),
            Some("system" | "developer")
        );
        if let Some(tools) = tools {
            // With tools, the template always opens a system turn: the tool catalogue lives
            // there, and a leading system message becomes the tail of that same turn.
            text.push_str(IM_START);
            text.push_str("system\n");
            if !reasoning_instructions.is_empty() {
                text.push_str(reasoning_instructions);
                text.push_str("\n\n");
            }
            text.push_str(TOOLS_PROMPT);
            for tool in tools {
                text.push('\n');
                push_python_json(&mut text, tool);
            }
            text.push_str("\n</tools>");
            text.push_str(TOOL_CALL_INSTRUCTIONS);
            if first_is_system {
                let content = content_text(&messages[0], Some(IMAGE_PAD))?;
                let content = content.trim();
                if !content.is_empty() {
                    text.push_str("\n\n");
                    text.push_str(content);
                }
                first = 1;
            }
            text.push_str(IM_END);
            text.push('\n');
        } else if first_is_system {
            let content = content_text(&messages[0], Some(IMAGE_PAD))?;
            let content = content.trim();
            if !content.is_empty() || !reasoning_instructions.is_empty() {
                text.push_str(IM_START);
                text.push_str("system\n");
                if !reasoning_instructions.is_empty() {
                    text.push_str(reasoning_instructions);
                    if !content.is_empty() {
                        text.push_str("\n\n");
                    }
                }
                text.push_str(content);
                text.push_str(IM_END);
                text.push('\n');
            }
            first = 1;
        } else if !reasoning_instructions.is_empty() {
            text.push_str(IM_START);
            text.push_str("system\n");
            text.push_str(reasoning_instructions);
            text.push_str(IM_END);
            text.push('\n');
        }

        for (index, message) in messages.iter().enumerate().skip(first) {
            let role = message.get("role").and_then(Value::as_str).ok_or_else(|| {
                PromptError::invalid(
                    "every message needs a role",
                    Some("messages"),
                    Some("invalid_request_error"),
                )
            })?;
            match role {
                "system" | "developer" => {
                    if index != 0 {
                        return Err(PromptError::invalid(
                            "a system message must be the first turn",
                            Some("messages"),
                            Some("invalid_request_error"),
                        ));
                    }
                }
                "user" => {
                    text.push_str(IM_START);
                    text.push_str("user\n");
                    text.push_str(content_text(message, Some(IMAGE_PAD))?.trim());
                    text.push_str(IM_END);
                    text.push('\n');
                }
                "assistant" => {
                    let content = content_text(message, Some(IMAGE_PAD))?;
                    let content = content.trim();
                    let reasoning = message
                        .get("reasoning_content")
                        .or_else(|| message.get("reasoning"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim();
                    text.push_str(IM_START);
                    text.push_str("assistant\n");
                    if preserve_thinking || index > last_query_index {
                        text.push_str("<think>\n");
                        text.push_str(reasoning);
                        text.push_str("\n</think>\n\n");
                    }
                    text.push_str(content);
                    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                        for (call_index, call) in tool_calls.iter().enumerate() {
                            let function = call.get("function").unwrap_or(call);
                            let name =
                                function
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .ok_or_else(|| {
                                        PromptError::invalid(
                                            "every tool call needs a function name",
                                            Some("messages"),
                                            Some("invalid_request_error"),
                                        )
                                    })?;
                            if call_index == 0 {
                                if content.is_empty() {
                                    text.push_str("<tool_call>\n");
                                } else {
                                    text.push_str("\n\n<tool_call>\n");
                                }
                            } else {
                                text.push_str("\n<tool_call>\n");
                            }
                            text.push_str("<function=");
                            text.push_str(name);
                            text.push_str(">\n");
                            if let Some(arguments) = function.get("arguments")
                                && let Some(arguments) = arguments_object(arguments)
                            {
                                for (key, value) in arguments {
                                    text.push_str("<parameter=");
                                    text.push_str(&key);
                                    text.push_str(">\n");
                                    push_template_value(&mut text, &value);
                                    text.push_str("\n</parameter>\n");
                                }
                            }
                            text.push_str("</function>\n</tool_call>");
                        }
                    }
                    text.push_str(IM_END);
                    text.push('\n');
                }
                "tool" => {
                    let merged_before = index > 0
                        && messages[index - 1].get("role").and_then(Value::as_str) == Some("tool");
                    let merged_after = messages
                        .get(index + 1)
                        .and_then(|next| next.get("role"))
                        .and_then(Value::as_str)
                        == Some("tool");
                    if !merged_before {
                        text.push_str(IM_START);
                        text.push_str("user");
                    }
                    text.push_str("\n<tool_response>\n");
                    text.push_str(content_text(message, Some(IMAGE_PAD))?.trim());
                    text.push_str("\n</tool_response>");
                    if !merged_after {
                        text.push_str(IM_END);
                        text.push('\n');
                    }
                }
                other => {
                    return Err(PromptError::invalid(
                        format!("this codec does not render the {other:?} role"),
                        Some("messages"),
                        Some("invalid_request_error"),
                    ));
                }
            }
        }

        let add_generation_prompt = kwargs
            .and_then(|value| value.get("add_generation_prompt"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if add_generation_prompt {
            text.push_str(IM_START);
            text.push_str("assistant\n");
            if thinking {
                text.push_str("<think>\n");
            } else {
                text.push_str("<think>\n\n</think>\n\n");
            }
        }
        Ok(text)
    }
}

/// The fixed tool catalogue heading in the released template.
const TOOLS_PROMPT: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

/// The fixed call-format instructions in the released template.
const TOOL_CALL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Whether the released template should open a thinking block.
fn thinking_enabled(request: &Value, kwargs: Option<&Value>) -> bool {
    if let Some(thinking) = kwargs.and_then(|value| value.get("enable_thinking")) {
        return thinking.as_bool().unwrap_or(true);
    }
    if let Some(thinking) = kwargs.and_then(|value| value.get("thinking")) {
        match thinking {
            Value::Bool(flag) => return *flag,
            Value::String(text) if text.eq_ignore_ascii_case("non-thinking") => return false,
            _ => {}
        }
    }
    request
        .get("enable_thinking")
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

/// The template's three reasoning-effort spellings.
fn reasoning_effort_level(request: &Value, kwargs: Option<&Value>) -> &'static str {
    let raw = kwargs
        .and_then(|value| value.get("reasoning_effort"))
        .or_else(|| request.get("reasoning_effort"));
    match raw {
        Some(Value::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "low" | "minimal" => "low",
            "medium" => "medium",
            _ => "xhigh",
        },
        Some(Value::Number(number)) => match number.as_i64() {
            Some(score) if score <= 25 => "low",
            Some(score) if score <= 50 => "medium",
            _ => "xhigh",
        },
        _ => "xhigh",
    }
}

fn reasoning_instructions(level: &str) -> &'static str {
    match level {
        "low" => {
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
        }
        "medium" => "",
        _ => {
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
        }
    }
}

/// The index of the last user turn that is not a tool result, for `preserve_thinking`.
fn last_query_index(messages: &[Value]) -> Result<usize, PromptError> {
    let mut last_query_index = messages.len().saturating_sub(1);
    let mut multi_step_tool = true;
    for (index, message) in messages.iter().enumerate().rev() {
        if !multi_step_tool || message.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let content = content_text(message, Some(IMAGE_PAD))?;
        let content = content.trim();
        if !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>")) {
            multi_step_tool = false;
            last_query_index = index;
        }
    }
    Ok(last_query_index)
}

/// A tool call's `arguments`: an object, or the OpenAI wire spelling of one.
fn arguments_object(value: &Value) -> Option<serde_json::Map<String, Value>> {
    match value {
        Value::Object(map) => Some(map.clone()),
        Value::String(text) if !text.trim().is_empty() => serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|value| value.as_object().cloned()),
        _ => None,
    }
}

/// The template's value rendering: strings raw, every other JSON value as `tojson` renders it.
fn push_template_value(text: &mut String, value: &Value) {
    match value {
        Value::String(value) => text.push_str(value),
        other => push_python_json(text, other),
    }
}

/// Python's `json.dumps` spacing, which is what the template's `tojson` filter emits: a space
/// after every `:` and `,`. The model was trained on that spelling, so the prompt keeps it.
fn push_python_json(text: &mut String, value: &Value) {
    match value {
        Value::Null => text.push_str("null"),
        Value::Bool(true) => text.push_str("true"),
        Value::Bool(false) => text.push_str("false"),
        Value::Number(number) => text.push_str(&number.to_string()),
        Value::String(string) => {
            text.push_str(&serde_json::to_string(string).unwrap_or_default());
        }
        Value::Array(items) => {
            text.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    text.push_str(", ");
                }
                push_python_json(text, item);
            }
            text.push(']');
        }
        Value::Object(map) => {
            text.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    text.push_str(", ");
                }
                text.push_str(&serde_json::to_string(key).unwrap_or_default());
                text.push_str(": ");
                push_python_json(text, item);
            }
            text.push('}');
        }
    }
}

/// The text of one message's content: a string, or a list of text and image parts.
///
/// An image part renders as the one placeholder the released template writes. The API's vision
/// front end expands that placeholder into the image span before generation.
fn content_text(message: &Value, image_placeholder: Option<&str>) -> Result<String, PromptError> {
    match message.get("content") {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Null) | None => Ok(String::new()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => text.push_str(
                        part.get("text").and_then(Value::as_str).ok_or_else(|| {
                            PromptError::invalid(
                                "a text part needs its text",
                                Some("messages"),
                                Some("invalid_request_error"),
                            )
                        })?,
                    ),
                    Some("image_url" | "input_image" | "image") => {
                        let Some(placeholder) = image_placeholder else {
                            return Err(PromptError::invalid(
                                "this codec has no vision front end to place an image",
                                Some("messages"),
                                Some("invalid_request_error"),
                            ));
                        };
                        text.push_str(placeholder);
                    }
                    Some(other) => {
                        return Err(PromptError::invalid(
                            format!("this codec does not render a {other:?} content part"),
                            Some("messages"),
                            Some("invalid_request_error"),
                        ));
                    }
                    None => {
                        return Err(PromptError::invalid(
                            "a content part needs a type",
                            Some("messages"),
                            Some("invalid_request_error"),
                        ));
                    }
                }
            }
            Ok(text)
        }
        _ => Err(PromptError::invalid(
            "every message needs string content or a list of text parts",
            Some("messages"),
            Some("invalid_request_error"),
        )),
    }
}

fn special_id(tokenizer: &tokenizers::Tokenizer, spelling: &str) -> Result<u32, String> {
    tokenizer
        .token_to_id(spelling)
        .ok_or_else(|| format!("the tokenizer does not carry the {spelling} special token"))
}

/// A byte codec over a small vocabulary, for the tiny fixture.
///
/// The fixture's vocabulary is 64 tokens, so a real tokenizer cannot address it and the released
/// ChatML ids are meaningless. This codec maps text bytes and token ids onto that vocabulary
/// (id 0 is never produced, because the fixture's end-of-text id is 2 and a demo that ends
/// immediately would prove nothing). It exists to exercise the serving path - router, queue,
/// streaming, sampling - on hardware that has no 132.6 GB checkpoint, and it says so in its type
/// name rather than pretending to be a tokenizer.
#[derive(Debug, Clone, Copy)]
pub struct ByteCodec {
    vocabulary: u32,
}

impl ByteCodec {
    /// A codec for a vocabulary of this size (at least 3, so a byte can avoid the stop ids).
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
        // Skip the low ids so a prompt byte does not land on an end-of-generation id.
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

impl PromptCodec for QwenCodec {
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

    fn vision_codec(&self) -> Option<Arc<dyn sparkflash_model::vision::VisionCodec>> {
        self.vision
            .clone()
            .map(|vision| Arc::new(vision) as Arc<dyn sparkflash_model::vision::VisionCodec>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A tiny word-level tokenizer carrying the ChatML specials at the checkpoint's ids.
    ///
    /// The ids are the released checkpoint's, because that is exactly what the codec validates:
    /// a toy layout with its own numbering would let a wrong id pass the check it exists to make.
    /// The vocabulary is sparse - a handful of words at the top of the range - which is a valid
    /// WordLevel model as long as every id is below the declared size.
    fn tiny_tokenizer() -> tokenizers::Tokenizer {
        let mut vocab = serde_json::Map::new();
        vocab.insert("<|endoftext|>".to_owned(), json!(ENDOFTEXT_TOKEN));
        vocab.insert(IM_END.to_owned(), json!(IM_END_TOKEN));
        vocab.insert(IM_START.to_owned(), json!(IM_END_TOKEN + 1));
        vocab.insert(IMAGE_PAD.to_owned(), json!(248_056));
        vocab.insert(VISION_START.to_owned(), json!(248_053));
        vocab.insert(VISION_END.to_owned(), json!(248_054));
        vocab.insert("hello".to_owned(), json!(IM_END_TOKEN + 2));
        vocab.insert("world".to_owned(), json!(IM_END_TOKEN + 3));
        let mut added = Vec::new();
        for (token, id) in [
            ("<|endoftext|>", ENDOFTEXT_TOKEN),
            (IM_END, IM_END_TOKEN),
            (IM_START, IM_END_TOKEN + 1),
            (IMAGE_PAD, 248_056),
            (VISION_START, 248_053),
            (VISION_END, 248_054),
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
                "unk_token": "<|endoftext|>",
            },
        });
        tokenizers::Tokenizer::from_bytes(document.to_string().as_bytes())
            .expect("the toy tokenizer document parses")
    }

    #[test]
    fn the_template_renders_the_released_chatml_shape() {
        let codec =
            QwenCodec::new(tiny_tokenizer()).expect("the toy tokenizer carries the specials");
        let request = json!({
            "messages": [{"role": "user", "content": "hello"}],
        });
        let text = codec.render(&request).expect("the template renders");
        assert_eq!(
            text,
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                reasoning_instructions("xhigh")
            )
        );
        let ids = codec.encode_chat(&request).expect("the template tokenizes");
        assert_eq!(
            ids[0],
            IM_END_TOKEN + 1,
            "the prompt must open with im_start"
        );
    }

    #[test]
    fn a_system_turn_carries_the_instructions_and_the_generation_prompt_can_be_dropped() {
        let codec =
            QwenCodec::new(tiny_tokenizer()).expect("the toy tokenizer carries the specials");
        let request = json!({
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ],
            "chat_template_kwargs": {"add_generation_prompt": false},
        });
        let text = codec.render(&request).expect("the template renders");
        assert!(
            text.contains(&format!(
                "<|im_start|>system\n{}\n\nbe brief<|im_end|>",
                reasoning_instructions("xhigh")
            )),
            "{text}"
        );
        assert!(!text.contains("assistant\n"), "{text}");
    }

    #[test]
    fn non_thinking_requests_close_the_think_block() {
        let codec =
            QwenCodec::new(tiny_tokenizer()).expect("the toy tokenizer carries the specials");
        let request = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": false},
        });
        let text = codec.render(&request).expect("the template renders");
        assert!(
            text.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{text}"
        );
    }

    #[test]
    fn tools_render_the_catalogue_and_the_xml_call_format() {
        let codec =
            QwenCodec::new(tiny_tokenizer()).expect("the toy tokenizer carries the specials");
        let request = json!({
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }],
        });
        let text = codec.render(&request).expect("the template renders");
        assert!(
            text.contains("# Tools\n\nYou have access to the following functions:"),
            "{text}"
        );
        assert!(text.contains("<tools>\n{"), "{text}");
        assert!(text.contains("\"name\": \"get_weather\""), "{text}");
        assert!(
            text.contains("\"properties\": {\"city\": {\"type\": \"string\"}}"),
            "{text}"
        );
        assert!(text.contains("\n</tools>"), "{text}");
        assert!(text.contains("<function=example_function_name>"), "{text}");
        assert!(text.contains("\n\nbe brief<|im_end|>\n"), "{text}");
        assert!(text.ends_with("<|im_start|>assistant\n<think>\n"), "{text}");
    }

    #[test]
    fn an_assistant_tool_turn_and_a_tool_response_continue_the_conversation() {
        let codec =
            QwenCodec::new(tiny_tokenizer()).expect("the toy tokenizer carries the specials");
        let request = json!({
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "I should call the tool.",
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Beijing\",\"days\":3}"
                        }
                    }]
                },
                {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
                {"role": "user", "content": "thanks"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "chat_template_kwargs": {"enable_thinking": false},
        });
        let text = codec.render(&request).expect("the template renders");
        assert!(text.contains("<|im_start|>assistant\n<think>\nI should call the tool.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>\n"), "{text}");
        assert!(
            text.contains("<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n"),
            "{text}"
        );
    }

    #[test]
    fn an_image_part_becomes_the_template_placeholder() {
        let codec =
            QwenCodec::new(tiny_tokenizer()).expect("the toy tokenizer carries the specials");
        let image = json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ]}],
        });
        let rendered = codec
            .render(&image)
            .expect("the template renders the image");
        assert!(rendered.contains(IMAGE_PAD), "{rendered}");
        let ids = codec.encode_text(&rendered, false).expect("tokenizes");
        assert_eq!(
            ids.iter().filter(|id| **id == 248_056).count(),
            1,
            "one image part is one placeholder"
        );
        let vision = codec
            .vision_codec()
            .expect("the pack has a vision front end");
        assert_eq!(vision.image_token_id(), 248_056);
    }

    #[test]
    fn a_tokenizer_with_the_wrong_specials_is_refused() {
        let document = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"},
        });
        let tokenizer = tokenizers::Tokenizer::from_bytes(document.to_string().as_bytes())
            .expect("the document parses");
        let error = QwenCodec::new(tokenizer).expect_err("no ChatML specials");
        assert!(error.contains("im_start"), "{error}");
    }
}
