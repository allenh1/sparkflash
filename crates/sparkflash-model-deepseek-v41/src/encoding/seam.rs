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

//! The prompt tokenizer seam: real token ids and the V4.1 chat template for
//! the HTTP path.
//!
//! Until this seam existed the router sized prompts with the Milestone 1
//! heuristic (prompt.rs) and handed placeholder ids to the engine, so a
//! request that talked to the model over HTTP did not carry the checkpoint's
//! tokenization. This module closes that gap for the raw-completion route and
//! for /v1/chat/completions.
//!
//! # What the reference server does
//!
//! The released recipe serves the model with vLLM tokenizer_mode=deepseek_v41.
//! That mode wraps the checkpoint's own tokenizer.json and routes
//! /v1/chat/completions through a DeepseekV4Renderer whose apply_chat_template
//! is implemented in code (not a Jinja template string) by
//! vllm/tokenizers/deepseek_v41.py and
//! vllm/tokenizers/deepseek_v41_encoding.py, a serving-side copy of the
//! checkpoint's encoding/encoding.py. This module mirrors that pipeline:
//!
//! 1. normalise the raw OpenAI messages exactly as _normalize_messages does,
//! 2. attach tools to the first system message (inserting an empty one if the
//!    conversation has none),
//! 3. resolve thinking/effort from the request the way the vLLM chat protocol
//!    merges them into chat_template_kwargs,
//! 4. render with the official encoder port (render::encode_messages_text),
//! 5. tokenize with the checkpoint tokenizer and add_special_tokens=false,
//!    which is exactly what the vLLM wrapper does.
//!
//! What is matched, precisely, is the image's encoding module
//! (deepseek_v41_encoding.py), not a Jinja template string: the image copy
//! reproduces reference/encoding/tests/test_output_1..5.txt byte-for-byte
//! when fed the fixtures, and the Rust port is golden-tested against the same
//! files. The refusal texts for invalid role, unsupported content part and
//! unsupported effort are byte-identical with the image's raises - a parity
//! property, not a paraphrase.
//!
//! Raw completion prompts are tokenized with the same tokenizer and also
//! without added special tokens. This tokenizer's post_processor is
//! ByteLevel, so add_special_tokens changes nothing: the released
//! tokenizer_config.json sets add_bos_token=false and add_eos_token=false and
//! the official tokenizer adds neither.
//!
//! # Provenance (recorded 2026-10-19)
//!
//! * Served snapshot (from start-vllm.log):
//!   /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/
//!   snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 (read on node .80
//!   under /home/user/.cache/...).
//! * tokenizer.json sha256
//!   c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b,
//!   byte-identical to this repository's reference/tokenizer.json.
//! * tokenizer_config.json sha256
//!   6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547,
//!   add_bos_token=false, add_eos_token=false,
//!   pad_token=<|end of sentence|> (the released AddedToken), no chat_template
//!   key.
//! * vLLM image vllm/vllm-openai:deepseekv41-flash-0909 (digest
//!   sha256:714a937591e162044e9142b0ba2ed18bd21734edcffb59f651db6ced941a2040):
//!   vllm/tokenizers/deepseek_v41.py and
//!   vllm/tokenizers/deepseek_v41_encoding.py; renderer registry maps
//!   deepseek_v41 to DeepseekV4Renderer.
//!
//! # Deliberate deviations
//!
//! * /v1/completions with prompt as an array of strings still joins them with
//!   a newline (the Milestone 1 single-prompt behaviour) and then tokenizes
//!   the joined text. vLLM treats the array as a batch of prompts. Batching is
//!   not implemented yet; the joined text is tokenized for real, which at
//!   least keeps the ids faithful to what the router hands the engine.
//! * /v1/chat/completions echo remains unimplemented (see schema.rs): the
//!   template now renders, but echoing the final message changes the
//!   generation assembly, not the prompt seam, and is deferred with the
//!   logprob protocol work.
//! * A developer role is accepted by vLLM's normaliser and then handled by
//!   the encoder's drop_thinking filter: in thinking mode (the default) the
//!   message is dropped, exactly as the released server drops it, and in chat
//!   mode the renderer refuses it with its typed UnknownRole error. Any role
//!   outside system/user/assistant/tool is refused as Invalid role: <role>,
//!   the exact vLLM text.
//! * Only an image content part becomes the image placeholder's id. vLLM
//!   tokenizes the rendered string, so the placeholder's spelling anywhere in
//!   the conversation (a tool result showing source code that names it, say)
//!   becomes the same id, and the request then fails for want of an image.
//!   Here that spelling is tokenized as the text it is. Every other id is
//!   unchanged: the chat prompt is tokenized with the checkpoint tokenizer
//!   minus that one added token, split where the image parts were rendered.
//!
//! # Token id evidence
//!
//! Pinned id vectors live in docs/build/task30-evidence/tokenizer-parity.json.
//! They were produced by loading the released snapshot's tokenizer.json with
//! the released image's own tokenizer stack (transformers AutoTokenizer, CPU
//! only: no model, no GPU, no server), tokenizing the encoder's prompt strings
//! exactly as the vLLM wrapper does (add_special_tokens=false). The
//! window-time endpoint cross-check (vLLM /tokenize) remains pending because
//! the window is closed; the released image's tokenizer code is the oracle
//! recorded here.

use serde_json::{Map, Value};
use sparkflash_model::{PromptCodec, PromptError};

use super::render::encode_messages_text;
use super::{
    CaseError, EncodingMessage, HfTokenizerEncoder, IMAGE_PLACEHOLDER, MessageContent, OpenAiTool,
    OpenAiToolCall, ThinkingMode, load_hf_tokenizer,
};

/// Alias budgets of the official V4.1 encoder (encoding.py): low 25, high 50,
/// xhigh 75, max 100. This is NOT the recipe-scale surface table in
/// crate::reasoning; the chat template renders this scale because that is what
/// the released server renders.
pub const TEMPLATE_EFFORT_LOW: u8 = 25;
/// Numeric budget for "high", the encoder's default.
pub const TEMPLATE_EFFORT_HIGH: u8 = 50;
/// Numeric budget for "xhigh".
pub const TEMPLATE_EFFORT_XHIGH: u8 = 75;
/// Numeric budget for "max".
pub const TEMPLATE_EFFORT_MAX: u8 = 100;

/// The default template budget, "high", used when the request does not name an
/// effort.
pub const TEMPLATE_EFFORT_DEFAULT: u8 = TEMPLATE_EFFORT_HIGH;

/// Why the tokenizer seam could not produce prompt ids.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerSeamError {
    /// No messages array was supplied where the chat encoder needs one.
    #[error("\"messages\" must be a JSON array")]
    MissingMessages,
    /// The chat conversation was empty (the HTTP handler already refuses this).
    #[error("\"messages\" must be a non-empty array")]
    EmptyMessages,
    /// One entry of the messages array is not a JSON object.
    #[error("messages[{index}] must be a JSON object")]
    MessageNotObject {
        /// Offending index.
        index: usize,
    },
    /// The role is outside the set vLLM's normaliser accepts.
    ///
    /// The text matches vllm/tokenizers/deepseek_v41.py exactly.
    #[error("Invalid role: {role}")]
    InvalidRole {
        /// Rendered role value (Python-style None when the key is absent).
        role: String,
    },
    /// A content-part type outside the text and image set.
    ///
    /// The text matches vllm/tokenizers/deepseek_v41.py exactly.
    #[error("DeepSeek V4.1 supports text and image content only; got {part}")]
    UnsupportedContentPart {
        /// Python repr-style rendering of the offending part type.
        part: String,
    },
    /// Message content was neither a string nor an array of parts.
    #[error("message content must be a string or an array of content parts")]
    InvalidContent,
    /// chat_template_kwargs was not an object.
    #[error("\"chat_template_kwargs\" must be a JSON object")]
    InvalidChatTemplateKwargs,
    /// tools was not an array of OpenAI function tools.
    #[error("\"tools\" must be an array of OpenAI function tools")]
    InvalidTools,
    /// tool_calls was not an array of OpenAI tool calls.
    #[error("assistant message \"tool_calls\" must be an array of OpenAI tool calls")]
    InvalidToolCalls,
    /// The requested reasoning effort is outside the encoder's accepted set.
    ///
    /// The text matches vllm/tokenizers/deepseek_v41.py exactly (that file
    /// raises this message before the encoder does).
    #[error(
        "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, or an integer within [1, 100] in chat_template_kwargs"
    )]
    InvalidReasoningEffort,
    /// The official encoder refused the message set.
    #[error(transparent)]
    Render(#[from] CaseError),
    /// The tokenizer itself failed.
    #[error("the tokenizer could not encode the prompt: {0}")]
    Tokenize(String),
}

/// Real prompt tokenization plus the V4.1 chat template, attached to the
/// router by sparkflash serve.
///
/// The seam is optional in crate::AppState so weight-free router tests keep the
/// Milestone 1 placeholder path; a production serve always attaches one.
pub struct TokenizerSeam {
    tokenizer: HfTokenizerEncoder,
    /// The checkpoint tokenizer without the image placeholder among its added
    /// tokens, for chat prompts: the placeholder's spelling in client text
    /// tokenizes as the text it is, and only image parts become its id.
    chat_tokenizer: tokenizers::Tokenizer,
    /// The image placeholder's id, when the vocabulary has one.
    image_id: Option<u32>,
}

impl std::fmt::Debug for TokenizerSeam {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TokenizerSeam")
            .field(
                "vocabulary",
                &self.tokenizer.tokenizer().get_vocab_size(true),
            )
            .finish()
    }
}

impl TokenizerSeam {
    /// Wrap an already loaded checkpoint tokenizer.
    #[must_use]
    pub fn new(tokenizer: HfTokenizerEncoder) -> Self {
        let image_id = tokenizer.tokenizer().token_to_id(IMAGE_PLACEHOLDER);
        let chat_tokenizer = without_added_token(tokenizer.tokenizer(), IMAGE_PLACEHOLDER)
            .unwrap_or_else(|| tokenizer.tokenizer().clone());
        Self {
            tokenizer,
            chat_tokenizer,
            image_id,
        }
    }

    /// Load the seam from a tokenizer.json path.
    ///
    /// # Errors
    ///
    /// Propagates the loader's error, which names the path.
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self::new(load_hf_tokenizer(path)?))
    }

    /// Tokenize raw completion prompt text the way vLLM tokenizes it.
    ///
    /// add_special_tokens=false; for this tokenizer's ByteLevel post-processor
    /// that is also what add_special_tokens=true would yield, and the released
    /// tokenizer_config.json adds neither BOS nor EOS.
    ///
    /// # Errors
    ///
    /// TokenizerSeamError::Tokenize when the tokenizer fails.
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>, TokenizerSeamError> {
        self.encode_text_with(text, false)
    }

    /// Tokenize raw text with the caller's `add_special_tokens` choice.
    ///
    /// This is what `POST /tokenize` exposes (vLLM's `TokenizeCompletionRequest.add_special_tokens`,
    /// default true). For this checkpoint's ByteLevel post-processor the flag does not change the
    /// ids - the released `tokenizer_config.json` adds neither BOS nor EOS - but the flag still has
    /// to reach the tokenizer instead of being assumed, because a different checkpoint may add
    /// them and the route exists to report exactly what the tokenizer does.
    ///
    /// # Errors
    ///
    /// TokenizerSeamError::Tokenize when the tokenizer fails.
    pub fn encode_text_with(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerSeamError> {
        self.tokenizer
            .tokenizer()
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| TokenizerSeamError::Tokenize(error.to_string()))
    }

    /// The string spelling of one vocabulary id, as `POST /tokenize` reports it in `token_strs`.
    ///
    /// `None` for an id outside the vocabulary, which the route renders as JSON null like vLLM.
    #[must_use]
    pub fn token_str(&self, id: u32) -> Option<String> {
        self.tokenizer.tokenizer().id_to_token(id)
    }

    /// The wrapped checkpoint tokenizer, for token-level work.
    #[must_use]
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.tokenizer.tokenizer()
    }

    /// Decode ids back to text with the checkpoint tokenizer.
    ///
    /// Returns None when decoding fails; used for echo-style round trips.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> Option<String> {
        self.tokenizer.tokenizer().decode(ids, false).ok()
    }

    /// Render and tokenize a chat-completion request body.
    ///
    /// request is the raw JSON body (before typed deserialisation), because the
    /// vLLM wrapper's rules depend on fields the named structs do not all keep
    /// (for example arbitrary chat_template_kwargs).
    ///
    /// Each image part becomes the placeholder's id; the placeholder's spelling
    /// in any text of the request is tokenized as text (see the module docs).
    ///
    /// # Errors
    ///
    /// A typed TokenizerSeamError for every refusal path; callers map it to an
    /// OpenAI-shaped 400.
    pub fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, TokenizerSeamError> {
        let messages_value = request
            .get("messages")
            .ok_or(TokenizerSeamError::MissingMessages)?;
        let messages_array = messages_value
            .as_array()
            .ok_or(TokenizerSeamError::MissingMessages)?;
        if messages_array.is_empty() {
            return Err(TokenizerSeamError::EmptyMessages);
        }

        let image_text = image_marker(request);
        let mut messages = Vec::with_capacity(messages_array.len());
        for (index, message) in messages_array.iter().enumerate() {
            messages.push(normalize_message(index, message, &image_text)?);
        }

        attach_tools(&mut messages, request.get("tools"))?;

        let kwargs = apply_chat_template_kwargs(request)?;
        let thinking = resolve_thinking(&kwargs);
        let drop_thinking = kwargs.get("drop_thinking").is_none_or(python_truthy);
        let effort = resolve_effort(&kwargs, thinking)?;

        let mode = if effort.thinking {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        };
        let prompt = encode_messages_text(
            &messages,
            mode,
            None,
            drop_thinking,
            true,
            Some(effort.score),
        )?;
        let mut ids = Vec::new();
        for (index, text) in prompt.split(image_text.as_str()).enumerate() {
            if index > 0 {
                match self.image_id {
                    Some(id) => ids.push(id),
                    None => ids.extend(self.encode_text(IMAGE_PLACEHOLDER)?),
                }
            }
            let encoding = self
                .chat_tokenizer
                .encode(text, false)
                .map_err(|error| TokenizerSeamError::Tokenize(error.to_string()))?;
            ids.extend_from_slice(encoding.get_ids());
        }
        Ok(ids)
    }
}

/// `tokenizer` without `content` among its added tokens, so that spelling
/// tokenizes as ordinary text; None when it is not an added token.
fn without_added_token(
    tokenizer: &tokenizers::Tokenizer,
    content: &str,
) -> Option<tokenizers::Tokenizer> {
    let mut json: Value = serde_json::from_str(&tokenizer.to_string(false).ok()?).ok()?;
    let added = json.get_mut("added_tokens")?.as_array_mut()?;
    let count = added.len();
    added.retain(|token| token.get("content").and_then(Value::as_str) != Some(content));
    if added.len() == count {
        return None;
    }
    json.to_string().parse().ok()
}

/// What the template renders an image part as until the prompt is tokenized:
/// a number between two private-use characters, the first such marker that
/// occurs nowhere in the request. Every occurrence in the rendered prompt is
/// then an image part's, and the placeholder's spelling in client text stays
/// text.
fn image_marker(request: &Value) -> String {
    let body = request.to_string();
    (0_u64..)
        .map(|n| format!("\u{e000}{n}\u{e001}"))
        .find(|marker| !body.contains(marker.as_str()))
        .expect("a finite request leaves a marker free")
}

/// The thinking flag and rendered budget the template will use.
#[derive(Debug, Clone, Copy)]
struct ResolvedEffort {
    thinking: bool,
    score: u8,
}

/// Python truthiness of a JSON value, matching bool(...) in the wrapper.
fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python repr() for the small set of values the wrapper interpolates.
///
/// The wrapper uses {part_type!r}, so a string renders single-quoted (and None
/// renders as the bare word); matching that keeps the error text byte-identical
/// with the released server's message.
fn python_repr(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_owned(),
        Some(Value::String(text)) => format!("'{text}'"),
        Some(other) => other.to_string(),
    }
}

/// Normalise one raw OpenAI message exactly as _normalize_messages does, with
/// each image part rendered as `image_text` rather than the placeholder.
fn normalize_message(
    index: usize,
    message: &Value,
    image_text: &str,
) -> Result<EncodingMessage, TokenizerSeamError> {
    let object = message
        .as_object()
        .ok_or(TokenizerSeamError::MessageNotObject { index })?;

    let role = object.get("role").cloned();
    let role_text = role.as_ref().map(|value| match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    });
    let role_allowed = matches!(
        role_text.as_deref(),
        Some("system" | "developer" | "user" | "assistant" | "tool")
    );
    if !role_allowed {
        return Err(TokenizerSeamError::InvalidRole {
            role: role_text.unwrap_or_else(|| "None".to_owned()),
        });
    }
    let role = role_text.unwrap_or_default();

    let content = match object.get("content") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(MessageContent::Text(text.clone())),
        Some(Value::Array(parts)) => {
            let mut rendered: Vec<String> = Vec::with_capacity(parts.len());
            for part in parts {
                let part_type = part.get("type");
                match part_type.and_then(Value::as_str) {
                    Some("text") => rendered.push(
                        part.get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    Some("image_url" | "input_image" | "image_pil") => {
                        rendered.push(image_text.to_owned());
                    }
                    _ => {
                        return Err(TokenizerSeamError::UnsupportedContentPart {
                            part: python_repr(part_type),
                        });
                    }
                }
            }
            Some(MessageContent::Text(rendered.join("\n\n")))
        }
        Some(_) => return Err(TokenizerSeamError::InvalidContent),
    };

    let reasoning_content = object
        .get("reasoning")
        .or_else(|| object.get("reasoning_content"))
        .and_then(Value::as_str)
        .map(str::to_owned);

    let tool_calls = match object.get("tool_calls") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            serde_json::from_value::<Vec<OpenAiToolCall>>(value.clone())
                .map_err(|_| TokenizerSeamError::InvalidToolCalls)?,
        ),
    };

    let tool_call_id = object
        .get("tool_call_id")
        .and_then(Value::as_str)
        .map(str::to_owned);

    Ok(EncodingMessage {
        role,
        content,
        content_blocks: None,
        reasoning_content,
        tool_calls,
        tools: None,
        response_format: None,
        task: None,
        wo_eos: false,
        tool_call_id,
        mask: None,
    })
}

/// Attach OpenAI tools to the first system message, inserting an empty system
/// message when the conversation has none (deepseek_v41.py behaviour).
fn attach_tools(
    messages: &mut Vec<EncodingMessage>,
    tools: Option<&Value>,
) -> Result<(), TokenizerSeamError> {
    let Some(tools) = tools else {
        return Ok(());
    };
    if tools.is_null() {
        return Ok(());
    }
    let array = tools.as_array().ok_or(TokenizerSeamError::InvalidTools)?;
    if array.is_empty() {
        return Ok(());
    }
    let parsed: Vec<OpenAiTool> = serde_json::from_value(Value::Array(array.clone()))
        .map_err(|_| TokenizerSeamError::InvalidTools)?;
    let system = messages.iter_mut().find(|message| message.role == "system");
    match system {
        Some(message) => message.tools = Some(parsed),
        None => messages.insert(
            0,
            EncodingMessage {
                role: "system".to_owned(),
                content: Some(MessageContent::Text(String::new())),
                content_blocks: None,
                reasoning_content: None,
                tool_calls: None,
                tools: Some(parsed),
                response_format: None,
                task: None,
                wo_eos: false,
                tool_call_id: None,
                mask: None,
            },
        ),
    }
    Ok(())
}

/// The chat_template_kwargs the vLLM chat protocol hands to
/// apply_chat_template: user kwargs, with the top-level reasoning_effort
/// overriding a user-supplied value (as vLLM's merge_kwargs does), plus the
/// implied enable_thinking flag.
fn apply_chat_template_kwargs(request: &Value) -> Result<Map<String, Value>, TokenizerSeamError> {
    let mut kwargs = match request.get("chat_template_kwargs") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(map)) => map.clone(),
        Some(_) => return Err(TokenizerSeamError::InvalidChatTemplateKwargs),
    };
    let top_level = request.get("reasoning_effort");
    let top_level = top_level.filter(|value| !value.is_null());
    if let Some(effort) = top_level {
        kwargs.insert("reasoning_effort".to_owned(), effort.clone());
        if !kwargs.contains_key("enable_thinking") {
            let enabled = effort.as_str() != Some("none");
            kwargs.insert("enable_thinking".to_owned(), Value::Bool(enabled));
        }
    }
    Ok(kwargs)
}

/// Thinking resolution of deepseek_v41.py: bool(thinking or enable_thinking),
/// and True when neither key is present.
fn resolve_thinking(kwargs: &Map<String, Value>) -> bool {
    let thinking_present = kwargs.contains_key("thinking");
    let enable_present = kwargs.contains_key("enable_thinking");
    if !thinking_present && !enable_present {
        return true;
    }
    let thinking = kwargs.get("thinking").is_some_and(python_truthy);
    let enable = kwargs.get("enable_thinking").is_some_and(python_truthy);
    thinking || enable
}

/// Effort resolution and validation of deepseek_v41.py.
///
/// "none" disables thinking and leaves the budget at the default "high" (the
/// Python wrapper assigns None and then immediately replaces it with "high";
/// chat mode then renders no effort line).
fn resolve_effort(
    kwargs: &Map<String, Value>,
    mut thinking: bool,
) -> Result<ResolvedEffort, TokenizerSeamError> {
    let raw = kwargs
        .get("reasoning_effort")
        .cloned()
        .unwrap_or(Value::Null);
    let effective = if raw == Value::String("none".to_owned()) {
        thinking = false;
        Value::Null
    } else {
        raw
    };
    let score = match &effective {
        Value::Null => TEMPLATE_EFFORT_DEFAULT,
        Value::String(name) => match name.as_str() {
            "low" => TEMPLATE_EFFORT_LOW,
            "high" => TEMPLATE_EFFORT_HIGH,
            "xhigh" => TEMPLATE_EFFORT_XHIGH,
            "max" => TEMPLATE_EFFORT_MAX,
            _ => return Err(TokenizerSeamError::InvalidReasoningEffort),
        },
        Value::Number(number) => {
            let Some(integer) = number.as_i64() else {
                return Err(TokenizerSeamError::InvalidReasoningEffort);
            };
            if !(1..=100).contains(&integer) {
                return Err(TokenizerSeamError::InvalidReasoningEffort);
            }
            u8::try_from(integer).map_err(|_| TokenizerSeamError::InvalidReasoningEffort)?
        }
        _ => return Err(TokenizerSeamError::InvalidReasoningEffort),
    };
    Ok(ResolvedEffort { thinking, score })
}

impl From<TokenizerSeamError> for PromptError {
    fn from(error: TokenizerSeamError) -> Self {
        let message = error.to_string();
        let (param, code) = match &error {
            TokenizerSeamError::MissingMessages
            | TokenizerSeamError::EmptyMessages
            | TokenizerSeamError::MessageNotObject { .. }
            | TokenizerSeamError::InvalidRole { .. }
            | TokenizerSeamError::UnsupportedContentPart { .. }
            | TokenizerSeamError::InvalidContent
            | TokenizerSeamError::InvalidToolCalls => (Some("messages"), Some("invalid_messages")),
            TokenizerSeamError::InvalidChatTemplateKwargs => (
                Some("chat_template_kwargs"),
                Some("invalid_chat_template_kwargs"),
            ),
            TokenizerSeamError::InvalidTools => (Some("tools"), Some("invalid_tools")),
            TokenizerSeamError::InvalidReasoningEffort => {
                (Some("reasoning_effort"), Some("invalid_reasoning_effort"))
            }
            TokenizerSeamError::Render(case) => match case {
                CaseError::InvalidReasoningEffort(_) => {
                    (Some("reasoning_effort"), Some("invalid_reasoning_effort"))
                }
                CaseError::InvalidThinkingMode(_) => {
                    (Some("chat_template_kwargs"), Some("invalid_thinking"))
                }
                CaseError::ImagePlaceholderInContent
                | CaseError::ImagePlaceholderInReasoning
                | CaseError::InvalidImageBlock
                | CaseError::UnknownRole(_)
                | CaseError::InvalidTask(_)
                | CaseError::UnmergedToolMessage
                | CaseError::ToolsWithoutMessages
                | CaseError::InvalidCaseShape
                | CaseError::InvalidJson(_) => (Some("messages"), Some("invalid_messages")),
            },
            TokenizerSeamError::Tokenize(_) => return PromptError::internal(message),
        };
        PromptError::invalid(message, param, code)
    }
}

impl PromptCodec for TokenizerSeam {
    fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, PromptError> {
        TokenizerSeam::encode_chat(self, request).map_err(PromptError::from)
    }

    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        self.encode_text_with(text, add_special_tokens)
            .map_err(PromptError::from)
    }

    fn decode(&self, ids: &[u32]) -> Option<String> {
        TokenizerSeam::decode(self, ids)
    }

    fn token_str(&self, id: u32) -> Option<String> {
        TokenizerSeam::token_str(self, id)
    }

    fn vision_codec(&self) -> Option<std::sync::Arc<dyn sparkflash_model::vision::VisionCodec>> {
        Some(std::sync::Arc::new(*crate::vision::codec()))
    }
}
