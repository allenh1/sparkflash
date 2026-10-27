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

//! DeepSeek-V4.1 prompt rendering with numeric reasoning effort.
//!
//! Rendering is byte-identical with the official Python encoder
//! (reference/encoding/encoding.py). The deepseek-recipe 0.1 crate supplies the
//! V4.1 special tokens, the Conversation/InputMessage types, the tokenizer
//! trait, and the RenderedPrompt/EncodingError types; the message renderer is a
//! faithful port because the published recipe renderer diverges from the
//! official encoder in four observable ways (documented in render.rs).
//!
//! The recipe only renders the effort budgets 50/75/100 (Low/High/Xhigh/Max),
//! so numeric budgets outside that set are applied by rewriting the leading
//! effort line with `retarget_reasoning_effort`.

mod case;
mod render;
mod seam;

pub use case::{
    CaseError, EncodingCase, EncodingMessage, IMAGE_PLACEHOLDER, MessageContent,
    OpenAiFunctionCall, OpenAiTool, OpenAiToolCall, ThinkingMode, load_first_case, parse_cases,
    process_image_messages,
};
pub use render::{DEFAULT_EFFORT_SCORE, render_single_message};
pub use seam::{
    TEMPLATE_EFFORT_DEFAULT, TEMPLATE_EFFORT_HIGH, TEMPLATE_EFFORT_LOW, TEMPLATE_EFFORT_MAX,
    TEMPLATE_EFFORT_XHIGH, TokenizerSeam, TokenizerSeamError,
};

use case::process_image_messages_lenient;

use std::path::{Path, PathBuf};

use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::ImageSource;
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use deepseek_recipe_encoding::v4::{BOS_TOKEN, SYSTEM_SP_TOKEN};
use deepseek_recipe_encoding::{EncodingError, RenderedPrompt, TokenizerEncoder};
use serde_json::{Value, json};

const EFFORT_LINE_SUFFIX: &str =
    " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

/// Rewrite the leading reasoning-effort budget without touching anything else.
///
/// The official encoder renders the effort line at message index 0 only, so the
/// rewrite is anchored to the start of the prompt. The function is a no-op when
/// the prompt has no effort line or the budget already matches.
pub fn retarget_reasoning_effort(prompt: String, score: u8) -> String {
    let prefixes = [
        format!("{BOS_TOKEN}{SYSTEM_SP_TOKEN}Reasoning Effort: "),
        format!("{SYSTEM_SP_TOKEN}Reasoning Effort: "),
    ];
    for prefix in &prefixes {
        let Some(rest) = prompt.strip_prefix(prefix.as_str()) else {
            continue;
        };
        let Some(end) = rest.find(EFFORT_LINE_SUFFIX) else {
            continue;
        };
        let digits = &rest[..end];
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let replacement = score.to_string();
        if digits == replacement {
            return prompt;
        }
        let mut rewritten = String::with_capacity(prompt.len() + replacement.len());
        rewritten.push_str(prefix);
        rewritten.push_str(&replacement);
        rewritten.push_str(EFFORT_LINE_SUFFIX);
        rewritten.push_str(&rest[end + EFFORT_LINE_SUFFIX.len()..]);
        return rewritten;
    }
    prompt
}

/// A problem found while resolving a requested reasoning effort.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReasoningEffortError {
    /// An integer outside 1..=100; the resolved score is clamped into range.
    #[error("reasoning_effort must be an integer in 1..=100, got {value} (clamped)")]
    OutOfRange {
        /// The rejected integer.
        value: i64,
    },
    /// An unrecognized string alias.
    #[error("unknown reasoning_effort alias {0}")]
    UnknownAlias(String),
    /// A JSON type that is neither an integer nor a string.
    #[error("reasoning_effort must be an integer in 1..=100 or one of low/high/xhigh/max")]
    UnsupportedType,
}

/// Reasoning effort resolved from a request value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningEffortSpec {
    /// Whether thinking is enabled for the request.
    pub thinking: bool,
    /// Recipe-compatible level when the request names one.
    pub effort: Option<ReasoningEffort>,
    /// Numeric budget in 1..=100, or None when no effort line is rendered.
    pub score: Option<u8>,
    /// Validation problem; the score is still usable (clamped or defaulted).
    pub error: Option<ReasoningEffortError>,
}

impl ReasoningEffortSpec {
    /// Whether this request renders a reasoning effort line.
    pub fn renders_effort_line(&self) -> bool {
        self.thinking && self.score.is_some()
    }
}

fn default_effort_spec(error: Option<ReasoningEffortError>) -> ReasoningEffortSpec {
    ReasoningEffortSpec {
        thinking: true,
        effort: Some(ReasoningEffort::High),
        score: Some(DEFAULT_EFFORT_SCORE),
        error,
    }
}

/// Resolve the reasoning effort of a request.
///
/// Accepts an integer in 1..=100, the aliases low=25, high/medium=50, xhigh=75,
/// and max=100, or the whole request object (whose reasoning_effort field is used).
/// Absent values default to 50 (`high`) while thinking, matching the released
/// V4.1 encoder. With thinking disabled no effort line is rendered. Out-of-range
/// integers are clamped into 1..=100 and reported through
/// `ReasoningEffortSpec::error`.
pub fn resolve_reasoning_effort(
    request_value: Option<&Value>,
    thinking: bool,
) -> ReasoningEffortSpec {
    if !thinking {
        return ReasoningEffortSpec {
            thinking: false,
            effort: None,
            score: None,
            error: None,
        };
    }
    let Some(value) = request_value else {
        return default_effort_spec(None);
    };
    let value = match value {
        Value::Object(map) => {
            if !resolve_thinking_from_request(value, true) {
                return ReasoningEffortSpec {
                    thinking: false,
                    effort: None,
                    score: None,
                    error: None,
                };
            }
            match map.get("reasoning_effort") {
                Some(inner) => inner,
                None => return default_effort_spec(Some(ReasoningEffortError::UnsupportedType)),
            }
        }
        other => other,
    };
    match value {
        Value::Null => default_effort_spec(None),
        Value::Number(number) => {
            let Some(integer) = number.as_i64() else {
                return default_effort_spec(Some(ReasoningEffortError::UnsupportedType));
            };
            if (1..=100).contains(&integer) {
                ReasoningEffortSpec {
                    thinking: true,
                    effort: None,
                    score: Some(integer as u8),
                    error: None,
                }
            } else {
                ReasoningEffortSpec {
                    thinking: true,
                    effort: None,
                    score: Some(integer.clamp(1, 100) as u8),
                    error: Some(ReasoningEffortError::OutOfRange { value: integer }),
                }
            }
        }
        Value::String(alias) => match alias.as_str() {
            "low" | "minimal" => ReasoningEffortSpec {
                thinking: true,
                effort: Some(ReasoningEffort::Low),
                score: Some(TEMPLATE_EFFORT_LOW),
                error: None,
            },
            "high" | "medium" => ReasoningEffortSpec {
                thinking: true,
                effort: Some(ReasoningEffort::High),
                score: Some(TEMPLATE_EFFORT_HIGH),
                error: None,
            },
            "xhigh" => ReasoningEffortSpec {
                thinking: true,
                effort: Some(ReasoningEffort::Xhigh),
                score: Some(TEMPLATE_EFFORT_XHIGH),
                error: None,
            },
            "max" => ReasoningEffortSpec {
                thinking: true,
                effort: Some(ReasoningEffort::Max),
                score: Some(TEMPLATE_EFFORT_MAX),
                error: None,
            },
            other => {
                default_effort_spec(Some(ReasoningEffortError::UnknownAlias(other.to_owned())))
            }
        },
        _ => default_effort_spec(Some(ReasoningEffortError::UnsupportedType)),
    }
}

/// Resolve whether thinking is enabled from a request body.
///
/// Honours enable_thinking, chat_template_kwargs.thinking, thinking (bool or
/// {type: enabled|disabled|adaptive}), and reasoning_effort "none", in that
/// order, falling back to default.
pub fn resolve_thinking_from_request(request: &Value, default: bool) -> bool {
    if let Some(enabled) = request.get("enable_thinking").and_then(Value::as_bool) {
        return enabled;
    }
    if let Some(enabled) = request
        .pointer("/chat_template_kwargs/thinking")
        .and_then(Value::as_bool)
    {
        return enabled;
    }
    if let Some(thinking) = request.get("thinking") {
        if let Some(enabled) = thinking.as_bool() {
            return enabled;
        }
        if let Some(kind) = thinking.get("type").and_then(Value::as_str) {
            match kind {
                "enabled" | "adaptive" => return true,
                "disabled" => return false,
                _ => {}
            }
        }
    }
    if request.get("reasoning_effort").and_then(Value::as_str) == Some("none") {
        return false;
    }
    default
}

fn python_effort_score(
    value: Option<&Value>,
    thinking: ThinkingMode,
) -> Result<Option<u8>, CaseError> {
    if thinking != ThinkingMode::Thinking {
        return Ok(None);
    }
    let Some(value) = value else {
        return Ok(Some(DEFAULT_EFFORT_SCORE));
    };
    match value {
        Value::Number(number) => {
            let Some(integer) = number.as_i64() else {
                return Err(CaseError::InvalidReasoningEffort(number.to_string()));
            };
            if (1..=100).contains(&integer) {
                Ok(Some(integer as u8))
            } else {
                Err(CaseError::InvalidReasoningEffort(integer.to_string()))
            }
        }
        Value::String(alias) => match alias.as_str() {
            "low" => Ok(Some(25)),
            "high" => Ok(Some(50)),
            "xhigh" => Ok(Some(75)),
            "max" => Ok(Some(100)),
            other => Err(CaseError::InvalidReasoningEffort(other.to_owned())),
        },
        other => Err(CaseError::InvalidReasoningEffort(other.to_string())),
    }
}

fn recipe_effort_score(conversation: &Conversation) -> Option<u8> {
    if !conversation.thinking_mode {
        return None;
    }
    Some(match conversation.reasoning_effort {
        Some(ReasoningEffort::Low) => TEMPLATE_EFFORT_LOW,
        Some(ReasoningEffort::Max) => TEMPLATE_EFFORT_MAX,
        Some(ReasoningEffort::Xhigh) => TEMPLATE_EFFORT_XHIGH,
        None | Some(ReasoningEffort::High) => TEMPLATE_EFFORT_HIGH,
    })
}

fn tool_call_to_open_ai(call: &ToolCall) -> OpenAiToolCall {
    OpenAiToolCall {
        id: (!call.id.is_empty()).then(|| call.id.clone()),
        function: OpenAiFunctionCall {
            name: call.name.clone(),
            id: None,
            arguments: Value::String(call.arguments.clone()),
        },
    }
}

fn tool_definition_to_open_ai(tool: &ToolDefinition) -> OpenAiTool {
    let mut function = serde_json::Map::new();
    function.insert("name".to_owned(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        function.insert("description".to_owned(), Value::String(description.clone()));
    }
    function.insert("parameters".to_owned(), tool.parameters.clone());
    OpenAiTool {
        function: Value::Object(function),
    }
}

fn input_message_to_encoding(message: &InputMessage) -> EncodingMessage {
    match message {
        InputMessage::System { content } => EncodingMessage {
            role: "system".to_owned(),
            content: Some(MessageContent::Text(content.clone())),
            ..EncodingMessage::default()
        },
        InputMessage::User { content, .. } => EncodingMessage {
            role: "user".to_owned(),
            content: Some(MessageContent::Text(content.clone())),
            ..EncodingMessage::default()
        },
        InputMessage::Assistant {
            content,
            reasoning_content,
            tool_calls,
        } => EncodingMessage {
            role: "assistant".to_owned(),
            content: Some(MessageContent::Text(content.clone())),
            reasoning_content: reasoning_content.clone(),
            tool_calls: tool_calls
                .as_ref()
                .map(|calls| calls.iter().map(tool_call_to_open_ai).collect()),
            ..EncodingMessage::default()
        },
        InputMessage::Tool {
            content,
            tool_call_id,
            ..
        } => EncodingMessage {
            role: "tool".to_owned(),
            content: Some(MessageContent::Text(content.clone())),
            tool_call_id: Some(tool_call_id.clone()),
            ..EncodingMessage::default()
        },
        InputMessage::LatestReminder { content } => EncodingMessage {
            role: "latest_reminder".to_owned(),
            content: Some(MessageContent::Text(content.clone())),
            ..EncodingMessage::default()
        },
    }
}

fn conversation_to_case(conversation: &Conversation) -> EncodingCase {
    let mut messages: Vec<EncodingMessage> = conversation
        .messages
        .iter()
        .map(input_message_to_encoding)
        .collect();

    let has_tools = conversation.tool_choice != ToolChoice::None && !conversation.tools.is_empty();
    let format_schema = match conversation.response_format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(json!({ "type": "json_object" })),
    };
    if has_tools || format_schema.is_some() {
        if !matches!(
            messages.first().map(|message| message.role.as_str()),
            Some("system")
        ) {
            messages.insert(
                0,
                EncodingMessage {
                    role: "system".to_owned(),
                    ..EncodingMessage::default()
                },
            );
        }
        if let Some(first) = messages.first_mut() {
            if has_tools {
                first.tools = Some(
                    conversation
                        .tools
                        .iter()
                        .map(tool_definition_to_open_ai)
                        .collect(),
                );
            }
            if let Some(schema) = format_schema {
                first.response_format = Some(schema);
            }
        }
    }

    EncodingCase {
        messages,
        context: None,
        thinking_mode: Some(
            if conversation.thinking_mode {
                "thinking"
            } else {
                "chat"
            }
            .to_owned(),
        ),
        reasoning_effort: None,
        tools: None,
    }
}

fn conversation_image_sources(conversation: &Conversation) -> Vec<ImageSource> {
    conversation
        .messages
        .iter()
        .filter_map(|message| match message {
            InputMessage::User { image_sources, .. } | InputMessage::Tool { image_sources, .. } => {
                Some(image_sources.as_slice())
            }
            _ => None,
        })
        .flatten()
        .cloned()
        .collect()
}

/// DeepSeek-V4.1 prompt renderer and token encoder.
pub struct SparkV41Encoding {
    tokenizer: Option<Box<dyn TokenizerEncoder>>,
}

impl SparkV41Encoding {
    /// Create an encoding without an attached tokenizer.
    pub fn new() -> Self {
        Self { tokenizer: None }
    }

    /// Attach the tokenizer used by `SparkV41Encoding::encode`.
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: impl TokenizerEncoder + 'static) -> Self {
        self.tokenizer = Some(Box::new(tokenizer));
        self
    }

    /// Render a conversation and the prefix for the next assistant turn.
    ///
    /// When effort is Some, the leading effort line is retargeted to that
    /// numeric budget; otherwise the recipe-compatible mapping is kept.
    pub fn render_conversation(
        &self,
        conversation: &Conversation,
        effort: Option<u8>,
    ) -> RenderedPrompt {
        let case = conversation_to_case(conversation);
        let thinking = if conversation.thinking_mode {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        };
        let base = recipe_effort_score(conversation);
        let rendered = self
            .render_case_internal(&case, thinking, base, false)
            .expect("conversation-derived cases are always renderable");
        let prompt = match effort {
            Some(score) => retarget_reasoning_effort(rendered.prompt, score),
            None => rendered.prompt,
        };
        RenderedPrompt {
            prompt,
            image_sources: conversation_image_sources(conversation),
        }
    }

    /// Encode a conversation into token ids with the attached tokenizer.
    ///
    /// # Errors
    ///
    /// Returns `EncodingError::MissingTokenizer` when no tokenizer was
    /// attached and `EncodingError::Encode` when tokenization fails.
    pub fn encode(
        &self,
        conversation: &Conversation,
        effort: Option<u8>,
    ) -> Result<Vec<u32>, EncodingError> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(EncodingError::MissingTokenizer)?;
        let rendered = self.render_conversation(conversation, effort);
        tokenizer
            .encode_ids(&rendered.prompt)
            .map_err(EncodingError::Encode)
    }

    /// Render a parsed case using the official encoder semantics.
    ///
    /// `RenderedPrompt::image_sources` holds the CURRENT TURN's images only.
    /// The reference discards the context's media
    /// (`processed_context, _ = process_image_messages(context)`) and never
    /// renders context messages into the prompt, so context images contribute
    /// neither tokens nor entries to this list.
    ///
    /// # Errors
    ///
    /// Returns `CaseError` when the case is malformed or uses an unsupported
    /// reasoning effort.
    pub fn render_case(
        &self,
        case: &EncodingCase,
        effort: Option<u8>,
    ) -> Result<RenderedPrompt, CaseError> {
        let thinking = case.thinking_mode()?;
        let base = python_effort_score(case.reasoning_effort.as_ref(), thinking)?;
        let mut rendered = self.render_case_internal(case, thinking, base, true)?;
        if let Some(score) = effort {
            rendered.prompt = retarget_reasoning_effort(rendered.prompt, score);
        }
        Ok(rendered)
    }

    fn render_case_internal(
        &self,
        case: &EncodingCase,
        thinking: ThinkingMode,
        effort_score: Option<u8>,
        validate_placeholders: bool,
    ) -> Result<RenderedPrompt, CaseError> {
        let mut messages = case.messages.clone();
        if let Some(tools) = &case.tools {
            if messages.is_empty() {
                return Err(CaseError::ToolsWithoutMessages);
            }
            messages[0].tools = Some(tools.clone());
        }
        let (messages, images) = if validate_placeholders {
            process_image_messages(&messages)?
        } else {
            process_image_messages_lenient(&messages)?
        };
        // The context is processed for validation (and for the placeholder
        // rewriting the reference performs), but its media is discarded:
        // encode_messages returns only the current turn's images.
        let context = match case.context.as_ref().filter(|c| !c.is_empty()) {
            Some(context) => {
                let (context, _context_images) = if validate_placeholders {
                    process_image_messages(context)?
                } else {
                    process_image_messages_lenient(context)?
                };
                Some(context)
            }
            None => None,
        };
        let prompt = render::encode_messages_text(
            &messages,
            thinking,
            context.as_deref(),
            true,
            true,
            effort_score,
        )?;
        Ok(RenderedPrompt {
            prompt,
            image_sources: images,
        })
    }
}

impl Default for SparkV41Encoding {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SparkV41Encoding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SparkV41Encoding")
            .field("has_tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}

/// Tokenizer adapter over the HuggingFace tokenizers crate.
pub struct HfTokenizerEncoder {
    tokenizer: tokenizers::Tokenizer,
}

impl HfTokenizerEncoder {
    /// Wrap a loaded tokenizer.
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Borrow the wrapped tokenizer.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }
}

impl TokenizerEncoder for HfTokenizerEncoder {
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }
}

/// Load a HuggingFace tokenizer.json from disk.
///
/// # Errors
///
/// Returns an error when the file is missing or not a valid tokenizer.
pub fn load_hf_tokenizer(path: &Path) -> anyhow::Result<HfTokenizerEncoder> {
    let tokenizer = tokenizers::Tokenizer::from_file(path)
        .map_err(|error| anyhow::anyhow!("failed to load tokenizer {}: {error}", path.display()))?;
    Ok(HfTokenizerEncoder::new(tokenizer))
}

/// The vendored tokenizer's location relative to a deployed directory.
const VENDORED_TOKENIZER: &str = "reference/tokenizer.json";

/// The vendored reference tokenizer's compile-time path in this checkout.
///
/// This is a development convenience: it embeds `CARGO_MANIFEST_DIR`, so it is only meaningful
/// for a test or binary running from the checkout that built it. A binary built in one directory
/// and copied elsewhere must resolve [`resolve_reference_tokenizer`] instead, or be given
/// `--tokenizer`; the M2 dev-window run of 2026-10-18 failed because a default based on this path
/// was carried to a node where that build directory does not exist.
pub fn reference_tokenizer_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/tokenizer.json")
}

/// Resolves the vendored reference tokenizer for a binary that may not run from this checkout.
///
/// `--tokenizer` is the operator's explicit override and is never routed through this function.
/// Otherwise the file is looked for, in order, at `<exe_dir>/reference/tokenizer.json` and
/// `<cwd>/reference/tokenizer.json`, which covers a deployed binary with the directory staged
/// next to it and one started from the stage root. The compile-time `CARGO_MANIFEST_DIR` is
/// deliberately not a candidate: it names a directory on the build host, not on the node.
///
/// # Errors
///
/// A message naming `--tokenizer` and every candidate that was tried, so the operator can stage
/// the file or pass the flag instead of guessing which default failed.
pub fn resolve_reference_tokenizer(
    exe_dir: Option<&Path>,
    cwd: Option<&Path>,
) -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    for root in [exe_dir, cwd].into_iter().flatten() {
        let candidate = root.join(VENDORED_TOKENIZER);
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    if let Some(candidate) = candidates.iter().find(|path| path.is_file()) {
        return Ok(candidate.clone());
    }
    if candidates.is_empty() {
        return Err(format!(
            "no default tokenizer found: --tokenizer was not given and {VENDORED_TOKENIZER} cannot be looked up because neither the executable directory nor the current directory is available; pass --tokenizer FILE (the model's own tokenizer.json)"
        ));
    }
    let tried = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "no default tokenizer found: --tokenizer was not given and {VENDORED_TOKENIZER} is not at {tried}; pass --tokenizer FILE (the model's own tokenizer.json)"
    ))
}
