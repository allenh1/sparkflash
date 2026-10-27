//! Case JSON model and preprocessing mirroring the official Python encoder
//! (reference/encoding/encoding.py).
//!
//! The Python reference accepts loosely-typed message dictionaries. This module
//! mirrors that schema with serde so a golden vector can be parsed and rendered
//! without re-implementing the loader in every test.

use deepseek_recipe_core::multimodal::{ImageDetail, ImageSource, is_http_url};
use serde::Deserialize;
use serde_json::{Value, json};

/// Placeholder the official V4.1 encoder writes into prompt text for each image.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// Prompt thinking mode. "chat" closes the thinking block immediately;
/// "thinking" asks the model to reason first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingMode {
    /// No reasoning is requested.
    #[default]
    Chat,
    /// Reasoning is requested and the effort line is rendered.
    Thinking,
}

impl ThinkingMode {
    /// Parse the thinking_mode case field, defaulting to chat.
    ///
    /// # Errors
    ///
    /// Returns `CaseError::InvalidThinkingMode` for an unknown mode.
    pub fn parse(value: Option<&str>) -> Result<Self, CaseError> {
        match value {
            None | Some("chat") => Ok(Self::Chat),
            Some("thinking") => Ok(Self::Thinking),
            Some(other) => Err(CaseError::InvalidThinkingMode(other.to_owned())),
        }
    }
}

/// A failure while parsing or preprocessing a case.
#[derive(Debug, thiserror::Error)]
pub enum CaseError {
    /// thinking_mode is neither chat nor thinking.
    #[error("invalid thinking_mode {0} (expected chat or thinking)")]
    InvalidThinkingMode(String),
    /// User text already contains the image placeholder.
    #[error("message content contains the image special token")]
    ImagePlaceholderInContent,
    /// Assistant reasoning already contains the image placeholder.
    #[error("reasoning_content contains the image special token")]
    ImagePlaceholderInReasoning,
    /// An image block carries no usable source.
    #[error("image block does not contain a valid source")]
    InvalidImageBlock,
    /// A message role the reference encoder does not render.
    #[error("unknown role: {0}")]
    UnknownRole(String),
    /// A task field outside the supported quick-instruction set.
    #[error("invalid task {0}")]
    InvalidTask(String),
    /// A raw tool message reached rendering without preprocessing.
    #[error("deepseek_v41 merges tool messages into user messages; preprocess first")]
    UnmergedToolMessage,
    /// A case-level tools list was supplied without messages.
    #[error("a case with tools must contain at least one message")]
    ToolsWithoutMessages,
    /// A reasoning effort outside 1..=100 or the alias set.
    #[error("invalid reasoning effort {0}: expected an integer in 1..=100 or low/high/xhigh/max")]
    InvalidReasoningEffort(String),
    /// The top-level JSON is neither a case object nor a message list.
    #[error("case JSON must be an object with messages or an array of messages")]
    InvalidCaseShape,
    /// The top-level JSON could not be decoded.
    #[error("invalid case JSON: {0}")]
    InvalidJson(String),
}

/// Message content, which is either a plain string or a block list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Content blocks (text, image_url, tool_result, ...).
    Blocks(Vec<Value>),
}

impl MessageContent {
    /// Text of this content, or the empty string for a block list.
    pub fn as_text(&self) -> &str {
        match self {
            Self::Text(text) => text,
            Self::Blocks(_) => "",
        }
    }
}

/// One OpenAI-format tool definition. Only function is rendered.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiTool {
    /// The function schema rendered into the system prompt.
    pub function: Value,
}

/// One OpenAI-format tool call attached to an assistant message.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiToolCall {
    /// Tool call id, used to order tool results.
    #[serde(default)]
    pub id: Option<String>,
    /// Function name and arguments.
    pub function: OpenAiFunctionCall,
}

/// The function object of a tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiFunctionCall {
    /// Function name.
    pub name: String,
    /// Some clients nest the tool call id here.
    #[serde(default)]
    pub id: Option<String>,
    /// Arguments, normally a JSON-encoded string but tolerated as any value.
    #[serde(default)]
    pub arguments: Value,
}

/// One message in the Python reference schema.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct EncodingMessage {
    /// system, user, assistant, tool, or latest_reminder.
    pub role: String,
    /// Message content.
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Pre-parsed content blocks, in reference order.
    #[serde(default)]
    pub content_blocks: Option<Vec<Value>>,
    /// Assistant reasoning content.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// OpenAI-format tool calls.
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    /// OpenAI-format tool definitions carried by a system message.
    #[serde(default)]
    pub tools: Option<Vec<OpenAiTool>>,
    /// Structured response format carried by a system message.
    #[serde(default)]
    pub response_format: Option<Value>,
    /// Quick-instruction task token (action, query, ...).
    #[serde(default)]
    pub task: Option<String>,
    /// Suppress the end-of-sequence token after an assistant message.
    #[serde(default)]
    pub wo_eos: bool,
    /// Tool result correlation id.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Accepted and ignored, as in the reference encoder.
    #[serde(default)]
    pub mask: Option<Value>,
}

/// One loaded case: messages plus call-time configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct EncodingCase {
    /// Conversation messages.
    #[serde(default)]
    pub messages: Vec<EncodingMessage>,
    /// Optional prior conversation rendered before the current messages.
    #[serde(default)]
    pub context: Option<Vec<EncodingMessage>>,
    /// chat or thinking; absent means chat.
    #[serde(default)]
    pub thinking_mode: Option<String>,
    /// Numeric or alias reasoning effort.
    #[serde(default)]
    pub reasoning_effort: Option<Value>,
    /// Case-level OpenAI tool definitions, attached to the first message.
    #[serde(default)]
    pub tools: Option<Vec<OpenAiTool>>,
}

impl EncodingCase {
    /// Thinking mode for this case, defaulting to chat.
    ///
    /// # Errors
    ///
    /// Returns `CaseError::InvalidThinkingMode` for an unknown mode.
    pub fn thinking_mode(&self) -> Result<ThinkingMode, CaseError> {
        ThinkingMode::parse(self.thinking_mode.as_deref())
    }
}

fn deserialize_case(value: &Value) -> Result<EncodingCase, CaseError> {
    serde_json::from_value(value.clone()).map_err(|error| CaseError::InvalidJson(error.to_string()))
}

/// Parse the top-level JSON of a case file into one or more `EncodingCase`
/// values, mirroring the reference load_cases.
///
/// # Errors
///
/// Returns `CaseError` when the JSON shape is unsupported or a case does not
/// match the message schema.
pub fn parse_cases(value: &Value) -> Result<Vec<EncodingCase>, CaseError> {
    let raw: Vec<Value> = match value {
        Value::Object(_) => vec![value.clone()],
        Value::Array(items) => {
            if items.first().and_then(|item| item.get("role")).is_some() {
                vec![json!({ "messages": items })]
            } else {
                items.clone()
            }
        }
        _ => return Err(CaseError::InvalidCaseShape),
    };

    let mut cases = Vec::with_capacity(raw.len());
    for mut case in raw {
        if let Some(tools) = case.get("tools").cloned() {
            let messages = case
                .get_mut("messages")
                .and_then(Value::as_array_mut)
                .ok_or(CaseError::InvalidCaseShape)?;
            if messages.is_empty() {
                return Err(CaseError::ToolsWithoutMessages);
            }
            if let Some(first) = messages[0].as_object_mut() {
                first.insert("tools".to_owned(), tools);
            }
        }
        cases.push(deserialize_case(&case)?);
    }
    Ok(cases)
}

/// Parse the first case in a case file body.
///
/// # Errors
///
/// Returns `CaseError::InvalidJson` for malformed JSON and propagates
/// `parse_cases` errors.
pub fn load_first_case(json: &str) -> Result<EncodingCase, CaseError> {
    let value: Value =
        serde_json::from_str(json).map_err(|error| CaseError::InvalidJson(error.to_string()))?;
    let mut cases = parse_cases(&value)?;
    if cases.is_empty() {
        return Err(CaseError::InvalidCaseShape);
    }
    Ok(cases.remove(0))
}

fn validate_no_image_placeholder(message: &EncodingMessage) -> Result<(), CaseError> {
    if let Some(MessageContent::Text(content)) = &message.content
        && content.contains(IMAGE_PLACEHOLDER)
    {
        return Err(CaseError::ImagePlaceholderInContent);
    }
    if let Some(reasoning) = &message.reasoning_content
        && reasoning.contains(IMAGE_PLACEHOLDER)
    {
        return Err(CaseError::ImagePlaceholderInReasoning);
    }
    Ok(())
}

fn image_source_from_url(url: String) -> ImageSource {
    if is_http_url(&url) {
        ImageSource::Url {
            url,
            detail: ImageDetail::High,
        }
    } else {
        ImageSource::DataUrl {
            data_url: url,
            detail: ImageDetail::High,
        }
    }
}

fn extract_image(block: &Value) -> Result<ImageSource, CaseError> {
    let object = block.as_object().ok_or(CaseError::InvalidImageBlock)?;
    let url = if object.get("type").and_then(Value::as_str) == Some("image_url") {
        match object.get("image_url") {
            Some(Value::String(url)) => Some(url.clone()),
            Some(Value::Object(image_url)) => image_url
                .get("url")
                .and_then(Value::as_str)
                .map(str::to_owned),
            _ => None,
        }
    } else {
        ["source", "url", "data"]
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_str))
            .map(str::to_owned)
    };
    let url = url
        .filter(|url| !url.is_empty())
        .ok_or(CaseError::InvalidImageBlock)?;
    Ok(image_source_from_url(url))
}

fn process_image_blocks(blocks: &[Value]) -> Result<(Vec<Value>, Vec<ImageSource>), CaseError> {
    let mut new_blocks = Vec::with_capacity(blocks.len());
    let mut images = Vec::new();
    for block in blocks {
        let Some(object) = block.as_object() else {
            new_blocks.push(block.clone());
            continue;
        };
        match object.get("type").and_then(Value::as_str) {
            Some("image" | "image_url") => {
                new_blocks.push(json!({ "type": "text", "text": IMAGE_PLACEHOLDER }));
                images.push(extract_image(block)?);
            }
            Some("tool_result") => {
                let mut block = block.clone();
                if let Some(content) = object.get("content").and_then(Value::as_array) {
                    let (nested, nested_images) = process_image_blocks(content)?;
                    block["content"] = Value::Array(nested);
                    images.extend(nested_images);
                }
                new_blocks.push(block);
            }
            Some("text") => {
                let text = object
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if text.contains(IMAGE_PLACEHOLDER) {
                    return Err(CaseError::ImagePlaceholderInContent);
                }
                new_blocks.push(block.clone());
            }
            _ => new_blocks.push(block.clone()),
        }
    }
    Ok((new_blocks, images))
}

/// Normalize image blocks, mirroring the reference process_image_messages.
///
/// Returns the processed messages together with the image sources in prompt
/// order.
///
/// # Errors
///
/// Returns `CaseError` when a message carries the image placeholder in text
/// or an image block has no usable source.
pub fn process_image_messages(
    messages: &[EncodingMessage],
) -> Result<(Vec<EncodingMessage>, Vec<ImageSource>), CaseError> {
    process_image_messages_inner(messages, true)
}

/// Like process_image_messages but treats content as already-flattened prompt
/// text, so an image placeholder is not rejected. Conversation input arrives
/// with its placeholders already resolved by the protocol adapter.
pub(crate) fn process_image_messages_lenient(
    messages: &[EncodingMessage],
) -> Result<(Vec<EncodingMessage>, Vec<ImageSource>), CaseError> {
    process_image_messages_inner(messages, false)
}

fn process_image_messages_inner(
    messages: &[EncodingMessage],
    validate_placeholders: bool,
) -> Result<(Vec<EncodingMessage>, Vec<ImageSource>), CaseError> {
    let mut processed = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    for message in messages {
        let mut message = message.clone();
        if validate_placeholders {
            validate_no_image_placeholder(&message)?;
        }

        if message.content_blocks.is_none() {
            match message.content.take() {
                Some(MessageContent::Blocks(blocks)) => message.content_blocks = Some(blocks),
                other => message.content = other,
            }
        }

        if let Some(blocks) = message
            .content_blocks
            .as_ref()
            .filter(|blocks| !blocks.is_empty())
        {
            let (new_blocks, message_images) = process_image_blocks(blocks)?;
            let needs_joined_text = !matches!(message.content, Some(MessageContent::Text(_)));
            message.content_blocks = Some(new_blocks);
            images.extend(message_images);
            if needs_joined_text {
                let texts = message
                    .content_blocks
                    .as_ref()
                    .map(|blocks| {
                        blocks
                            .iter()
                            .filter(|block| {
                                block.get("type").and_then(Value::as_str) == Some("text")
                            })
                            .map(|block| {
                                block
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_owned()
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                message.content = Some(MessageContent::Text(texts.join("\n\n")));
            }
        }
        processed.push(message);
    }
    Ok((processed, images))
}
