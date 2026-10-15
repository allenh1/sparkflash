//! OpenAI and Anthropic request schemas accepted by the router.
//!
//! The types only describe what the protocol layer understands; validation (reasoning
//! effort ranges, model existence, tool shape) happens in the handlers so that every
//! failure can be rendered as an OpenAI-shaped error envelope.

use serde::Deserialize;
use serde_json::Value;

/// OpenAI `POST /v1/chat/completions` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Requested model id.
    pub model: Option<String>,
    /// Conversation messages.
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    /// Whether to stream server-sent events.
    #[serde(default)]
    pub stream: bool,
    /// Streaming options (`include_usage`).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Legacy completion cap.
    pub max_tokens: Option<usize>,
    /// Current completion cap; wins over `max_tokens`.
    pub max_completion_tokens: Option<usize>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Tool definitions (accepted verbatim in Milestone 1).
    pub tools: Option<Value>,
    /// Stop sequence(s).
    #[serde(default)]
    pub stop: Option<StopSpec>,
    /// Deterministic sampling seed.
    pub seed: Option<u64>,
    /// Raw reasoning effort, validated by crate::reasoning.
    pub reasoning_effort: Option<Value>,
    /// Whether thinking is enabled.
    pub enable_thinking: Option<bool>,
    /// Chat-template kwargs (`thinking` lives here).
    pub chat_template_kwargs: Option<Value>,
    /// Number of choices requested; only 1 is supported.
    pub n: Option<usize>,
    /// End-user identifier.
    pub user: Option<String>,
}

/// One chat message.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// String or content-part array.
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Optional participant name.
    pub name: Option<String>,
    /// Assistant tool calls echoed back into the context.
    pub tool_calls: Option<Value>,
    /// Id of the tool call this message answers.
    pub tool_call_id: Option<String>,
}

/// Chat message content: either a plain string or a content-part array.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Multimodal content parts.
    Parts(Vec<ContentPart>),
}

/// One content part of a multimodal message.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// OpenAI/Anthropic text part.
    Text {
        /// The text.
        text: String,
    },
    /// OpenAI Responses text part.
    InputText {
        /// The text.
        text: String,
    },
    /// OpenAI chat image part.
    ImageUrl {
        /// Image URL or base64 data URL (accepted verbatim).
        #[serde(default)]
        image_url: Option<Value>,
    },
    /// OpenAI Responses image part.
    InputImage {
        /// Image URL or base64 data URL (accepted verbatim).
        #[serde(default)]
        image_url: Option<Value>,
    },
    /// Any other part type; accepted but ignored.
    #[serde(other)]
    Other,
}

/// `stream_options` object.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct StreamOptions {
    /// Emit a final chunk carrying the usage block.
    pub include_usage: Option<bool>,
}

/// `stop` may be a single string or an array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    /// A single stop sequence.
    One(String),
    /// Several stop sequences.
    Many(Vec<String>),
}

/// OpenAI `POST /v1/completions` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Requested model id.
    pub model: Option<String>,
    /// Raw prompt text.
    pub prompt: Option<PromptSpec>,
    /// Completion cap.
    pub max_tokens: Option<usize>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Stop sequence(s).
    pub stop: Option<StopSpec>,
    /// Deterministic sampling seed.
    pub seed: Option<u64>,
    /// Whether to stream server-sent events.
    #[serde(default)]
    pub stream: bool,
    /// Streaming options (`include_usage`).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Echo the prompt in the response.
    pub echo: Option<bool>,
    /// Raw reasoning effort, validated by crate::reasoning.
    pub reasoning_effort: Option<Value>,
    /// Whether thinking is enabled.
    pub enable_thinking: Option<bool>,
    /// Chat-template kwargs.
    pub chat_template_kwargs: Option<Value>,
}

/// `prompt` may be a single string or an array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptSpec {
    /// A single prompt.
    Text(String),
    /// Several prompts (joined with newlines in Milestone 1).
    Many(Vec<String>),
}

/// OpenAI `POST /v1/responses` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    /// Requested model id.
    pub model: Option<String>,
    /// String, message array or content-part array.
    pub input: Option<Value>,
    /// System-style instructions.
    pub instructions: Option<String>,
    /// Output cap.
    pub max_output_tokens: Option<usize>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Whether the client asked for streaming. Milestone 1 answers a true value with a
    /// 400 unsupported_parameter error rather than silently returning a buffered body.
    pub stream: Option<bool>,
    /// Tool definitions, including `apply_patch` custom tools.
    pub tools: Option<Value>,
    /// Reasoning configuration object.
    pub reasoning: Option<Value>,
    /// Whether to persist the response.
    pub store: Option<bool>,
}

/// Anthropic `POST /v1/messages` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    /// Requested model id.
    pub model: Option<String>,
    /// Output cap (required by Anthropic).
    pub max_tokens: Option<usize>,
    /// Conversation messages.
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
    /// System prompt: string or block array.
    pub system: Option<Value>,
    /// Whether the client asked for streaming. Milestone 1 answers a true value with a
    /// 400 unsupported_parameter error rather than silently returning a buffered body.
    pub stream: Option<bool>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Tool definitions.
    pub tools: Option<Value>,
    /// Request metadata.
    pub metadata: Option<Value>,
}

/// One Anthropic message.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    /// `user` or `assistant`.
    pub role: String,
    /// String or block array.
    pub content: Option<Value>,
}
