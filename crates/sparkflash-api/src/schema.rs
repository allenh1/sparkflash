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
    /// Whether to report logprobs for the generated tokens.
    ///
    /// OpenAI's boolean form. When true, `top_logprobs` selects how many alternatives ride along
    /// (default `0`: only each token's own logprob, which is OpenAI's behaviour).
    pub logprobs: Option<bool>,
    /// Per generated token: number of alternative logprobs, or `-1` for the whole vocabulary.
    pub top_logprobs: Option<i32>,
    /// Per prompt token: number of alternative logprobs, or `-1` for the whole vocabulary.
    pub prompt_logprobs: Option<i32>,
    /// Specific vocabulary ids whose logprobs are wanted at every generated position.
    pub logprob_token_ids: Option<Vec<u32>>,
    /// vLLM's chat `echo`: prepend the final message to the continuation.
    ///
    /// The reference chat path cannot render a chat template yet (the router's prompt ids are
    /// placeholders, see `prompt.rs`), so this flag is parsed for protocol compatibility and used
    /// only for its vLLM side effect: when `prompt_logprobs` is absent it selects
    /// `top_logprobs`'s value, as vLLM does. It does not change the prompt.
    pub echo: Option<bool>,
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
    /// Anthropic-style image part, also accepted on the chat route.
    ///
    /// The Anthropic block (`{"type":"image","source":{...}}`) is a shape the router parses on
    /// `/v1/messages` and `/v1/responses`; a client that sends it to `/v1/chat/completions` must not
    /// fall through to [`Self::Other`], because an ignored image part would reach generation with no
    /// image and the capability refusal would never run.
    Image {
        /// Anthropic source object (`{"type":"base64","media_type":...,"data":...}` or a URL).
        #[serde(default)]
        source: Option<Value>,
        /// OpenAI-style URL, if the client sent one under `type: "image"`.
        #[serde(default)]
        image_url: Option<Value>,
    },
    /// Any other part type; accepted but ignored **only if it is not an image carrier**.
    ///
    /// This catch-all is deliberately fail-open for genuinely non-image content (OpenAI adds part
    /// types over time, and Milestone 1 does not consume audio or files). The known protocol image
    /// types (`image_url`, `input_image` and Anthropic `image`) are explicit variants above;
    /// everything else deserialises here. The tag match is case-sensitive, so a spelling such as
    /// `Image` lands here too - which is why the enum is not the guarantee. The guarantee is
    /// `crate::prompt::find_image_carrier`, which scans the raw request body in every handler and
    /// refuses an image-family `type` value or carrier key at any depth, in any case, before
    /// generation (D27); a part that deserialised to `Other` produces no
    /// [`crate::multimodal::ImagePart`], so without that scan the capability check would pass and
    /// the request would be answered from text - exactly the silent fallback D27 forbids. The scan
    /// does not carrier-match the immediate child keys of a JSON Schema name map
    /// (`properties`, `patternProperties`, `$defs`, `definitions`) **when the mapped
    /// value is itself a JSON Schema**, where a property named `image` is a name rather than media,
    /// so a text request is not refused for its schema; it still walks each child schema value, and a
    /// non-schema value under such a name is payload (the name is matched), so every other key -
    /// including an `arguments` or `metadata` object - is refused fail-closed if it carries a
    /// carrier.
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
    /// Per generated token: number of alternative logprobs, or `-1` for the whole vocabulary.
    ///
    /// The vLLM wire integer for /v1/completions. `0` reports only each generated token's own
    /// logprob; a positive `n` reports up to `n` alternatives besides it; `-1` reports the
    /// whole vocabulary (bounded by the server's `--max-logprobs`).
    pub logprobs: Option<i32>,
    /// Per prompt token: number of alternative logprobs, or `-1` for the whole vocabulary.
    ///
    /// Position 0 has no conditioning token, so it always reports `null`. `echo` computes this
    /// implicitly with `logprobs`'s value when the field is absent, which is vLLM's rule.
    pub prompt_logprobs: Option<i32>,
    /// Specific vocabulary ids whose logprobs are wanted at every generated position.
    ///
    /// Takes precedence over `logprobs`'s natural top-k, as vLLM's extension does.
    pub logprob_token_ids: Option<Vec<u32>>,
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
