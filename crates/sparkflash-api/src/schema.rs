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

//! OpenAI and Anthropic request schemas accepted by the router.
//!
//! The types only describe what the protocol layer understands; validation (reasoning
//! effort ranges, model existence, tool shape) happens in the handlers so that every
//! failure can be rendered as an OpenAI-shaped error envelope.

use serde::Deserialize;
use serde_json::{Map, Value};

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
    /// Number of most likely tokens the candidate filter keeps: 0 and -1 disable it, values above
    /// zero keep that many (vLLM's top_k; sampling_params.py).
    pub top_k: Option<i32>,
    /// Repetition penalty over the prompt and generated tokens; 1.0 disables it (vLLM's
    /// repetition_penalty; sampling_params.py lines 248-253).
    pub repetition_penalty: Option<f32>,
    /// Presence penalty over generated tokens; 0.0 disables it (vLLM's presence_penalty;
    /// sampling_params.py lines 240-243).
    pub presence_penalty: Option<f32>,
    /// Frequency penalty over generated tokens; 0.0 disables it (vLLM's frequency_penalty;
    /// sampling_params.py lines 244-246).
    pub frequency_penalty: Option<f32>,
    /// Whether special tokens are hidden from the generated text. vLLM's OpenAI default is true
    /// (chat/protocol.py:274); an explicit false passes the raw decoded text through.
    pub skip_special_tokens: Option<bool>,
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
    /// When the tokenizer seam is attached the prompt ids are the released V4.1 template's real ids
    /// (see `encoding/seam.rs`), but echoing the final message changes the generation assembly rather
    /// than the prompt, so it is still deferred with the logprob protocol work. This flag is parsed
    /// for protocol compatibility and used only for its vLLM side effect: when `prompt_logprobs` is
    /// absent it selects `top_logprobs`'s value, as vLLM does. It does not change the prompt.
    pub echo: Option<bool>,
    /// Tool definitions (accepted verbatim in Milestone 1).
    pub tools: Option<Value>,
    /// OpenAI's tool-choice selector: `"none"`, `"auto"`, `"required"` or
    /// `{"type":"function","function":{"name":...}}`.
    ///
    /// `"none"` is honoured: the tools are removed from the rendered prompt and the response never
    /// carries a parsed tool call, which is what the OpenAI contract means by "do not call a tool".
    /// `"auto"` (the default) keeps today's behaviour. `"required"` and a named function are
    /// **refused with a typed 400**: this seam renders the tool list but has no policy that can
    /// force a call, so accepting them would be a silent promise the server does not keep.
    pub tool_choice: Option<Value>,
    /// Whether the model may answer with more than one tool call in one response.
    ///
    /// OpenAI's default is true. An explicit false is enforced at the parser boundary: only the
    /// first parsed call is reported (buffered and streamed), instead of returning a parallel set
    /// the caller said it cannot accept.
    pub parallel_tool_calls: Option<bool>,
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
    /// Number of choices requested. Each choice is one independent sample of the same prompt;
    /// a request with n > 1 is refused with a typed error on an engine that cannot guarantee
    /// independent draws (the router keeps single-choice behaviour byte-for-byte unchanged).
    pub n: Option<usize>,
    /// Structured-output request. `text` is the default and a no-op; `json_object` and
    /// `json_schema` need constrained decoding, which this server does not implement, so they are
    /// refused with a typed error rather than silently returning unconstrained text.
    pub response_format: Option<ResponseFormat>,
    /// End-user identifier.
    pub user: Option<String>,
    /// OpenAI's deprecated "sample best_of, return n" knob.
    ///
    /// vLLM accepts the field and ignores it (measured on the reference deployment 2026-10-21: a
    /// single-choice request with `best_of: 2` answers one choice), so this server parses it and
    /// documents the same no-op rather than pretending it selected a sample.
    pub best_of: Option<usize>,
    /// Per-token logit bias: token id (as a JSON key) to a value in `-100..=100`.
    ///
    /// Validated here and refused with a typed 400: the engine seam has no logits processor to apply
    /// it, and the reference deployment refuses it as well (its speculative-decoding sampler reports
    /// `min_p` and `logit_bias` as unsupported).
    pub logit_bias: Option<Map<String, Value>>,
    /// Minimum-probability sampler threshold; `0.0` means disabled and is accepted as a no-op.
    pub min_p: Option<f32>,
}

/// OpenAI `response_format` for chat completions.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text output, the default.
    Text,
    /// Unconstrained JSON-object mode.
    ///
    /// The model is not constrained to emit valid JSON on this server, so the request is refused
    /// instead of returning text that may not parse.
    JsonObject,
    /// JSON Schema mode.
    ///
    /// The schema is not enforced by the decoder on this server, so the request is refused instead
    /// of returning text that may not match the schema.
    JsonSchema {
        /// The OpenAI schema wrapper.
        json_schema: JsonSchemaResponseFormat,
    },
}

/// The `json_schema` member of a `response_format`.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaResponseFormat {
    /// Schema name, required by OpenAI.
    pub name: String,
    /// The JSON Schema document.
    pub schema: Value,
    /// Whether OpenAI clients mark the schema strict; parsed, never enforced here.
    #[serde(default)]
    pub strict: Option<bool>,
    /// Optional schema description.
    #[serde(default)]
    pub description: Option<String>,
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
    /// Put the running usage block on every chunk, not only on the final one.
    ///
    /// vLLM only honours this together with `include_usage`; the reference server answers a
    /// request that sets it alone with a normal stream that carries no usage anywhere, and so does
    /// this server.
    pub continuous_usage_stats: Option<bool>,
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
    /// Number of most likely tokens the candidate filter keeps: 0 and -1 disable it, values above
    /// zero keep that many (vLLM's top_k; sampling_params.py).
    pub top_k: Option<i32>,
    /// Repetition penalty over the prompt and generated tokens; 1.0 disables it (vLLM's
    /// repetition_penalty; sampling_params.py lines 248-253).
    pub repetition_penalty: Option<f32>,
    /// Presence penalty over generated tokens; 0.0 disables it (vLLM's presence_penalty;
    /// sampling_params.py lines 240-243).
    pub presence_penalty: Option<f32>,
    /// Frequency penalty over generated tokens; 0.0 disables it (vLLM's frequency_penalty;
    /// sampling_params.py lines 244-246).
    pub frequency_penalty: Option<f32>,
    /// Whether special tokens are hidden from the generated text. vLLM's OpenAI default is true
    /// (completion/protocol.py:82); an explicit false passes the raw decoded text through.
    pub skip_special_tokens: Option<bool>,
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
    /// Completions per prompt. The OpenAI default is 1; choice index is
    /// `prompt_index * n + choice_index`, and each choice is an independent sample.
    pub n: Option<usize>,
    /// Structured-output request, parsed with the chat rule (S2).
    ///
    /// The vLLM completions protocol does not carry this member and the reference deployment
    /// ignores it; this server refuses the two JSON modes with the same typed 400 as chat rather
    /// than silently returning unconstrained prose for a request that asked for JSON.
    pub response_format: Option<ResponseFormat>,
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
    /// OpenAI's deprecated "sample best_of, return n" knob.
    ///
    /// vLLM accepts the field and ignores it (only `n` schedules samples; measured on the
    /// reference deployment 2026-10-21: `best_of=2, n absent` answers one choice). This schema keeps
    /// the field for wire compatibility and documents the same no-op instead of silently dropping
    /// an unknown member.
    pub best_of: Option<usize>,
    /// Per-token logit bias: token id (as a JSON key) to a value in `-100..=100`.
    ///
    /// Validated here (vLLM's conversion errors are copied) and then refused with a typed 400,
    /// because the reference engine's sampler has no logits processor to apply it. The reference
    /// deployment refuses it too - its speculative-decoding sampler reports `min_p` and
    /// `logit_bias` as unsupported - so the status matches even though the cause differs.
    pub logit_bias: Option<Map<String, Value>>,
    /// Minimum-probability sampler threshold, `0.0..=1.0` with `0.0` meaning disabled.
    ///
    /// Validated here and refused typed for a positive value, for the same reason as
    /// `logit_bias`; `0.0` is the disabled value and is accepted as a no-op.
    pub min_p: Option<f32>,
}

/// `prompt` may be a single string, an array of strings, or an array of token ids.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptSpec {
    /// A single prompt.
    Text(String),
    /// Several independent prompts: one response choice per prompt, vLLM's list form.
    ///
    /// The choice index is the prompt's position in this list. The nested token-id form
    /// (`int[][]`) is not accepted; a token-id batch is expressed as one `int[]` prompt.
    Many(Vec<String>),
    /// A pre-tokenized prompt, passed to the engine unchanged (vLLM's `int[]` form).
    Tokens(Vec<u32>),
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
    /// Whether the client asked for streaming. The handler emits the Responses SSE event family
    /// when true and the buffered response object when false.
    pub stream: Option<bool>,
    /// Tool definitions in the Responses shape (`{type:"function", name, parameters}`).
    pub tools: Option<Value>,
    /// Tool-choice selector. `"auto"` and `"none"` are honoured; `"required"` and a named
    /// function selector are refused typed because this server cannot force the call.
    pub tool_choice: Option<Value>,
    /// Whether one answer may carry more than one tool call.
    pub parallel_tool_calls: Option<bool>,
    /// Reasoning configuration object.
    pub reasoning: Option<Value>,
    /// Output-format configuration. Only the text default is supported; JSON modes are refused.
    pub text: Option<Value>,
    /// Whether to persist the response (accepted as a no-op: this server keeps no store).
    pub store: Option<bool>,
    /// Truncation policy (`auto` or `disabled`); accepted as a no-op.
    pub truncation: Option<Value>,
    /// Previous response id. This server keeps no response store, so references are refused.
    pub previous_response_id: Option<String>,
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
    /// Whether the client asked for streaming. The handler emits the Anthropic Messages SSE event
    /// family when true and the buffered message object when false.
    pub stream: Option<bool>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Tool definitions in the Anthropic shape (`{name, description, input_schema}`).
    pub tools: Option<Value>,
    /// Tool-choice selector. Anthropic `auto` and `none` are honoured; `any` and `tool` are
    /// refused typed because this server cannot force the call.
    pub tool_choice: Option<Value>,
    /// Extended-thinking configuration.
    pub thinking: Option<Value>,
    /// Anthropic's `output_config` (adaptive thinking's `effort` lives here).
    pub output_config: Option<Value>,
    /// Top-k sampling (Anthropic's extension over the OpenAI surface).
    pub top_k: Option<i32>,
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
