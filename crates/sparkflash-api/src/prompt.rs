//! Prompt summarisation: text preparation, text-token estimation and visual-token
//! accounting.
//!
//! Milestone 1 owns no tokenizer (the deepseek-recipe encoder lives in the encoding
//! module), so the router needs a deterministic, weight-free way to size a prompt. The
//! estimate here is intentionally trivial and documented: four characters per token, plus
//! a fixed chat-template overhead per message and a fixed reply primer. Image content
//! parts contribute count_image_tokens() visual tokens each.
//!
//! Text is prepared first: `<image>...</image>` spans become the canonical placeholder and
//! every placeholder spelling is normalised to U+FF5C before estimation.

use serde_json::Value;

use crate::multimodal::{
    ImagePart, MultimodalError, anthropic_source_to_data_url, count_image_placeholders,
    default_image_tokens, parse_image_url, prepare_text,
};
use crate::schema::{AnthropicMessage, ChatMessage, ContentPart, MessageContent};

/// Chat-template overhead charged for every message.
pub const CHAT_MESSAGE_OVERHEAD_TOKENS: usize = 4;
/// Tokens charged for the assistant reply primer.
pub const CHAT_REPLY_PRIMER_TOKENS: usize = 3;
/// Deterministic character-per-token divisor used by the Milestone 1 estimator.
pub const CHARS_PER_TOKEN: usize = 4;

/// A sized prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptSummary {
    /// Placeholder token ids (all zero in Milestone 1).
    pub token_ids: Vec<u32>,
    /// Visual tokens contributed by images.
    pub image_tokens: usize,
    /// Total UTF-8 bytes of prepared text seen.
    pub text_bytes: usize,
    /// Image references in request order.
    pub images: Vec<ImagePart>,
}

impl PromptSummary {
    /// Prompt tokens including visual tokens.
    #[must_use]
    pub fn prompt_tokens(&self) -> usize {
        self.token_ids.len() + self.image_tokens
    }

    /// Appends another summary's tokens, images and byte count to this one.
    pub fn absorb(&mut self, other: Self) {
        self.token_ids.extend(other.token_ids);
        self.image_tokens += other.image_tokens;
        self.text_bytes += other.text_bytes;
        self.images.extend(other.images);
    }

    fn push_tokens(&mut self, count: usize) {
        self.token_ids.resize(self.token_ids.len() + count, 0);
    }

    fn push_text(&mut self, text: &str) {
        let prepared = prepare_text(text);
        self.text_bytes += prepared.len();
        let placeholders = count_image_placeholders(&prepared);
        self.push_tokens(estimate_text_tokens(&prepared));
        for _ in 0..placeholders {
            self.push_image(ImagePart::Placeholder);
        }
    }

    fn push_image(&mut self, part: ImagePart) {
        self.image_tokens += default_image_tokens();
        self.images.push(part);
    }
}

/// Estimates the number of text tokens in `text`.
///
/// Empty text costs nothing; otherwise every four Unicode scalar values cost one token.
#[must_use]
pub fn estimate_text_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        0
    } else {
        chars.div_ceil(CHARS_PER_TOKEN)
    }
}

/// Sizes an OpenAI chat message list.
///
/// # Errors
/// Returns MultimodalError when an image part is not a decodable base64 data URL or an
/// http(s) URL.
pub fn summarize_chat_messages(messages: &[ChatMessage]) -> Result<PromptSummary, MultimodalError> {
    let mut summary = PromptSummary::default();
    for message in messages {
        summary.push_tokens(CHAT_MESSAGE_OVERHEAD_TOKENS);
        match &message.content {
            None => {}
            Some(MessageContent::Text(text)) => summary.push_text(text),
            Some(MessageContent::Parts(parts)) => {
                for part in parts {
                    match part {
                        ContentPart::Text { text } | ContentPart::InputText { text } => {
                            summary.push_text(text);
                        }
                        ContentPart::ImageUrl { image_url }
                        | ContentPart::InputImage { image_url } => {
                            let url = extract_image_url(image_url.as_ref())?;
                            summary.push_image(parse_image_url(url)?);
                        }
                        ContentPart::Other => {}
                    }
                }
            }
        }
    }
    summary.push_tokens(CHAT_REPLY_PRIMER_TOKENS);
    Ok(summary)
}

/// Sizes a raw completion prompt. Never fails: plain text can only carry placeholders.
#[must_use]
pub fn summarize_prompt_text(text: &str) -> PromptSummary {
    let mut summary = PromptSummary::default();
    summary.push_text(text);
    summary
}

/// Sizes the OpenAI Responses `input` field, which may be a string, a content-part array
/// or an array of message objects.
///
/// # Errors
/// Returns MultimodalError for malformed image parts.
pub fn summarize_responses_input(input: &Value) -> Result<PromptSummary, MultimodalError> {
    let mut summary = PromptSummary::default();
    push_value(&mut summary, input)?;
    summary.push_tokens(CHAT_REPLY_PRIMER_TOKENS);
    Ok(summary)
}

/// Sizes the Anthropic `system` + `messages` pair.
///
/// # Errors
/// Returns MultimodalError for malformed image blocks.
pub fn summarize_anthropic_messages(
    system: Option<&Value>,
    messages: &[AnthropicMessage],
) -> Result<PromptSummary, MultimodalError> {
    let mut summary = PromptSummary::default();
    if let Some(system) = system {
        push_value(&mut summary, system)?;
    }
    for message in messages {
        summary.push_tokens(CHAT_MESSAGE_OVERHEAD_TOKENS);
        if let Some(content) = &message.content {
            push_value(&mut summary, content)?;
        }
    }
    summary.push_tokens(CHAT_REPLY_PRIMER_TOKENS);
    Ok(summary)
}

fn extract_image_url(value: Option<&Value>) -> Result<&str, MultimodalError> {
    match value {
        Some(Value::String(url)) => Ok(url),
        Some(Value::Object(map)) => map.get("url").and_then(Value::as_str).ok_or_else(|| {
            MultimodalError::Unsupported("image_url object is missing a string url".to_owned())
        }),
        _ => Err(MultimodalError::Unsupported(
            "image_url must be a string or an object with a string url".to_owned(),
        )),
    }
}

fn push_value(summary: &mut PromptSummary, value: &Value) -> Result<(), MultimodalError> {
    match value {
        Value::String(text) => summary.push_text(text),
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(text) => summary.push_text(text),
                    Value::Object(map) => push_object(summary, map)?,
                    _ => {}
                }
            }
        }
        Value::Object(map) => push_object(summary, map)?,
        _ => {}
    }
    Ok(())
}

fn push_object(
    summary: &mut PromptSummary,
    map: &serde_json::Map<String, Value>,
) -> Result<(), MultimodalError> {
    let part_type = map.get("type").and_then(Value::as_str).unwrap_or_default();
    match part_type {
        "text" | "input_text" | "output_text" => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                summary.push_text(text);
            }
        }
        "image_url" | "input_image" => {
            let url = extract_image_url(map.get("image_url"))?;
            summary.push_image(parse_image_url(url)?);
        }
        "image" => {
            if let Some(source) = map.get("source").and_then(Value::as_object) {
                let media_type = source
                    .get("media_type")
                    .and_then(Value::as_str)
                    .unwrap_or("application/octet-stream");
                let data = source
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let url = anthropic_source_to_data_url(media_type, data);
                summary.push_image(parse_image_url(&url)?);
            } else if map.contains_key("image_url") {
                let url = extract_image_url(map.get("image_url"))?;
                summary.push_image(parse_image_url(url)?);
            } else {
                summary.push_image(ImagePart::Placeholder);
            }
        }
        "message" => {
            if let Some(content) = map.get("content") {
                push_value(summary, content)?;
            }
        }
        _ => {
            if let Some(content) = map.get("content") {
                push_value(summary, content)?;
            } else if let Some(text) = map.get("text").and_then(Value::as_str) {
                summary.push_text(text);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(role: &str, content: Value) -> ChatMessage {
        serde_json::from_value(json!({ "role": role, "content": content })).expect("message")
    }

    fn data_url() -> &'static str {
        "data:image/png;base64,aGVsbG8="
    }

    #[test]
    fn empty_text_costs_nothing_and_four_chars_cost_one_token() {
        assert_eq!(estimate_text_tokens(""), 0);
        assert_eq!(estimate_text_tokens("abcd"), 1);
        assert_eq!(estimate_text_tokens("abcde"), 2);
        assert_eq!(
            estimate_text_tokens("\u{1f600}\u{1f600}\u{1f600}\u{1f600}"),
            1
        );
    }

    #[test]
    fn chat_messages_are_charged_overhead_and_a_reply_primer() {
        let summary = summarize_chat_messages(&[message("user", json!("hi"))]).expect("sized");
        // 4 overhead + 1 text + 3 primer.
        assert_eq!(summary.prompt_tokens(), 8);
        assert_eq!(summary.image_tokens, 0);
        assert!(summary.images.is_empty());
    }

    #[test]
    fn image_url_parts_are_decoded_and_counted() {
        let summary = summarize_chat_messages(&[message(
            "user",
            json!([
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": data_url()}},
            ]),
        )])
        .expect("sized");
        assert_eq!(
            summary.images,
            vec![ImagePart::Data {
                media_type: "image/png".to_owned(),
                byte_len: 5,
            }]
        );
        assert_eq!(summary.image_tokens, default_image_tokens());
        assert_eq!(summary.prompt_tokens(), 4 + 2 + default_image_tokens() + 3);
    }

    #[test]
    fn http_image_parts_are_accepted_without_fetching() {
        let summary = summarize_chat_messages(&[message(
            "user",
            json!([{"type": "image_url", "image_url": "https://example.invalid/a.png"}]),
        )])
        .expect("sized");
        assert_eq!(
            summary.images,
            vec![ImagePart::Url {
                url: "https://example.invalid/a.png".to_owned(),
            }]
        );
    }

    #[test]
    fn malformed_image_parts_are_rejected() {
        let error = summarize_chat_messages(&[message(
            "user",
            json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!"}}]),
        )])
        .expect_err("malformed data url");
        assert!(matches!(error, MultimodalError::MalformedDataUrl(_)));

        let error = summarize_chat_messages(&[message(
            "user",
            json!([{"type": "image_url", "image_url": "ftp://example.invalid/a.png"}]),
        )])
        .expect_err("unsupported scheme");
        assert!(matches!(error, MultimodalError::Unsupported(_)));
    }

    #[test]
    fn inline_image_tags_become_placeholders_and_count_as_images() {
        let summary =
            summarize_chat_messages(&[message("user", json!("look <image>AAAA</image> here"))])
                .expect("sized");
        assert_eq!(summary.images, vec![ImagePart::Placeholder]);
        assert_eq!(summary.image_tokens, default_image_tokens());
    }

    #[test]
    fn ascii_placeholders_in_raw_text_are_normalised_and_counted() {
        let summary =
            summarize_chat_messages(&[message("user", json!("<|image|>"))]).expect("sized");
        assert_eq!(summary.images, vec![ImagePart::Placeholder]);

        let raw = summarize_prompt_text("<|deepseek_image|>");
        assert_eq!(raw.image_tokens, default_image_tokens());
        assert_eq!(raw.text_bytes, crate::multimodal::IMAGE_PLACEHOLDER.len());
    }

    #[test]
    fn responses_input_accepts_strings_arrays_and_messages() {
        let string = summarize_responses_input(&json!("hello world")).expect("sized");
        assert_eq!(string.prompt_tokens(), 3 + 3);
        let parts = summarize_responses_input(&json!([
            {"type": "input_text", "text": "abcd"},
            {"type": "input_image", "image_url": "http://example.invalid/x.png"},
        ]))
        .expect("sized");
        assert_eq!(parts.image_tokens, default_image_tokens());
        assert_eq!(parts.token_ids.len(), 1 + 3);
        let messages = summarize_responses_input(&json!([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "abcd"}]},
        ]))
        .expect("sized");
        assert_eq!(messages.token_ids.len(), 1 + 3);
    }

    #[test]
    fn anthropic_system_and_blocks_are_sized() {
        let messages: Vec<AnthropicMessage> = serde_json::from_value(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "abcd"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
            ]},
        ]))
        .expect("anthropic messages");
        let summary =
            summarize_anthropic_messages(Some(&json!("be nice")), &messages).expect("sized");
        // system (2) + overhead (4) + text (1) + image + primer (3)
        assert_eq!(
            summary.prompt_tokens(),
            2 + 4 + 1 + default_image_tokens() + 3
        );
        assert_eq!(summary.image_tokens, default_image_tokens());
        assert_eq!(
            summary.images,
            vec![ImagePart::Data {
                media_type: "image/png".to_owned(),
                byte_len: 5,
            }]
        );
    }

    #[test]
    fn anthropic_image_without_a_source_is_a_placeholder() {
        let messages: Vec<AnthropicMessage> =
            serde_json::from_value(json!([{"role": "user", "content": [{"type": "image"}]}]))
                .expect("anthropic messages");
        let summary = summarize_anthropic_messages(None, &messages).expect("sized");
        assert_eq!(summary.images, vec![ImagePart::Placeholder]);
    }
}
