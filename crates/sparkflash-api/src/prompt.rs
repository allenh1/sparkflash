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

//! Prompt summarisation: text preparation, text-token estimation and visual-token
//! accounting.
//!
//! Milestone 1 owns no tokenizer (the deepseek-recipe encoder lives in the encoding
//! module), so the router needs a deterministic, weight-free way to size a prompt. The
//! estimate here is intentionally trivial and documented: four characters per token, plus
//! a fixed chat-template overhead per message and a fixed reply primer.
//!
//! This heuristic is the fallback for a router with no checkpoint tokenizer attached (the
//! weight-free tests). A production `sparkflash serve` attaches a model pack's `PromptCodec`,
//! and then `/v1/completions`, `/v1/chat/completions`, `/v1/responses` and `/v1/messages` carry
//! the checkpoint's real token ids instead of these placeholders. The Responses and Anthropic
//! adapters convert their request shapes to the chat template and tokenize through the same
//! `PromptCodec::encode_chat` seam, so the fallback only serves a router with no tokenizer.
//!
//! Image content parts are **recorded**, not charged. The vision path is 263
//! `vision.*`/`aligner.*` tensors plus the three image delimiters, none of which the tensor plan
//! loads (DECISIONS.md D18), so the engine has no honest visual-token count to report and the router
//! refuses a request carrying an image part with a typed missing-capability error
//! ([`crate::engine::MissingCapability`]). In particular there is no assumed 32x32 grid: a
//! fabricated count would be a number the engine never produced.
//!
//! Text is prepared first: `<image>...</image>` spans become the canonical placeholder and
//! every placeholder spelling is normalised to U+FF5C before estimation.
//!
//! The typed parsers are not the only defense: `find_image_carrier` scans the raw request body
//! for image carriers at any depth (any image-family `type` value or carrier key,
//! case-insensitively), so a spelling the typed enum does not know is refused rather than ignored.
//! The one exemption is the immediate child keys of the JSON Schema name maps
//! (`properties`, `patternProperties`, `$defs`, `definitions`) **when the mapped value is
//! itself a JSON Schema** (bool, empty object, or keyword-only object): those keys are schema
//! names - a tool parameter or `response_format` schema may legitimately be called `image` -
//! so they are not carrier-matched, while each child schema value is still walked. A non-schema
//! value under such a name is payload, so the name is carrier-matched; no subtree is blanket-exempt
//! and a carrier under `tools`, `arguments` or `metadata` is refused fail-closed,
//! because silently serving it would be the same fallback D27 forbids.

use serde_json::Value;

use crate::multimodal::{
    ImagePart, MultimodalError, anthropic_source_to_data_url, count_image_placeholders,
    parse_image_url, prepare_text,
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
    /// Total UTF-8 bytes of prepared text seen.
    pub text_bytes: usize,
    /// Image references in request order. A non-empty list is what the engine seam refuses today.
    pub images: Vec<ImagePart>,
}

impl PromptSummary {
    /// Prompt tokens. Text only: an image part is recorded in [`Self::images`] and refused by the
    /// engine seam rather than charged a visual-token count nothing produced.
    #[must_use]
    pub fn prompt_tokens(&self) -> usize {
        self.token_ids.len()
    }

    /// Appends another summary's tokens, images and byte count to this one.
    pub fn absorb(&mut self, other: Self) {
        self.token_ids.extend(other.token_ids);
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
        // Recording the part is the whole job: no visual token is charged (see the module docs).
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
                        ContentPart::Image { source, image_url } => {
                            push_image_source(&mut summary, source.as_ref(), image_url.as_ref())?;
                        }
                        // Genuinely non-image parts only. The protocol image `type`s are explicit
                        // variants above; a spelling the enum does not know (a case variant, a future
                        // item type) lands here, which is why `find_image_carrier` scans the raw body
                        // in every handler before generation (D27).
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

/// Records one image part from either an Anthropic `source` object or an `image_url` value.
///
/// Shared by the raw-object paths (`/v1/messages`, `/v1/responses`) and the typed chat-part path,
/// so every route that accepts an image turns it into an [`ImagePart`] for the engine seam's
/// capability check.
fn push_image_source(
    summary: &mut PromptSummary,
    source: Option<&Value>,
    image_url: Option<&Value>,
) -> Result<(), MultimodalError> {
    if let Some(Value::Object(source)) = source {
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
    } else if let Some(image_url) = image_url {
        let url = extract_image_url(Some(image_url))?;
        summary.push_image(parse_image_url(url)?);
    } else {
        summary.push_image(ImagePart::Placeholder);
    }
    Ok(())
}

/// Object keys that carry an image in the request protocols this router serves.
///
/// The typed parsers know the protocol spellings they were written for; this list is the raw-body
/// backstop, so a key can appear at any depth and an item type added later (or a spelling an
/// existing parser never learned) still cannot smuggle an image past the capability check.
const IMAGE_CARRIER_KEYS: [&str; 5] = [
    "image_url",
    "file_id",
    "input_image",
    "image",
    "computer_screenshot",
];

/// Whether an object key names an image carrier, case-insensitively.
fn is_image_carrier_key(key: &str) -> bool {
    IMAGE_CARRIER_KEYS
        .iter()
        .any(|carrier| key.eq_ignore_ascii_case(carrier))
}

/// JSON Schema keywords whose values are maps from *names* to schemas.
///
/// The immediate child keys of these maps are schema names (a tool parameter may legitimately be
/// called `image` or `image_url`), so the scan does not carrier-match those keys when the
/// value they map to is itself a JSON Schema (see [`is_json_schema_value`]) - and it still walks
/// that schema value, so an image carrier inside a schema is still refused. No other key is exempt:
/// in particular `tools`, `response_format`, `arguments` and `metadata` are
/// scanned normally (an `arguments` object carrying `image_url` is nonstandard and is
/// refused fail-closed). Review round 3: exempting whole subtrees let a real carrier under those
/// keys return `None` and be answered from text, the silent fallback these rounds exist to close.
const SCHEMA_NAME_MAPS: [&str; 4] = ["properties", "patternProperties", "$defs", "definitions"];

/// JSON Schema keywords: the keys a mapped value may use and still be recognised as a schema.
///
/// Draft 2020-12 core/validation/applicator/format/annotation/content keywords, the draft-07 legacy
/// spellings (`definitions`, `dependencies`, `additionalItems`), `nullable`, and
/// the OpenAPI annotation keys common tool schemas carry (`example`, `xml`,
/// `externalDocs`; `deprecated`, `readOnly` and `writeOnly` are already schema
/// annotations). A mapped object with any other key is treated as payload, not a schema - the
/// deliberate fail-closed policy: an unknown keyword cannot be proven to be a schema, so a carrier
/// name under it is refused rather than served (review round 3, third false positive: `minProperties`
/// and friends were missing from the first list, so valid schemas were refused).
const SCHEMA_KEYWORDS: [&str; 65] = [
    "$schema",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$ref",
    "$dynamicRef",
    "$vocabulary",
    "$comment",
    "$defs",
    "definitions",
    "title",
    "description",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
    "examples",
    "type",
    "enum",
    "const",
    "format",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "properties",
    "patternProperties",
    "additionalProperties",
    "propertyNames",
    "unevaluatedProperties",
    "required",
    "dependentRequired",
    "dependentSchemas",
    "items",
    "prefixItems",
    "additionalItems",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "discriminator",
    "nullable",
    "minProperties",
    "maxProperties",
    "dependencies",
    "example",
    "xml",
    "externalDocs",
];

/// Whether a value mapped from a schema name is a JSON Schema rather than payload.
///
/// A JSON Schema is a boolean (the `true`/`false` schemas) or an object that is empty or whose
/// keys are all [`SCHEMA_KEYWORDS`]. Anything else - a string, a number, an array, or an object
/// with an unknown key such as `url` - is not a schema, so the name it is mapped from is
/// payload and is carrier-matched. Without this test a non-schema value under
/// `properties.image` was invisible: review round 3 found
/// `{"properties":{"image":"data:image/png;base64,..."}}` served with generation entered.
fn is_json_schema_value(value: &Value) -> bool {
    match value {
        Value::Bool(_) => true,
        Value::Object(map) => {
            map.is_empty()
                || map
                    .keys()
                    .all(|key| SCHEMA_KEYWORDS.iter().any(|keyword| keyword == key))
        }
        _ => false,
    }
}

/// Whether an object key heads a map of schema names.
fn is_schema_name_map(key: &str) -> bool {
    SCHEMA_NAME_MAPS
        .iter()
        .any(|keyword| key.eq_ignore_ascii_case(keyword))
}

/// Whether a `type` value names the image family, case-insensitively.
///
/// This is a substring test on purpose: it catches the protocol spellings (`image_url`,
/// `input_image`, `computer_screenshot`) and any future combination that keeps the family
/// word, without a list that goes stale when a protocol adds an item type.
fn names_the_image_family(type_value: &str) -> bool {
    let lowered = type_value.to_ascii_lowercase();
    ["image", "screenshot", "vision"]
        .iter()
        .any(|family| lowered.contains(family))
}

/// Finds an image carrier anywhere in a raw request body, without the typed parsers.
///
/// This is the structural half of D27's refusal. The typed parsers record the protocol image
/// spellings they know; review found three successive spellings that reached generation because an
/// enumeration is only as complete as the last shape read (a chat-route Anthropic block, a nested
/// `function_call_output` array, and the documented `computer_screenshot` output pair).
/// This scan is not an enumeration of containers: it walks **every** object and array in the body
/// and refuses a carrier wherever it appears. The single exception is an immediate child key of one
/// of the schema-name maps in [`SCHEMA_NAME_MAPS`] **when the value it maps to is itself a JSON
/// Schema** ([`is_json_schema_value`]): that key is a schema name, not payload, so it is not
/// carrier-matched, while the schema value is still walked. A non-schema value under such a name is
/// payload: the name is carrier-matched and the value falls through to the normal scan.
///
/// A carrier is (a) any object key in [`IMAGE_CARRIER_KEYS`], (b) any string `type` value
/// containing `image`, `screenshot` or `vision` case-insensitively, or (c) **any
/// value** under one of those keys - the key alone is the carrier, so `{"image_url": null}` is
/// refused and the `data:` URL case is not special. Arbitrary string *values* elsewhere are not
/// scanned: a text field that mentions the word "image", or even contains a `data:` URL, is
/// text, not a carrier, and stays accepted (D27's fail-open rule for genuinely non-image content).
///
/// Returns a description of the first carrier found - its JSON path and the key or `type`
/// spelling that matched - so the HTTP refusal can name the shape it found.
#[must_use]
pub fn find_image_carrier(value: &Value) -> Option<String> {
    let mut found = None;
    scan_for_image_carrier(value, "$", &mut found);
    found
}

fn scan_for_image_carrier(value: &Value, path: &str, found: &mut Option<String>) {
    if found.is_some() {
        return;
    }
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if is_schema_name_map(key)
                    && let Value::Object(names) = child
                {
                    for (name, mapped) in names {
                        if is_json_schema_value(mapped) {
                            // A schema name: not carrier-matched, but the schema value is walked.
                            scan_for_image_carrier(mapped, &format!("{path}.{key}.{name}"), found);
                        } else {
                            // Not a schema: the name is payload and is carrier-matched.
                            if is_image_carrier_key(name) {
                                *found = Some(format!("the {name:?} key at {path}.{key}.{name}"));
                                return;
                            }
                            scan_for_image_carrier(mapped, &format!("{path}.{key}.{name}"), found);
                        }
                        if found.is_some() {
                            return;
                        }
                    }
                    continue;
                }
                if is_image_carrier_key(key) {
                    *found = Some(format!("the {key:?} key at {path}.{key}"));
                    return;
                }
                if key.eq_ignore_ascii_case("type")
                    && let Some(type_value) = child.as_str()
                    && names_the_image_family(type_value)
                {
                    *found = Some(format!("the {type_value:?} type at {path}.type"));
                    return;
                }
                scan_for_image_carrier(child, &format!("{path}.{key}"), found);
                if found.is_some() {
                    return;
                }
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                scan_for_image_carrier(item, &format!("{path}[{index}]"), found);
                if found.is_some() {
                    return;
                }
            }
        }
        _ => {}
    }
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
        "image" => push_image_source(summary, map.get("source"), map.get("image_url"))?,
        "message" => {
            if let Some(content) = map.get("content") {
                push_value(summary, content)?;
            }
        }
        _ => {
            // Responses items nest their payload under `content` (messages) or `output`
            // (`function_call_output`, `computer_call_output`). Both are walked: an image inside
            // either must become an [`ImagePart`] or the request would be answered from its text
            // alone - the silent fallback D27 forbids (review round 2 found the `output` case).
            match (map.get("content"), map.get("output")) {
                (Some(content), Some(output)) => {
                    push_value(summary, content)?;
                    push_value(summary, output)?;
                }
                (Some(content), None) => push_value(summary, content)?,
                (None, Some(output)) => push_value(summary, output)?,
                (None, None) => {
                    if let Some(text) = map.get("text").and_then(Value::as_str) {
                        summary.push_text(text);
                    }
                }
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
        assert!(summary.images.is_empty());
    }

    #[test]
    fn image_url_parts_are_decoded_and_recorded_without_charging_visual_tokens() {
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
            vec![ImagePart::data("image/png", b"hello".to_vec())]
        );
        // 4 overhead + 2 text + 3 primer. No visual token is charged: the engine cannot produce
        // one, and a fabricated grid count would be a number nothing measured (D27).
        assert_eq!(summary.prompt_tokens(), 4 + 2 + 3);
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
    fn chat_route_anthropic_image_block_is_recorded_not_ignored() {
        // Regression: this exact body sent to /v1/chat/completions used to deserialise to
        // `ContentPart::Other` and be dropped, so the capability refusal never saw an image and the
        // request was answered from its text tokens (critic finding on 8a1b99b, fixed under D27).
        let summary = summarize_chat_messages(&[message(
            "user",
            json!([
                {"type": "text", "text": "hi"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
            ]),
        )])
        .expect("sized");
        assert_eq!(
            summary.images,
            vec![ImagePart::data("image/png", b"hello".to_vec())]
        );
    }

    #[test]
    fn a_responses_function_call_output_nested_image_is_recorded() {
        // Review round 2: the default arm used to recurse only into `content`/`text`, so an image
        // nested under a function_call_output's `output` array never became an ImagePart and the
        // request was answered from its text alone (HTTP 200).
        let summary = summarize_responses_input(&json!([
            {"type": "function_call_output", "call_id": "c1", "output": [
                {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
            ]},
        ]))
        .expect("sized");
        assert_eq!(
            summary.images,
            vec![ImagePart::data("image/png", b"hello".to_vec())]
        );
    }

    #[test]
    fn a_genuinely_non_image_unknown_part_is_still_accepted_and_ignored() {
        // The `ContentPart::Other` catch-all stays fail-open for non-image content, so adding the
        // image variant did not turn the chat route into a whitelist (D27).
        let summary = summarize_chat_messages(&[message(
            "user",
            json!([{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]),
        )])
        .expect("sized");
        assert!(summary.images.is_empty());
    }

    #[test]
    fn inline_image_tags_become_placeholders_and_record_the_image_part() {
        let summary =
            summarize_chat_messages(&[message("user", json!("look <image>AAAA</image> here"))])
                .expect("sized");
        assert_eq!(summary.images, vec![ImagePart::Placeholder]);
    }

    #[test]
    fn ascii_placeholders_in_raw_text_are_normalised_and_counted() {
        let summary =
            summarize_chat_messages(&[message("user", json!("<|image|>"))]).expect("sized");
        assert_eq!(summary.images, vec![ImagePart::Placeholder]);

        let raw = summarize_prompt_text("<|deepseek_image|>");
        assert_eq!(raw.images, vec![ImagePart::Placeholder]);
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
        assert_eq!(
            parts.images,
            vec![ImagePart::Url {
                url: "http://example.invalid/x.png".to_owned(),
            }]
        );
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
        // system (2) + overhead (4) + text (1) + primer (3); the image part is recorded, not charged.
        assert_eq!(summary.prompt_tokens(), 2 + 4 + 1 + 3);
        assert_eq!(
            summary.images,
            vec![ImagePart::data("image/png", b"hello".to_vec())]
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

    #[test]
    fn the_image_carrier_detector_finds_arbitrary_nested_shapes() {
        // A carrier under a key no typed parser knows and one container deeper than any arm walks:
        // the point of the scan is that the next item type cannot need a code change.
        let future = json!({
            "input": [{
                "type": "future_tool_call",
                "payload": {"nested": [{"blob": {"frames": {"type": "holographic_Screenshot"}}}]},
            }],
        });
        let found = find_image_carrier(&future).expect("the type is in the image family");
        assert!(found.contains("holographic_Screenshot"), "found: {found}");
        assert!(found.contains("type"), "found: {found}");

        // A carrier key in an arbitrary nested object.
        let keyed = json!({"a": {"b": [{"c": {"computer_screenshot": "file-1"}}]}});
        let found = find_image_carrier(&keyed).expect("the key is a carrier");
        assert!(
            found.contains("$.a.b[0].c.computer_screenshot"),
            "found: {found}"
        );

        // file_id alone is enough: a file reference cannot be shown to be a non-image file.
        assert!(find_image_carrier(&json!({"x": {"y": {"file_id": "file-1"}}})).is_some());
        // Any value under a carrier key (here a data: URL): the key alone is the carrier.
        assert!(
            find_image_carrier(&json!({"a": {"image_url": "data:image/png;base64,aGVsbG8="}}))
                .is_some()
        );
    }

    #[test]
    fn the_image_carrier_detector_does_not_scan_arbitrary_text() {
        // The word "image" in a text field is text; so is a data: URL in a text field.
        let prose = json!({
            "messages": [{"role": "user", "content": [{
                "type": "text",
                "text": "draw an image; image_url and file_id are not keys in this string",
            }]}],
        });
        assert_eq!(find_image_carrier(&prose), None);

        let data_url_text = json!({
            "messages": [{"role": "user", "content": [{
                "type": "text",
                "text": "data:image/png;base64,aGVsbG8=",
            }]}],
        });
        assert_eq!(find_image_carrier(&data_url_text), None);

        // A genuinely non-image part type stays accepted, and so does a plain text body.
        assert_eq!(find_image_carrier(&json!({"type": "input_audio"})), None);
        assert_eq!(find_image_carrier(&json!({"messages": []})), None);
    }

    #[test]
    fn the_image_carrier_detector_exempts_schema_names_only() {
        // Round-3 false positive and its narrowing. The immediate child keys of a JSON Schema name
        // map (properties/patternProperties/$defs/definitions) are names, not payload: a tool
        // parameter or a response_format schema may legitimately be called "image". Everything else
        // is scanned, so a real carrier under tools/arguments/metadata is refused fail-closed.
        let tool_schema = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "draw",
                    "parameters": {
                        "type": "object",
                        "properties": {"image": {"type": "string"}, "image_url": {"type": "string"}},
                    },
                },
            }],
        });
        assert_eq!(find_image_carrier(&tool_schema), None);

        let response_format = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"properties": {"image": {"type": "string"}}}},
            },
        });
        assert_eq!(find_image_carrier(&response_format), None);

        // The exemption is the *names*, not the subtree: each child schema value is still walked.
        let carrier_in_schema_value = json!({
            "tools": [{"type": "function", "function": {
                "name": "draw",
                "parameters": {"type": "object", "properties": {
                    "shot": {"default": {"image_url": "https://example.invalid/x"}},
                }},
            }}],
        });
        assert!(find_image_carrier(&carrier_in_schema_value).is_some());

        // Fail-closed, not silent: a nonstandard arguments object or metadata carrying a carrier is
        // refused rather than answered from text (review round 3, narrowing).
        let tool_arguments = json!({
            "messages": [{"role": "assistant", "tool_calls": [{
                "type": "function",
                "function": {"name": "draw", "arguments": {"image": "https://example.invalid/x"}},
            }]}],
        });
        assert!(find_image_carrier(&tool_arguments).is_some());
        assert!(
            find_image_carrier(&json!({"metadata": {"image_url": "https://example.invalid/x"}}))
                .is_some()
        );

        // A carrier in a media container is still found even beside tools.
        let real = json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ]}],
            "tools": [{"type": "function", "function": {"name": "draw"}}],
        });
        assert!(find_image_carrier(&real).is_some());
    }

    #[test]
    fn a_non_schema_value_under_a_schema_name_is_still_refused() {
        // Review round 3, second false-negative finding: the name exemption must depend on the
        // mapped value being a schema. A string/number/array, or an object with a non-keyword key,
        // under properties/patternProperties/$defs/definitions is payload: its carrier name is
        // matched and the value falls through to the normal scan. These are invalid JSON Schemas
        // but valid JSON the router accepts, so the detector is the only thing that can refuse them.
        let data_url = "data:image/png;base64,aGVsbG8=";
        let under_properties = json!({"tools": [{"type": "function", "function": {"parameters":
            {"properties": {"image": data_url}}}}]});
        assert!(find_image_carrier(&under_properties).is_some());

        let response_format = json!({"response_format": {"type": "json_schema", "json_schema":
            {"schema": {"properties": {"image_url": {"url": data_url}}}}}});
        assert!(find_image_carrier(&response_format).is_some());

        let nested = json!({"tools": [{"type": "function", "function": {"parameters": {
            "properties": {"outer": {"properties": {"image_url": {"url": data_url}}}}}}}]});
        assert!(find_image_carrier(&nested).is_some());

        let pattern = json!({"tools": [{"type": "function", "function": {"parameters": {
            "patternProperties": {"image": data_url}}}}]});
        assert!(find_image_carrier(&pattern).is_some());

        let definitions = json!({"tools": [{"type": "function", "function": {"parameters": {
            "definitions": {"image_url": 7}}}}]});
        assert!(find_image_carrier(&definitions).is_some());

        // Valid schemas under an image name stay exempt: bool schema, empty object, keyword-only.
        let boolean_schema = json!({"parameters": {"properties": {"image": true}}});
        assert_eq!(find_image_carrier(&boolean_schema), None);
        let empty_schema = json!({"parameters": {"properties": {"image": {}}}});
        assert_eq!(find_image_carrier(&empty_schema), None);
        let keyword_only = json!({"parameters": {"properties": {"image":
            {"type": "string", "description": "a picture"}}}});
        assert_eq!(find_image_carrier(&keyword_only), None);
    }

    #[test]
    fn standard_schema_keywords_outside_the_first_list_do_not_trip_the_refusal() {
        // Review round 3, third false positive: a legitimate schema using a standard keyword that
        // was missing from SCHEMA_KEYWORDS (minProperties, maxProperties, draft-07 dependencies,
        // OpenAPI annotation keys) was classified as payload and refused under a property named
        // image/image_url. These are text requests and must not be refused.
        let min_properties = json!({"tools": [{"type": "function", "function": {"parameters":
            {"properties": {"image": {"type": "object", "minProperties": 1}}}}}]});
        assert_eq!(find_image_carrier(&min_properties), None);

        let max_properties = json!({"parameters": {"properties": {"image_url":
            {"type": "object", "maxProperties": 3}}}});
        assert_eq!(find_image_carrier(&max_properties), None);

        let draft07_dependencies = json!({"parameters": {"properties": {"image":
            {"type": "object", "dependencies": {"shot": ["caption"]}}}}});
        assert_eq!(find_image_carrier(&draft07_dependencies), None);

        let defs_min_properties = json!({"parameters": {"$defs": {"image_url":
            {"type": "object", "minProperties": 1}}}});
        assert_eq!(find_image_carrier(&defs_min_properties), None);

        let openapi_annotations = json!({"parameters": {"properties": {"image":
            {"type": "string", "example": "a cat", "xml": {"name": "image"},
             "externalDocs": {"url": "https://example.invalid"},
             "discriminator": {"propertyName": "kind"}}}}});
        assert_eq!(find_image_carrier(&openapi_annotations), None);

        // Deliberate fail-closed policy: an unknown keyword cannot be proven to be a schema, so a
        // carrier name under it is refused rather than served.
        let unknown_keyword = json!({"parameters": {"properties": {"image":
            {"type": "string", "x-not-a-schema-keyword": 1}}}});
        assert!(find_image_carrier(&unknown_keyword).is_some());
    }
}
