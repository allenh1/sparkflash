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

//! Multimodal helpers: image-placeholder normalisation, image content-part parsing and
//! visual-token accounting.
//!
//! The authoritative placeholder is `<｜deepseek_image｜>` (token id 129264), built from
//! U+FF5C FULLWIDTH VERTICAL LINE, matching reference/encoding/encoding.py line 39 and the
//! added token row in the vendored tokenizer. Two upstream spellings have to be folded onto
//! it before any text reaches the engine:
//!
//! * `<｜image｜>`, the full-width placeholder exported by deepseek-recipe-core.
//! * ASCII-pipe variants such as `<|deepseek_image|>`.
//!
//! `<｜rl_image_start｜>` (129263) and `<｜rl_image_pad｜>` (129262) are distinct real
//! tokens used by the vision path; only their ASCII spellings are normalised.
//!
//! # No visual-token count is fabricated here
//!
//! The vision path is 263 `vision.*`/`aligner.*` tensors plus the three image delimiters, none of
//! which the tensor plan loads (DECISIONS.md D18), so no engine in this workspace can turn an image
//! part into visual tokens. This module therefore parses, normalises and labels image parts and
//! stops there: it does not assume a 32x32 patch grid or charge the 121 tokens such a grid would
//! imply. The router parses the parts, carries them (with the decoded payload bytes for inline
//! images) to the engine seam, and the engine refuses the request with a typed
//! [`crate::engine::MissingCapability`] naming the missing vision channel; a future vision front
//! end decodes those bytes into pixels and derives the real grid.

use std::fmt;
use std::sync::Arc;

use thiserror::Error;

/// Canonical DeepSeek image placeholder (U+FF5C fullwidth vertical lines).
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";
/// Canonical start-of-image run marker.
pub const RL_IMAGE_START: &str = "<｜rl_image_start｜>";
/// Canonical image padding marker.
pub const RL_IMAGE_PAD: &str = "<｜rl_image_pad｜>";
/// Raw-text image tag recognised by the official encoder (encoding.py lines 225-226).
pub const IMAGE_TAG_OPEN: &str = "<image>";
/// Closing raw-text image tag.
pub const IMAGE_TAG_CLOSE: &str = "</image>";

/// ViT patch size in pixels.
pub const VISION_PATCH_SIZE: u32 = 14;
/// 3x3 pixel-unshuffle downsample ratio.
pub const VISION_DOWNSAMPLE_RATIO: u32 = 3;
/// Upper bound on visual tokens per image, from reference/inference_config.json.
pub const VISION_MAX_IMAGE_TOKENS: usize = 1024;

/// Full-width aliases that mean the same image placeholder as the canonical spelling.
const CANONICAL_ALIASES: [(&str, &str); 1] = [("<｜image｜>", IMAGE_PLACEHOLDER)];

/// ASCII-pipe spellings and the canonical full-width forms they normalise to.
const ASCII_VARIANTS: [(&str, &str); 4] = [
    ("<|deepseek_image|>", IMAGE_PLACEHOLDER),
    ("<|image|>", IMAGE_PLACEHOLDER),
    ("<|rl_image_start|>", RL_IMAGE_START),
    ("<|rl_image_pad|>", RL_IMAGE_PAD),
];

/// Why an image content part could not be understood.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MultimodalError {
    /// The URL was neither a base64 `data:` URL nor an http(s) URL.
    #[error("image_url is not a supported base64 data URL or http(s) URL: {0}")]
    Unsupported(String),
    /// The `data:` URL was structurally invalid or its payload was not valid base64.
    #[error("malformed image data URL: {0}")]
    MalformedDataUrl(String),
}

/// One image reference extracted from a request.
///
/// The inline variant carries the decoded payload bytes, not just their length: a future vision
/// front end must decode the file bytes to pixels to derive the real token grid, and the engine
/// seam does not retain the request JSON, so a length alone would force it to re-read the request.
#[derive(Clone, PartialEq, Eq)]
pub enum ImagePart {
    /// Inline base64 payload decoded from a `data:` URL.
    Data {
        /// Declared media type, lower-cased, e.g. `image/png`.
        media_type: String,
        /// Decoded payload bytes: the image file (PNG/JPEG/...), not raw pixels.
        bytes: Arc<[u8]>,
    },
    /// Remote image; Milestone 1 does not fetch it (deferred to the vision path).
    Url {
        /// The http(s) URL.
        url: String,
    },
    /// An `<image>...</image>` tag or literal placeholder found in raw text.
    Placeholder,
}

impl ImagePart {
    /// Builds an inline part from a declared media type and the decoded payload bytes.
    #[must_use]
    pub fn data(media_type: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self::Data {
            media_type: media_type.into(),
            bytes: bytes.into(),
        }
    }

    /// Short stable kind name used in logs and tests.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Data { .. } => "data",
            Self::Url { .. } => "url",
            Self::Placeholder => "placeholder",
        }
    }
}

/// Prints the payload length rather than every byte, so a real image does not flood a log line.
impl fmt::Debug for ImagePart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data { media_type, bytes } => f
                .debug_struct("Data")
                .field("media_type", media_type)
                .field("byte_len", &bytes.len())
                .finish(),
            Self::Url { url } => f.debug_struct("Url").field("url", url).finish(),
            Self::Placeholder => f.write_str("Placeholder"),
        }
    }
}

/// Rewrites every known placeholder spelling onto its canonical U+FF5C form.
///
/// The result never contains an ASCII `|` that came from a placeholder spelling, and the
/// two `rl_image_*` markers are preserved as themselves.
#[must_use]
pub fn normalize_image_placeholders(text: &str) -> String {
    let mut out = text.to_owned();
    for (alias, canonical) in CANONICAL_ALIASES {
        if out.contains(alias) {
            out = out.replace(alias, canonical);
        }
    }
    for (ascii, canonical) in ASCII_VARIANTS {
        if out.contains(ascii) {
            out = out.replace(ascii, canonical);
        }
    }
    out
}

/// Replaces every complete `<image>...</image>` span with the canonical placeholder, as the
/// official encoder does. An unterminated opening tag is left untouched.
#[must_use]
pub fn replace_image_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(IMAGE_TAG_OPEN) {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + IMAGE_TAG_OPEN.len()..];
        let Some(end) = after_open.find(IMAGE_TAG_CLOSE) else {
            out.push_str(&rest[start..]);
            return out;
        };
        out.push_str(IMAGE_PLACEHOLDER);
        rest = &after_open[end + IMAGE_TAG_CLOSE.len()..];
    }
    out.push_str(rest);
    out
}

/// Full text preparation applied before token estimation: expand raw image tags, then
/// normalise every placeholder spelling.
#[must_use]
pub fn prepare_text(text: &str) -> String {
    normalize_image_placeholders(&replace_image_tags(text))
}

/// Counts canonical image placeholders in prepared text.
#[must_use]
pub fn count_image_placeholders(text: &str) -> usize {
    text.matches(IMAGE_PLACEHOLDER).count()
}

/// Visual tokens a `grid_h x grid_w` ViT patch grid produces after 3x3 pixel-unshuffle.
///
/// This is the published tiling arithmetic - ceil(grid / 3) tiles per axis, multiplied together -
/// for a front end that has **derived** its grid from the decoded image
/// (reference/inference/image_processor.py is the authoritative preprocessing). It is exposed as
/// arithmetic only: this crate never assumes a grid, so no request is charged a count from it.
#[must_use]
pub fn count_image_tokens(grid_h: u32, grid_w: u32) -> usize {
    let rows = grid_h.div_ceil(VISION_DOWNSAMPLE_RATIO) as usize;
    let cols = grid_w.div_ceil(VISION_DOWNSAMPLE_RATIO) as usize;
    rows * cols
}

/// Parses an OpenAI `image_url` value.
///
/// * `data:<media-type>;base64,<payload>` is decoded; only the length is retained.
/// * `http://` and `https://` URLs are accepted and represented as ImagePart::Url;
///   Milestone 1 never performs the fetch.
///
/// # Errors
/// Returns MultimodalError::MalformedDataUrl for broken data URLs and
/// MultimodalError::Unsupported for any other scheme.
pub fn parse_image_url(raw: &str) -> Result<ImagePart, MultimodalError> {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("data:") {
        let (meta, payload) = rest.split_once(',').ok_or_else(|| {
            MultimodalError::MalformedDataUrl("missing ',' payload separator".to_owned())
        })?;
        let (media_type, is_base64) = match meta.strip_suffix(";base64") {
            Some(media) => (media, true),
            None => (meta, false),
        };
        if !is_base64 {
            return Err(MultimodalError::MalformedDataUrl(
                "only base64 data URLs are supported".to_owned(),
            ));
        }
        let media_type = if media_type.is_empty() {
            "application/octet-stream".to_owned()
        } else {
            media_type.to_ascii_lowercase()
        };
        let decoded = decode_base64(payload)?;
        return Ok(ImagePart::data(media_type, decoded));
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Ok(ImagePart::Url {
            url: trimmed.to_owned(),
        });
    }
    Err(MultimodalError::Unsupported(trimmed.to_owned()))
}

/// Builds an Anthropic `{"type":"base64","media_type":...,"data":...}` source into the
/// equivalent data URL so both providers share one parser.
#[must_use]
pub fn anthropic_source_to_data_url(media_type: &str, data: &str) -> String {
    format!("data:{media_type};base64,{data}")
}

/// Strict RFC 4648 base64 decoder (standard alphabet, optional padding, whitespace
/// tolerated). Dependency-free so the API crate does not grow a base64 dependency.
fn decode_base64(payload: &str) -> Result<Vec<u8>, MultimodalError> {
    let mut out = Vec::with_capacity(payload.len() / 4 * 3);
    let mut buffer: u32 = 0;
    let mut bits: u32 = 0;
    for (index, byte) in payload.bytes().enumerate() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => {
                return Err(MultimodalError::MalformedDataUrl(format!(
                    "invalid base64 character at offset {index}"
                )));
            }
        };
        buffer = (buffer << 6) | u32::from(value);
        bits += 6;
        while bits >= 8 {
            bits -= 8;
            out.push(((buffer >> bits) & 0xFF) as u8);
        }
    }
    if bits >= 6 {
        return Err(MultimodalError::MalformedDataUrl(
            "truncated base64 group".to_owned(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_placeholder_spelling_normalises_onto_u_ff5c() {
        assert_eq!(
            normalize_image_placeholders("<｜image｜>"),
            "<｜deepseek_image｜>"
        );
        assert_eq!(
            normalize_image_placeholders("<|deepseek_image|>"),
            "<｜deepseek_image｜>"
        );
        assert_eq!(
            normalize_image_placeholders("<|image|>"),
            "<｜deepseek_image｜>"
        );
        assert_eq!(
            normalize_image_placeholders("<|rl_image_start|>"),
            "<｜rl_image_start｜>"
        );
        assert_eq!(
            normalize_image_placeholders("<|rl_image_pad|>"),
            "<｜rl_image_pad｜>"
        );
    }

    #[test]
    fn normalisation_output_never_keeps_an_ascii_pipe() {
        let mixed = normalize_image_placeholders(
            "before <|image|> mid <|rl_image_pad|> end <|deepseek_image|>",
        );
        assert!(!mixed.contains('|'));
        assert!(mixed.contains(IMAGE_PLACEHOLDER));
        assert!(mixed.contains(RL_IMAGE_PAD));
        assert_eq!(normalize_image_placeholders(&mixed), mixed, "idempotent");
    }

    #[test]
    fn canonical_markers_never_contain_an_ascii_pipe() {
        assert!(!IMAGE_PLACEHOLDER.contains('|'));
        assert!(IMAGE_PLACEHOLDER.contains('\u{ff5c}'));
        assert!(!RL_IMAGE_START.contains('|'));
        assert!(!RL_IMAGE_PAD.contains('|'));
    }

    #[test]
    fn rl_image_markers_are_not_rewritten_to_the_main_placeholder() {
        let out = normalize_image_placeholders("<|rl_image_start|>");
        assert_eq!(out, RL_IMAGE_START);
        assert_ne!(out, IMAGE_PLACEHOLDER);
    }

    #[test]
    fn image_tags_are_replaced_by_the_placeholder() {
        assert_eq!(
            replace_image_tags("a<image>xyz</image>b"),
            format!("a{IMAGE_PLACEHOLDER}b")
        );
        assert_eq!(
            replace_image_tags("<image>1</image><image>2</image>"),
            format!("{IMAGE_PLACEHOLDER}{IMAGE_PLACEHOLDER}")
        );
        assert_eq!(replace_image_tags("a<image>b"), "a<image>b");
        assert_eq!(replace_image_tags("no tags"), "no tags");
    }

    #[test]
    fn prepare_text_handles_tags_and_ascii_placeholders_together() {
        let prepared = prepare_text("<image>x</image> plus <|image|>");
        assert!(!prepared.contains('|'));
        assert!(!prepared.contains("<image>"));
        assert_eq!(count_image_placeholders(&prepared), 2);
    }

    #[test]
    fn count_image_tokens_matches_ceil_grid_over_three() {
        assert_eq!(count_image_tokens(0, 0), 0);
        assert_eq!(count_image_tokens(1, 1), 1);
        assert_eq!(count_image_tokens(3, 3), 1);
        assert_eq!(count_image_tokens(4, 3), 2);
        assert_eq!(count_image_tokens(3, 4), 2);
        assert_eq!(count_image_tokens(14, 14), 25);
        assert_eq!(count_image_tokens(14, 28), 50);
        assert_eq!(count_image_tokens(32, 32), 121);
        assert_eq!(
            count_image_tokens(33, 33),
            121,
            "33 is exactly divisible by 3"
        );
        assert_eq!(count_image_tokens(34, 34), 144);
    }

    #[test]
    fn data_urls_are_decoded_to_their_payload_bytes() {
        let part = parse_image_url("data:image/png;base64,aGVsbG8=").expect("hello");
        assert_eq!(part, ImagePart::data("image/png", b"hello".to_vec()));
        assert_eq!(part.kind(), "data");
        let empty_media = parse_image_url("data:;base64,////").expect("no media type");
        assert_eq!(
            empty_media,
            ImagePart::data("application/octet-stream", vec![0xFF, 0xFF, 0xFF])
        );
        let wrapped = parse_image_url("data:image/jpeg;base64, aGVs\nbG8 ").expect("wrapped");
        assert_eq!(wrapped, ImagePart::data("image/jpeg", b"hello".to_vec()));
    }

    #[test]
    fn http_urls_are_accepted_without_fetching() {
        let part = parse_image_url("https://example.invalid/a.png").expect("https");
        assert_eq!(
            part,
            ImagePart::Url {
                url: "https://example.invalid/a.png".to_owned()
            }
        );
        assert_eq!(part.kind(), "url");
        let plain = parse_image_url(" http://example.invalid/a.png ").expect("http");
        assert_eq!(plain.kind(), "url");
    }

    #[test]
    fn malformed_and_unsupported_urls_are_rejected() {
        assert!(matches!(
            parse_image_url("ftp://example.invalid/a.png"),
            Err(MultimodalError::Unsupported(_))
        ));
        assert!(matches!(
            parse_image_url("data:image/png;base64,!!!!"),
            Err(MultimodalError::MalformedDataUrl(_))
        ));
        assert!(matches!(
            parse_image_url("data:image/png,plaintext"),
            Err(MultimodalError::MalformedDataUrl(_))
        ));
        assert!(matches!(
            parse_image_url("data:image/png;base64"),
            Err(MultimodalError::MalformedDataUrl(_))
        ));
        assert!(matches!(
            parse_image_url("data:image/png;base64,AAAAA"),
            Err(MultimodalError::MalformedDataUrl(_))
        ));
    }

    #[test]
    fn anthropic_sources_round_trip_through_the_shared_parser() {
        let url = anthropic_source_to_data_url("image/png", "aGVsbG8=");
        let part = parse_image_url(&url).expect("decodes");
        assert_eq!(part, ImagePart::data("image/png", b"hello".to_vec()));
    }
}
