//! The image parts of a chat request: discovery, fetching, preparation and prompt placement.
//!
//! The chat template renders every `image_url` / `input_image` part as one image placeholder
//! (`encoding/seam.rs`), so the rendered prompt has one image token per part, in message and part
//! order, and the placeholder's spelling in text stays text. [`take_chat_images`] collects the
//! parts in that order, and the handler pairs them with the placeholders, refusing a request
//! where the two counts differ (an `image_pil` part that carries no data).
//!
//! The protocol layer owns the wire shapes and the size limits; the model's vision front end,
//! reached through `PromptCodec::vision_codec`, owns decoding, canvas planning and token span
//! expansion. Each part is fetched (an http(s) URL) or taken inline (a base64 data URL) and
//! prepared by that front end: decoded, planned and padded exactly as the reference `load_image`
//! does. The limits are the serving recipe's (`deepseek-recipe-image` 0.2.0 `ImageLimits`): 600
//! images and 64 MiB a request, 32 MiB and 8192 pixels a side an image (4096 from the fifteenth
//! image on), and a long side of 512 at `detail: "low"`.

use std::fmt;
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use sparkflash_model::vision::{ImageSpan, PreparedImage, VisionCodec, VisionOptions};

use crate::error::ApiError;
use crate::multimodal::{ImagePart, parse_image_url};

/// Most images one request may carry.
pub const MAX_IMAGES: usize = 600;
/// Most encoded image bytes one request may carry.
pub const MAX_TOTAL_BYTES: usize = 64 << 20;
/// Most encoded bytes one image may have.
pub const MAX_IMAGE_BYTES: usize = 32 << 20;
/// Longest http(s) image URL accepted (MAX_IMAGE_URL_LEN in deepseek-recipe-core); a data URL
/// is bounded by MAX_IMAGE_BYTES instead.
pub const MAX_URL_LEN: usize = 8192;
/// From this many images on, a side may be at most MAX_SIDE_MANY pixels.
const MANY_IMAGES: usize = 15;
const MAX_SIDE: u32 = 8192;
const MAX_SIDE_MANY: u32 = 4096;
/// Long side at detail: "low".
const LOW_DETAIL_SIDE: u32 = 512;
/// The recipe's fetch client: 10 s to connect, 60 s in all, five redirects.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const FETCH_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_REDIRECTS: u32 = 5;

/// One image as the engine receives it: where its span sits in the prompt and its canvas.
#[derive(Clone, PartialEq, Eq)]
pub struct VisionImage {
    /// The span's position, grid and content key.
    pub span: ImageSpan,
    /// The canvas: best_height * best_width * 3 bytes of RGB.
    pub pixels: Arc<[u8]>,
}

impl fmt::Debug for VisionImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VisionImage")
            .field("span", &self.span)
            .field("pixels", &self.pixels.len())
            .finish()
    }
}

/// Sum of the span lengths: the usage block's image_tokens.
#[must_use]
pub fn image_tokens(images: &[VisionImage]) -> usize {
    images.iter().map(|image| image.span.len()).sum()
}

/// An image part found in a chat request, in the order the template renders it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatImage {
    /// The parsed URL or inline payload.
    pub part: ImagePart,
    /// detail: "low" was asked for.
    pub low_detail: bool,
}

/// Collects the image parts of a chat body's messages, and returns a copy of the body with each
/// collected part replaced by an empty text part.
///
/// The copy is what the image-carrier scan runs on afterwards (see refuse_unread_image): an
/// image carrier it still finds is a shape this function did not take, so its image would reach no
/// placeholder, and the request is refused rather than answered without it (D27).
///
/// # Errors
///
/// A 400 for an image part with no string URL, an http(s) URL over MAX_URL_LEN, or a URL that
/// is not a base64 data URL or http(s) (MultimodalError).
pub fn take_chat_images(body: &Value) -> Result<(Vec<ChatImage>, Value), ApiError> {
    let mut stripped = body.clone();
    let mut images = Vec::new();
    let Some(messages) = stripped.get_mut("messages").and_then(Value::as_array_mut) else {
        return Ok((images, stripped));
    };
    for message in messages {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            let kind = part.get("type").and_then(Value::as_str);
            if !matches!(kind, Some("image_url" | "input_image")) {
                continue;
            }
            let (url, detail) = match part.get("image_url") {
                Some(Value::String(url)) => (url.as_str(), part.get("detail")),
                Some(Value::Object(object)) => match object.get("url") {
                    Some(Value::String(url)) => (
                        url.as_str(),
                        object.get("detail").or_else(|| part.get("detail")),
                    ),
                    _ => return Err(invalid_image("an image_url object has no string url")),
                },
                _ => {
                    return Err(invalid_image(
                        "an image part's image_url must be a string or an object with a string url",
                    ));
                }
            };
            let image = parse_image_url(url)?;
            // The recipe bounds the URLs it fetches; an inline image is bounded by its bytes.
            if matches!(image, ImagePart::Url { .. }) && url.len() > MAX_URL_LEN {
                return Err(invalid_image(format!(
                    "an image URL is {} bytes, over the {MAX_URL_LEN}-byte limit",
                    url.len()
                )));
            }
            images.push(ChatImage {
                part: image,
                low_detail: detail.and_then(Value::as_str) == Some("low"),
            });
            *part = json!({"type": "text", "text": ""});
        }
    }
    Ok((images, stripped))
}

/// Refuses a request whose body, after take_chat_images, still carries an image: one this
/// server would not show the model.
///
/// # Errors
///
/// An unsupported_capability 400 naming the shape and its JSON path.
pub fn refuse_unread_image(stripped: &Value) -> Result<(), ApiError> {
    let Some(shape) = crate::prompt::find_image_carrier(stripped) else {
        return Ok(());
    };
    Err(ApiError::invalid_request(
        format!(
            "the request body carries an image at {shape}, in a shape this server does not read \
             (it reads image_url and input_image content parts of chat messages); the request is \
             refused rather than answered without the image"
        ),
        None,
        Some("unsupported_capability".to_owned()),
    ))
}

fn invalid_image(message: impl Into<String>) -> ApiError {
    ApiError::invalid_request(
        message,
        Some("messages".to_owned()),
        Some("invalid_image".to_owned()),
    )
}

/// Whether http(s) image URLs are fetched: on unless SPARKFLASH_IMAGE_FETCH=0.
#[must_use]
pub fn fetch_enabled() -> bool {
    std::env::var("SPARKFLASH_IMAGE_FETCH").map_or(true, |value| value != "0")
}

/// Downloads one image, at most MAX_IMAGE_BYTES of it.
fn fetch(url: &str) -> Result<Vec<u8>, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(CONNECT_TIMEOUT)
        .timeout(FETCH_TIMEOUT)
        .redirects(MAX_REDIRECTS)
        .build();
    let response = agent.get(url).call().map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(MAX_IMAGE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "the image is over the {MAX_IMAGE_BYTES}-byte limit"
        ));
    }
    Ok(bytes)
}

/// Fetches and prepares a request's images, each on a blocking thread.
///
/// # Errors
///
/// An invalid_image 400 naming the image (1-based) and the reason: too many images or bytes, a
/// URL that could not be fetched (or fetching is off), a placeholder with no image, or a file
/// the model's VisionCodec refused.
pub async fn prepare_images(
    images: Vec<ChatImage>,
    fetch_urls: bool,
    vision: Arc<dyn VisionCodec>,
) -> Result<Vec<PreparedImage>, ApiError> {
    if images.len() > MAX_IMAGES {
        return Err(invalid_image(format!(
            "the request carries {} images, over the limit of {MAX_IMAGES}",
            images.len()
        )));
    }
    let max_dimension = if images.len() < MANY_IMAGES {
        MAX_SIDE
    } else {
        MAX_SIDE_MANY
    };
    let mut payloads: Vec<(Arc<[u8]>, bool)> = Vec::with_capacity(images.len());
    let mut fetches = Vec::new();
    for (index, image) in images.into_iter().enumerate() {
        let bytes: Arc<[u8]> = match image.part {
            ImagePart::Data { bytes, .. } => bytes,
            ImagePart::Url { url } => {
                if !fetch_urls {
                    return Err(invalid_image(format!(
                        "image {}: this server does not fetch image URLs \
                         (SPARKFLASH_IMAGE_FETCH=0); send the image as a base64 data URL",
                        index + 1
                    )));
                }
                fetches.push((index, tokio::task::spawn_blocking(move || fetch(&url))));
                Arc::from(Vec::new())
            }
            ImagePart::Placeholder => {
                return Err(invalid_image(format!(
                    "image {}: the prompt has an image placeholder with no image",
                    index + 1
                )));
            }
        };
        payloads.push((bytes, image.low_detail));
    }
    for (index, task) in fetches {
        let bytes = task
            .await
            .map_err(|error| invalid_image(format!("image {}: {error}", index + 1)))?
            .map_err(|error| {
                invalid_image(format!(
                    "image {}: fetching the URL failed: {error}",
                    index + 1
                ))
            })?;
        payloads[index].0 = bytes.into();
    }
    let total: usize = payloads.iter().map(|(bytes, _)| bytes.len()).sum();
    if total > MAX_TOTAL_BYTES {
        return Err(invalid_image(format!(
            "the request's images are {total} bytes, over the {MAX_TOTAL_BYTES}-byte limit"
        )));
    }
    let tasks: Vec<_> = payloads
        .into_iter()
        .map(|(bytes, low_detail)| {
            let options = VisionOptions {
                max_bytes: MAX_IMAGE_BYTES,
                max_dimension,
                max_long_side: low_detail.then_some(LOW_DETAIL_SIDE),
            };
            let vision = Arc::clone(&vision);
            tokio::task::spawn_blocking(move || vision.prepare(&bytes, &options))
        })
        .collect();
    let mut prepared = Vec::with_capacity(tasks.len());
    for (index, task) in tasks.into_iter().enumerate() {
        let image = task
            .await
            .map_err(|error| invalid_image(format!("image {}: {error}", index + 1)))?
            .map_err(|error| invalid_image(format!("image {}: {error}", index + 1)))?;
        prepared.push(image);
    }
    Ok(prepared)
}

/// Replaces each placeholder in prompt with its image's span.
///
/// # Errors
///
/// An invalid_image 400 when the placeholders and the images differ in number.
pub fn place_images(
    prompt: &[u32],
    prepared: Vec<PreparedImage>,
    vision: &dyn VisionCodec,
) -> Result<(Vec<u32>, Vec<VisionImage>), ApiError> {
    let (ids, spans) = vision
        .expand(prompt, &prepared)
        .map_err(|error| invalid_image(error.to_string()))?;
    let images = spans
        .into_iter()
        .zip(prepared)
        .map(|(span, image)| VisionImage {
            span,
            pixels: image.pixels,
        })
        .collect();
    Ok((ids, images))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::vision::ImageGrid;
    use sparkflash_model_deepseek_v41::vision::V41Vision;

    /// A 1x1 PNG, base64.
    const DOT: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

    fn vision() -> V41Vision {
        V41Vision
    }

    #[test]
    fn image_parts_are_taken_in_render_order_and_stripped() {
        let body = json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{DOT}"), "detail": "low"}},
                    {"type": "input_image", "image_url": "https://example.com/a.png"},
                ]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": "https://example.com/b.png"},
                ]},
            ]
        });
        let (images, stripped) = take_chat_images(&body).unwrap();
        assert_eq!(images.len(), 3);
        assert!(matches!(&images[0].part, ImagePart::Data { bytes, .. } if bytes.len() == 69));
        assert!(images[0].low_detail);
        assert_eq!(
            images[1].part,
            ImagePart::Url {
                url: "https://example.com/a.png".to_owned()
            }
        );
        assert!(!images[1].low_detail);
        assert_eq!(
            stripped["messages"][1]["content"][1],
            json!({"type": "text", "text": ""})
        );
        assert!(refuse_unread_image(&stripped).is_ok());
        assert_eq!(stripped["messages"][1]["content"][0]["text"], "first");
    }

    #[test]
    fn an_unknown_image_shape_survives_the_strip() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_pil", "image": "..."},
        ]}]});
        let (images, stripped) = take_chat_images(&body).unwrap();
        assert!(images.is_empty());
        let error = refuse_unread_image(&stripped).unwrap_err();
        assert!(error.to_string().contains("image_pil"), "{error}");
    }

    #[tokio::test]
    async fn inline_images_are_prepared_and_placed() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{DOT}")}},
        ]}]});
        let (images, _) = take_chat_images(&body).unwrap();
        let prepared = prepare_images(images, false, Arc::new(vision()))
            .await
            .unwrap();
        assert_eq!(prepared[0].grid.tokens(), 184);
        let id = vision().image_token_id();
        let (ids, placed) = place_images(&[1, id, 2], prepared, &vision()).unwrap();
        assert_eq!(ids.len(), 186);
        assert_eq!(placed[0].span.start, 1);
        assert_eq!(image_tokens(&placed), 184);
        assert_eq!(placed[0].pixels.len(), 546 * 546 * 3);
        assert!(place_images(&[1, 2], Vec::new(), &vision()).is_ok());
        assert_eq!(
            ImageGrid {
                n_llm_h: 1,
                n_llm_w: 1,
                best_height: 546,
                best_width: 546,
            }
            .tokens(),
            4
        );
    }

    #[tokio::test]
    async fn urls_are_refused_when_fetching_is_off_and_placeholders_without_images_always() {
        let url = vec![ChatImage {
            part: ImagePart::Url {
                url: "https://example.com/a.png".to_owned(),
            },
            low_detail: false,
        }];
        let error = prepare_images(url, false, Arc::new(vision()))
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("SPARKFLASH_IMAGE_FETCH"),
            "{error}"
        );
        let placeholder = vec![ChatImage {
            part: ImagePart::Placeholder,
            low_detail: false,
        }];
        assert!(
            prepare_images(placeholder, true, Arc::new(vision()))
                .await
                .is_err()
        );
        let id = vision().image_token_id();
        assert!(place_images(&[id, id], Vec::new(), &vision()).is_err());
    }
}
