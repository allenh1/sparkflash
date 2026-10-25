//! The image front end of the vision path: everything between an image file and the rows the ViT
//! reads, on the host and without a GPU.
//!
//! The model's reference preprocessing (checkpoint `inference/image_processor.py`) is
//!
//! 1. decode the file and `convert("RGB")`;
//! 2. plan the pixel size from the original size ([`plan::plan_grid`]);
//! 3. `ImageOps.pad` to that size with a 127 grey fill, which is a Pillow BICUBIC resize that keeps
//!    the aspect ratio and a centred paste ([`prep::pad`], [`resample::resize_bicubic`]);
//! 4. cut the canvas into 14x14 patches for the ViT, whose 3x3 aligner gives the token grid.
//!
//! This crate reproduces steps 2 and 3 bit for bit, so a canvas it builds is the one Pillow builds
//! from the same decoded pixels (the goldens in `tests/fixtures/prep` come from Pillow). Decoding
//! (step 1) follows the official serving recipe (`deepseek-recipe-image`) rather than Pillow where
//! the two differ: transparency is blended onto `#FDFDFD` and 16-bit samples are rounded to 8 bits,
//! where `convert("RGB")` drops alpha and truncates. JPEG and lossy WebP decoders are not
//! bit-identical across libraries; the decode tests bound the difference.
//!
//! The recipe also resizes with OpenCV and re-encodes the result as WebP before the reference sees
//! it. That step is not taken: the canvas comes straight from the decoded file.
//!
//! [`span`] lays out the image token span the language model sees and the type of each position.

pub mod decode;
pub mod plan;
pub mod prep;
pub mod resample;
pub mod span;

use thiserror::Error;

pub use decode::{DecodeLimits, ImageFormat, decode};
pub use plan::{Geometry, Grid, plan_grid};
pub use prep::{PrepOptions, PreparedImage, prepare};
pub use resample::{RgbImage, resize_bicubic};
pub use span::{IMAGE_TOKEN_ID, ImageSpan, SpanRow, TokenType};

/// Why an image could not be turned into a canvas.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VisionError {
    /// The payload was empty.
    #[error("the image is empty")]
    Empty,
    /// The payload is larger than the per-image limit.
    #[error("the image is {bytes} bytes, over the {limit}-byte limit")]
    TooManyBytes {
        /// Payload size.
        bytes: usize,
        /// The limit it broke.
        limit: usize,
    },
    /// The file header is not one of the supported formats.
    #[error("unsupported image format (PNG, JPEG, GIF and WebP are supported)")]
    UnsupportedFormat,
    /// The declared width or height is over the limit.
    #[error("the image is {width}x{height} pixels; neither side may exceed {limit}")]
    TooLarge {
        /// Declared width.
        width: u32,
        /// Declared height.
        height: u32,
        /// The limit it broke.
        limit: u32,
    },
    /// The image has no pixels.
    #[error("the image has a zero width or height")]
    ZeroSize,
    /// The decoder rejected the file.
    #[error("{format} decode failed: {detail}")]
    Decode {
        /// The format the header named.
        format: ImageFormat,
        /// The decoder's message.
        detail: String,
    },
    /// The prompt's image placeholders and the images do not pair up.
    #[error("the prompt has {placeholders} image placeholders but the request has {images} images")]
    PlaceholderCount {
        /// Placeholder tokens in the prompt.
        placeholders: usize,
        /// Images in the request.
        images: usize,
    },
}
