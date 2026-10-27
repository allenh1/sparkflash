//! The image front-end contract: what the HTTP layer needs from a model's vision path.
//!
//! The protocol crate owns image part discovery, fetching and size accounting. A model pack owns
//! decoding, planning the canvas, padding, and expanding a prompt placeholder into the model's
//! image span. Keeping the span types here lets the router carry prepared images and the engine
//! request without linking a concrete vision crate.

use std::fmt;
use std::sync::Arc;

/// An image's canvas size and the token grid the model's aligner makes from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageGrid {
    /// Token rows.
    pub n_llm_h: usize,
    /// Token columns.
    pub n_llm_w: usize,
    /// Canvas height in pixels.
    pub best_height: usize,
    /// Canvas width in pixels.
    pub best_width: usize,
}

impl ImageGrid {
    /// Positions in the span: a start, each row's tokens and its newline, and an end.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.n_llm_h * (self.n_llm_w + 1) + 2
    }

    /// Aligner rows: the positions that carry image features.
    #[must_use]
    pub const fn image_rows(&self) -> usize {
        self.n_llm_h * self.n_llm_w
    }
}

/// What fills one position of an image span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpanRow {
    /// The image's start delimiter.
    Start,
    /// Aligner row n, in reading order.
    Image(usize),
    /// The newline delimiter after a token row.
    NewLine,
    /// The image's end delimiter.
    End,
}

/// Where one image sits in a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageSpan {
    /// First position of the span.
    pub start: usize,
    /// The image's grid.
    pub grid: ImageGrid,
    /// The canvas content key, used by the prefix cache to match images.
    pub key: [u8; 32],
}

impl ImageSpan {
    /// Positions in the span.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.grid.tokens()
    }

    /// Never true: a span has at least its start and end.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// One past the last position.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.start + self.len()
    }

    /// Whether position pos is in the span.
    #[must_use]
    pub const fn contains(&self, pos: usize) -> bool {
        self.start <= pos && pos < self.end()
    }

    /// What fills position pos, which must be in the span.
    #[must_use]
    pub fn row(&self, pos: usize) -> SpanRow {
        debug_assert!(self.contains(pos), "{pos} not in {self:?}");
        let offset = pos - self.start;
        if offset == 0 {
            return SpanRow::Start;
        }
        if offset == self.len() - 1 {
            return SpanRow::End;
        }
        let width = self.grid.n_llm_w;
        let (row, column) = ((offset - 1) / (width + 1), (offset - 1) % (width + 1));
        if column == width {
            SpanRow::NewLine
        } else {
            SpanRow::Image(row * width + column)
        }
    }
}

/// One image ready for the model's vision tower.
#[derive(Clone, PartialEq, Eq)]
pub struct PreparedImage {
    /// Canvas size and token grid.
    pub grid: ImageGrid,
    /// The canvas: best_height * best_width * 3 bytes of RGB, rows top to bottom.
    pub pixels: Arc<[u8]>,
    /// The canvas content key.
    pub key: [u8; 32],
}

impl fmt::Debug for PreparedImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedImage")
            .field("grid", &self.grid)
            .field("pixels", &self.pixels.len())
            .finish()
    }
}

/// How a model's image preparation treats one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionOptions {
    /// Largest payload in bytes.
    pub max_bytes: usize,
    /// Largest width or height in pixels.
    pub max_dimension: u32,
    /// Low detail: shrink the image so its long side is at most this many pixels before planning.
    pub max_long_side: Option<u32>,
}

/// Why the vision path could not prepare or place an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionError {
    message: String,
    internal: bool,
}

impl VisionError {
    /// A client-visible refusal.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            internal: false,
        }
    }

    /// A server-side failure.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            internal: true,
        }
    }

    /// The human-readable reason.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether this is a server-side failure rather than a bad request.
    #[must_use]
    pub fn is_internal(&self) -> bool {
        self.internal
    }
}

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for VisionError {}

/// A model's image front end.
pub trait VisionCodec: Send + Sync {
    /// The placeholder token the template writes, and the id of every position in a span.
    fn image_token_id(&self) -> u32;

    /// Decodes, plans and pads one image file.
    ///
    /// # Errors
    /// A refusal when the payload is empty, too large, malformed or an unsupported format.
    fn prepare(&self, bytes: &[u8], options: &VisionOptions) -> Result<PreparedImage, VisionError>;

    /// Replaces each placeholder in prompt with its image's span, the images taken in order.
    ///
    /// # Errors
    /// A refusal when the placeholders and the images differ in number.
    fn expand(
        &self,
        prompt: &[u32],
        images: &[PreparedImage],
    ) -> Result<(Vec<u32>, Vec<ImageSpan>), VisionError>;
}
