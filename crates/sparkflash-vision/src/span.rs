//! The token span an image occupies in the prompt (`prepare_vl_inputs` and `image_token_types` in
//! the reference):
//!
//! ```text
//! [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
//! ```
//!
//! Every position carries [`IMAGE_TOKEN_ID`]; only its [`TokenType`] tells them apart. The IMAGE
//! positions take the aligner's rows in reading order, the other three the model's learned
//! `image_start`, `image_newline` and `image_end` vectors.

use crate::VisionError;
use crate::plan::Grid;
use crate::prep::PreparedImage;

/// The placeholder token and the id of every position in an image span (`image_token_id`).
pub const IMAGE_TOKEN_ID: u32 = 129_264;

/// What a prompt position holds (`TEXT`, `IMAGE_START`, ... in the reference).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum TokenType {
    /// An ordinary token.
    Text = -1,
    /// The span's first position.
    ImageStart = 0,
    /// One aligner row.
    Image = 1,
    /// The end of a token row.
    ImageNewLine = 2,
    /// The span's last position.
    ImageEnd = 3,
}

/// What fills one position of a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpanRow {
    /// `image_start`.
    Start,
    /// Aligner row `n` (`row * n_llm_w + column`).
    Image(usize),
    /// `image_newline`.
    NewLine,
    /// `image_end`.
    End,
}

impl SpanRow {
    /// The position's token type.
    #[must_use]
    pub const fn token_type(self) -> TokenType {
        match self {
            Self::Start => TokenType::ImageStart,
            Self::Image(_) => TokenType::Image,
            Self::NewLine => TokenType::ImageNewLine,
            Self::End => TokenType::ImageEnd,
        }
    }
}

/// Where one image sits in a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageSpan {
    /// First position of the span.
    pub start: usize,
    /// The image's grid.
    pub grid: Grid,
    /// The canvas's content key.
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

    /// Whether position `pos` is in the span.
    #[must_use]
    pub const fn contains(&self, pos: usize) -> bool {
        self.start <= pos && pos < self.end()
    }

    /// What fills position `pos`, which must be in the span.
    #[must_use]
    pub fn row(&self, pos: usize) -> SpanRow {
        debug_assert!(self.contains(pos), "{pos} not in {self:?}");
        let o = pos - self.start;
        if o == 0 {
            return SpanRow::Start;
        }
        if o == self.len() - 1 {
            return SpanRow::End;
        }
        let w = self.grid.n_llm_w;
        let (r, c) = ((o - 1) / (w + 1), (o - 1) % (w + 1));
        if c == w {
            SpanRow::NewLine
        } else {
            SpanRow::Image(r * w + c)
        }
    }
}

/// The span's token types in order (`image_token_types`).
#[must_use]
pub fn token_types(grid: &Grid) -> Vec<TokenType> {
    let mut types = Vec::with_capacity(grid.tokens());
    types.push(TokenType::ImageStart);
    for _ in 0..grid.n_llm_h {
        types.extend(std::iter::repeat_n(TokenType::Image, grid.n_llm_w));
        types.push(TokenType::ImageNewLine);
    }
    types.push(TokenType::ImageEnd);
    types
}

/// Replaces each [`IMAGE_TOKEN_ID`] in `prompt` with its image's span, the images taken in order.
///
/// # Errors
///
/// [`VisionError::PlaceholderCount`] when the placeholders and images differ in number.
pub fn expand(
    prompt: &[u32],
    images: &[PreparedImage],
) -> Result<(Vec<u32>, Vec<ImageSpan>), VisionError> {
    let placeholders = prompt.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
    if placeholders != images.len() {
        return Err(VisionError::PlaceholderCount {
            placeholders,
            images: images.len(),
        });
    }
    let extra: usize = images.iter().map(|i| i.grid.tokens() - 1).sum();
    let mut ids = Vec::with_capacity(prompt.len() + extra);
    let mut spans = Vec::with_capacity(images.len());
    let mut next = images.iter();
    for &t in prompt {
        if t != IMAGE_TOKEN_ID {
            ids.push(t);
            continue;
        }
        let image = next.next().expect("counted");
        let span = ImageSpan {
            start: ids.len(),
            grid: image.grid,
            key: image.key,
        };
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, span.len()));
        spans.push(span);
    }
    Ok((ids, spans))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::ImageFormat;

    fn image(n_llm_h: usize, n_llm_w: usize) -> PreparedImage {
        PreparedImage {
            grid: Grid {
                n_llm_h,
                n_llm_w,
                best_height: n_llm_h * 42,
                best_width: n_llm_w * 42,
            },
            pixels: Vec::new().into(),
            key: [n_llm_h as u8; 32],
            format: ImageFormat::Png,
            source: (1, 1),
        }
    }

    #[test]
    fn rows_follow_the_reference_layout() {
        let img = image(2, 3);
        let span = ImageSpan {
            start: 10,
            grid: img.grid,
            key: img.key,
        };
        assert_eq!(span.len(), 10);
        let rows: Vec<SpanRow> = (10..20).map(|p| span.row(p)).collect();
        use SpanRow::*;
        assert_eq!(
            rows,
            [
                Start,
                Image(0),
                Image(1),
                Image(2),
                NewLine,
                Image(3),
                Image(4),
                Image(5),
                NewLine,
                End
            ]
        );
        let types: Vec<TokenType> = rows.iter().map(|r| r.token_type()).collect();
        assert_eq!(types, token_types(&img.grid));
    }

    #[test]
    fn expansion_places_each_span_at_its_placeholder() {
        let p = IMAGE_TOKEN_ID;
        let (ids, spans) = expand(&[5, p, 6, 7, p], &[image(1, 1), image(2, 2)]).unwrap();
        assert_eq!(spans[0].start, 1);
        assert_eq!(spans[0].len(), 4);
        assert_eq!(spans[1].start, 7);
        assert_eq!(spans[1].len(), 8);
        assert_eq!(ids.len(), 15);
        assert_eq!(&ids[..1], &[5]);
        assert!(ids[1..5].iter().all(|&t| t == p));
        assert_eq!(&ids[5..7], &[6, 7]);
        assert!(ids[7..].iter().all(|&t| t == p));
        assert_eq!(
            expand(&[p, p], &[image(1, 1)]),
            Err(VisionError::PlaceholderCount {
                placeholders: 2,
                images: 1
            })
        );
    }
}
