//! Image file to 8-bit RGB, following the serving recipe (`deepseek-recipe-image` 0.2.0,
//! `opencv.rs`) where it and Pillow's `convert("RGB")` differ:
//!
//! * the format comes from the file header, not the declared media type;
//! * a side over [`DecodeLimits::max_dimension`] is refused from the header, before any pixels are
//!   decoded;
//! * 16-bit samples become `round(v / 257)` (Pillow keeps the high byte);
//! * an alpha channel, a PNG `tRNS` entry or a transparent GIF index is blended onto `#FDFDFD` as
//!   `(c * a + 253 * (255 - a) + 127) / 255` (Pillow drops alpha);
//! * grey is copied to the three channels;
//! * a GIF is its first frame on a canvas of the logical screen, transparent outside the frame; an
//!   animated WebP is its first frame;
//! * EXIF orientation is not applied (neither OpenCV's `IMREAD_UNCHANGED` nor Pillow's `open`
//!   applies it).

use std::fmt;
use std::io::Cursor;
use std::num::NonZeroU64;

use crate::VisionError;
use crate::resample::RgbImage;

/// What the transparent parts of an image are blended onto.
pub const BACKGROUND: u8 = 0xfd;

/// Size limits checked before decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Largest payload in bytes.
    pub max_bytes: usize,
    /// Largest width or height in pixels.
    pub max_dimension: u32,
}

impl DecodeLimits {
    /// The recipe's defaults: 32 MiB and 8192 pixels a side.
    pub const RECIPE: Self = Self {
        max_bytes: 32 << 20,
        max_dimension: 8192,
    };
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::RECIPE
    }
}

/// The supported file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// PNG, including APNG (its default image).
    Png,
    /// Baseline and progressive JPEG.
    Jpeg,
    /// GIF87a and GIF89a.
    Gif,
    /// Lossy, lossless and animated WebP.
    WebP,
}

impl ImageFormat {
    /// The format named by the file header, if it is a supported one.
    #[must_use]
    pub fn sniff(bytes: &[u8]) -> Option<Self> {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            Some(Self::Png)
        } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
            Some(Self::Jpeg)
        } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
            Some(Self::Gif)
        } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            Some(Self::WebP)
        } else {
            None
        }
    }

    /// The IANA media type.
    #[must_use]
    pub const fn media_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::WebP => "image/webp",
        }
    }
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Gif => "GIF",
            Self::WebP => "WebP",
        })
    }
}

/// Decodes an image file to RGB.
///
/// # Errors
///
/// [`VisionError::Empty`], [`VisionError::TooManyBytes`], [`VisionError::UnsupportedFormat`],
/// [`VisionError::TooLarge`], [`VisionError::ZeroSize`] or [`VisionError::Decode`].
pub fn decode(bytes: &[u8], limits: &DecodeLimits) -> Result<RgbImage, VisionError> {
    if bytes.is_empty() {
        return Err(VisionError::Empty);
    }
    if bytes.len() > limits.max_bytes {
        return Err(VisionError::TooManyBytes {
            bytes: bytes.len(),
            limit: limits.max_bytes,
        });
    }
    let format = ImageFormat::sniff(bytes).ok_or(VisionError::UnsupportedFormat)?;
    let fail = |detail: String| VisionError::Decode { format, detail };
    let image = match format {
        ImageFormat::Png => png_rgb(bytes, limits, &fail)?,
        ImageFormat::Jpeg => jpeg_rgb(bytes, limits, &fail)?,
        ImageFormat::Gif => gif_rgb(bytes, limits, &fail)?,
        ImageFormat::WebP => webp_rgb(bytes, limits, &fail)?,
    };
    debug_assert_eq!(image.data.len(), image.width * image.height * 3);
    Ok(image)
}

fn check(width: u32, height: u32, limits: &DecodeLimits) -> Result<(), VisionError> {
    if width == 0 || height == 0 {
        return Err(VisionError::ZeroSize);
    }
    if width > limits.max_dimension || height > limits.max_dimension {
        return Err(VisionError::TooLarge {
            width,
            height,
            limit: limits.max_dimension,
        });
    }
    Ok(())
}

/// OpenCV's `convertTo(CV_8U, 1 / 257)`: round to nearest (no sample lands on a tie).
fn from16(v: u16) -> u8 {
    ((u32::from(v) + 128) / 257) as u8
}

fn blend(c: u8, a: u8) -> u8 {
    let (c, a) = (u32::from(c), u32::from(a));
    ((c * a + u32::from(BACKGROUND) * (255 - a) + 127) / 255) as u8
}

/// Grey, grey + alpha, RGB or RGBA 8-bit samples to RGB.
fn to_rgb(width: usize, height: usize, channels: usize, samples: &[u8]) -> RgbImage {
    let mut data = Vec::with_capacity(width * height * 3);
    for px in samples.chunks_exact(channels).take(width * height) {
        match channels {
            1 => data.extend_from_slice(&[px[0]; 3]),
            2 => data.extend_from_slice(&[blend(px[0], px[1]); 3]),
            3 => data.extend_from_slice(px),
            _ => data.extend(px[..3].iter().map(|&c| blend(c, px[3]))),
        }
    }
    RgbImage {
        width,
        height,
        data,
    }
}

fn png_rgb(
    bytes: &[u8],
    limits: &DecodeLimits,
    fail: &dyn Fn(String) -> VisionError,
) -> Result<RgbImage, VisionError> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    // Palette to RGB(A), low bit depths to 8, tRNS to an alpha channel; 16-bit stays 16-bit.
    decoder.set_transformations(png::Transformations::EXPAND);
    let info = decoder
        .read_header_info()
        .map_err(|e| fail(e.to_string()))?;
    check(info.width, info.height, limits)?;
    let mut reader = decoder.read_info().map_err(|e| fail(e.to_string()))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| fail("the frame size overflows".into()))?;
    let mut buf = vec![0; size];
    let out = reader
        .next_frame(&mut buf)
        .map_err(|e| fail(e.to_string()))?;
    let (w, h) = (out.width as usize, out.height as usize);
    let channels = out.color_type.samples();
    let samples = match out.bit_depth {
        png::BitDepth::Eight => {
            // Rows are packed when the depth is 8, so the frame is one run of samples.
            buf.truncate(out.line_size * h);
            buf
        }
        png::BitDepth::Sixteen => buf[..out.line_size * h]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| from16(u16::from_be_bytes(b)))
            .collect(),
        depth => return Err(fail(format!("unexpanded bit depth {depth:?}"))),
    };
    if !matches!(channels, 1..=4) {
        return Err(fail(format!("{channels} channels")));
    }
    Ok(to_rgb(w, h, channels, &samples))
}

fn jpeg_rgb(
    bytes: &[u8],
    limits: &DecodeLimits,
    fail: &dyn Fn(String) -> VisionError,
) -> Result<RgbImage, VisionError> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;

    // The decoder's own size cap sits above ours so that ours reports the error.
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(usize::from(u16::MAX))
        .set_max_height(usize::from(u16::MAX));
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder.decode_headers().map_err(|e| fail(e.to_string()))?;
    let info = decoder
        .info()
        .ok_or_else(|| fail("no frame header".into()))?;
    check(u32::from(info.width), u32::from(info.height), limits)?;
    let pixels = decoder.decode().map_err(|e| fail(e.to_string()))?;
    let channels = decoder
        .output_colorspace()
        .map_or(0, |c| c.num_components());
    let (w, h) = (usize::from(info.width), usize::from(info.height));
    if !matches!(channels, 1 | 3) || pixels.len() < w * h * channels {
        return Err(fail(format!(
            "{} bytes in {channels} channels for {w}x{h}",
            pixels.len()
        )));
    }
    Ok(to_rgb(w, h, channels, &pixels))
}

fn gif_rgb(
    bytes: &[u8],
    limits: &DecodeLimits,
    fail: &dyn Fn(String) -> VisionError,
) -> Result<RgbImage, VisionError> {
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::RGBA);
    let cap = u64::from(limits.max_dimension).pow(2) * 4;
    options.set_memory_limit(gif::MemoryLimit::Bytes(
        NonZeroU64::new(cap.max(1)).expect("nonzero"),
    ));
    let mut decoder = options
        .read_info(Cursor::new(bytes))
        .map_err(|e| fail(e.to_string()))?;
    let (w, h) = (usize::from(decoder.width()), usize::from(decoder.height()));
    check(w as u32, h as u32, limits)?;
    let frame = decoder
        .read_next_frame()
        .map_err(|e| fail(e.to_string()))?
        .ok_or_else(|| fail("no frames".into()))?;
    // The first frame on a transparent canvas the size of the logical screen, clipped to it.
    let mut canvas = vec![0_u8; w * h * 4];
    let (fw, fh) = (usize::from(frame.width), usize::from(frame.height));
    let (left, top) = (usize::from(frame.left), usize::from(frame.top));
    for y in 0..fh.min(h.saturating_sub(top)) {
        let n = fw.min(w.saturating_sub(left)) * 4;
        let src = &frame.buffer[y * fw * 4..][..n];
        canvas[((top + y) * w + left) * 4..][..n].copy_from_slice(src);
    }
    Ok(to_rgb(w, h, 4, &canvas))
}

fn webp_rgb(
    bytes: &[u8],
    limits: &DecodeLimits,
    fail: &dyn Fn(String) -> VisionError,
) -> Result<RgbImage, VisionError> {
    let mut decoder =
        image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(|e| fail(e.to_string()))?;
    let (w, h) = decoder.dimensions();
    check(w, h, limits)?;
    let size = decoder
        .output_buffer_size()
        .ok_or_else(|| fail("the frame size overflows".into()))?;
    let mut buf = vec![0; size];
    decoder
        .read_image(&mut buf)
        .map_err(|e| fail(e.to_string()))?;
    let channels = if decoder.has_alpha() { 4 } else { 3 };
    Ok(to_rgb(w as usize, h as usize, channels, &buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixteen_bit_samples_round_to_nearest() {
        assert_eq!(from16(0), 0);
        assert_eq!(from16(128), 0);
        assert_eq!(from16(129), 1);
        assert_eq!(from16(257), 1);
        assert_eq!(from16(65535), 255);
        for v in 0..=u16::MAX {
            let exact = f64::from(v) / 257.0;
            assert_eq!(f64::from(from16(v)), exact.round(), "{v}");
        }
    }

    #[test]
    fn blending_keeps_opaque_pixels_and_fills_transparent_ones() {
        for c in 0..=255 {
            assert_eq!(blend(c, 255), c);
            assert_eq!(blend(c, 0), BACKGROUND);
        }
        assert_eq!(blend(0, 128), 126);
    }

    #[test]
    fn only_supported_headers_are_recognised() {
        assert_eq!(
            ImageFormat::sniff(b"\x89PNG\r\n\x1a\n...."),
            Some(ImageFormat::Png)
        );
        assert_eq!(
            ImageFormat::sniff(&[0xff, 0xd8, 0xff, 0xe0]),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(ImageFormat::sniff(b"GIF89a"), Some(ImageFormat::Gif));
        assert_eq!(
            ImageFormat::sniff(b"RIFF\0\0\0\0WEBPVP8 "),
            Some(ImageFormat::WebP)
        );
        assert_eq!(ImageFormat::sniff(b"BM6\0"), None);
        assert_eq!(
            decode(b"BM6\0", &DecodeLimits::RECIPE),
            Err(VisionError::UnsupportedFormat)
        );
        assert_eq!(decode(b"", &DecodeLimits::RECIPE), Err(VisionError::Empty));
    }
}
