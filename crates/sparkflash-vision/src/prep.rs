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

//! Image file to ViT canvas: decode, plan, then `ImageOps.pad` as the reference's `load_image`
//! does, and the content key the prefix cache matches images by.

use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::VisionError;
use crate::decode::{DecodeLimits, ImageFormat, decode};
use crate::plan::{Geometry, Grid, plan_grid};
use crate::resample::{RgbImage, resize_bicubic};

/// The fill `load_image` pads with.
pub const PAD: u8 = 127;

/// How [`prepare`] treats an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepOptions {
    /// Size limits checked before decoding.
    pub limits: DecodeLimits,
    /// Low detail: shrink the decoded image so that its long side is at most this many pixels
    /// before planning. The recipe shrinks with OpenCV's `INTER_AREA`; this uses the same Pillow
    /// BICUBIC as the pad, so a low-detail canvas is not the recipe's bit for bit.
    pub max_long_side: Option<u32>,
    /// The vision tower's shape.
    pub geometry: Geometry,
}

impl Default for PrepOptions {
    fn default() -> Self {
        Self {
            limits: DecodeLimits::RECIPE,
            max_long_side: None,
            geometry: Geometry::V41,
        }
    }
}

/// One image ready for the ViT.
#[derive(Clone, PartialEq, Eq)]
pub struct PreparedImage {
    /// Canvas size and token grid.
    pub grid: Grid,
    /// The canvas: `best_height * best_width * 3` bytes of RGB, rows top to bottom.
    pub pixels: Arc<[u8]>,
    /// [`content_key`] of the canvas.
    pub key: [u8; 32],
    /// The file's format.
    pub format: ImageFormat,
    /// Decoded width and height, before any low-detail shrink.
    pub source: (usize, usize),
}

impl std::fmt::Debug for PreparedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedImage")
            .field("grid", &self.grid)
            .field("format", &self.format)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// Decodes, plans and pads one image file.
///
/// # Errors
///
/// Whatever [`decode`] refuses.
pub fn prepare(bytes: &[u8], options: &PrepOptions) -> Result<PreparedImage, VisionError> {
    let format = ImageFormat::sniff(bytes);
    let mut image = decode(bytes, &options.limits)?;
    let source = (image.width, image.height);
    if let Some(limit) = options.max_long_side {
        image = shrink_long_side(&image, limit as usize);
    }
    let grid = plan_grid(image.width, image.height, &options.geometry);
    let canvas = pad(&image, grid.best_width, grid.best_height);
    Ok(PreparedImage {
        grid,
        key: content_key(&grid, &canvas.data),
        pixels: canvas.data.into(),
        format: format.expect("decode sniffed it"),
        source,
    })
}

/// The recipe's `limit_image_mat` size: the long side scaled to `limit`, both sides rounded half
/// away from zero and at least 1.
fn shrink_long_side(image: &RgbImage, limit: usize) -> RgbImage {
    let long = image.width.max(image.height);
    if long <= limit {
        return image.clone();
    }
    let ratio = limit as f64 / long as f64;
    let w = ((image.width as f64 * ratio).round() as usize).max(1);
    let h = ((image.height as f64 * ratio).round() as usize).max(1);
    resize_bicubic(image, w, h)
}

/// `ImageOps.pad(image, (width, height), color=(127, 127, 127))` with its default BICUBIC and
/// centring.
#[must_use]
pub fn pad(image: &RgbImage, width: usize, height: usize) -> RgbImage {
    // ImageOps.contain
    let im_ratio = image.width as f64 / image.height as f64;
    let dest_ratio = width as f64 / height as f64;
    let (mut w, mut h) = (width, height);
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            h = (image.height as f64 / image.width as f64 * width as f64).round_ties_even()
                as usize;
        } else {
            w = (image.width as f64 / image.height as f64 * height as f64).round_ties_even()
                as usize;
        }
    }
    if (w, h) == (width, height) {
        return resize_bicubic(image, w, h);
    }
    let mut out = RgbImage::filled(width, height, [PAD; 3]);
    if w == 0 || h == 0 {
        // contain() returns an empty image, and pasting it changes nothing.
        return out;
    }
    let resized = resize_bicubic(image, w, h);
    if w != width {
        out.paste(
            &resized,
            ((width - w) as f64 * 0.5).round_ties_even() as usize,
            0,
        );
    } else {
        out.paste(
            &resized,
            0,
            ((height - h) as f64 * 0.5).round_ties_even() as usize,
        );
    }
    out
}

/// What the prefix cache compares images by: a SHA-256 of the canvas and its grid, so that two
/// files that pad to the same canvas share the cached rows.
#[must_use]
pub fn content_key(grid: &Grid, pixels: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"sparkflash-vision canvas v1\0");
    for v in [
        grid.n_llm_h,
        grid.n_llm_w,
        grid.best_height,
        grid.best_width,
    ] {
        hash.update((v as u64).to_le_bytes());
    }
    hash.update(pixels);
    hash.finalize().into()
}

/// The bf16 bits the ViT input takes for each byte value: `load_image`'s f32
/// `(v / 255 - 0.5) / 0.5`, rounded to nearest even.
#[must_use]
pub fn normalized_bf16() -> [u16; 256] {
    let mut lut = [0; 256];
    for (v, out) in lut.iter_mut().enumerate() {
        let x = (v as f32 / 255.0 - 0.5) / 0.5;
        let bits = x.to_bits();
        *out = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_canvas_sized_image_is_not_resampled() {
        let mut img = RgbImage::filled(14, 28, [1, 2, 3]);
        img.data[5] = 77;
        assert_eq!(pad(&img, 14, 28), img);
    }

    #[test]
    fn padding_centres_with_ties_to_even() {
        // 3 spare columns: round(1.5) = 2. 5 spare: round(2.5) = 2.
        let img = RgbImage::filled(4, 4, [9, 9, 9]);
        let out = pad(&img, 7, 4);
        let row: Vec<u8> = out.data[..21].iter().step_by(3).copied().collect();
        assert_eq!(row, [PAD, PAD, 9, 9, 9, 9, PAD]);
        let out = pad(&img, 4, 9);
        let col: Vec<u8> = (0..9).map(|y| out.data[y * 12]).collect();
        assert_eq!(col, [PAD, PAD, 9, 9, 9, 9, PAD, PAD, PAD]);
    }

    #[test]
    fn the_bf16_table_spans_minus_one_to_one() {
        let lut = normalized_bf16();
        assert_eq!(lut[0], 0xbf80); // -1.0
        assert_eq!(lut[255], 0x3f80); // 1.0
        assert_eq!(lut[127], 0xbb81); // -1/255 = -0.0039216 -> -0.0039215 (rounded up)
        assert!(lut.windows(2).all(|w| {
            let f = |b: u16| f32::from_bits(u32::from(b) << 16);
            f(w[0]) <= f(w[1])
        }));
    }

    #[test]
    fn the_key_covers_the_grid_and_the_pixels() {
        let g = plan_grid(640, 480, &Geometry::V41);
        let px = vec![3_u8; g.best_height * g.best_width * 3];
        let k = content_key(&g, &px);
        let mut px2 = px.clone();
        px2[100] = 4;
        assert_ne!(k, content_key(&g, &px2));
        let g2 = Grid {
            n_llm_h: g.n_llm_h + 1,
            ..g
        };
        assert_ne!(k, content_key(&g2, &px));
        assert_eq!(k, content_key(&g, &px));
    }
}
