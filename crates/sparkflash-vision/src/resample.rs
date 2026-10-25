//! Pillow's BICUBIC resize of an 8-bit RGB image, bit for bit.
//!
//! A port of `ImagingResample` in Pillow's `libImaging/Resample.c` for `Image.resize(size,
//! BICUBIC)` with the default box and no `reducing_gap` (what `ImageOps.pad` calls):
//!
//! * per output index, `precompute_coeffs` gives the source window and the normalised f64 filter
//!   weights, `support = 2 * max(scale, 1)` wide (the filter is stretched when shrinking, which is
//!   Pillow's antialiasing);
//! * `normalize_coeffs_8bpc` turns them into 22-bit fixed point, rounding half away from zero;
//! * the horizontal pass runs first, and only when the width changes; then the vertical pass, only
//!   when the height changes. Each output is `clip8((1 << 21) + sum(src * k) >> 22)` in i32.

/// An 8-bit RGB image, rows top to bottom, three bytes a pixel.
#[derive(Clone, PartialEq, Eq)]
pub struct RgbImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` bytes.
    pub data: Vec<u8>,
}

impl RgbImage {
    /// An image filled with one colour.
    #[must_use]
    pub fn filled(width: usize, height: usize, rgb: [u8; 3]) -> Self {
        let mut data = Vec::with_capacity(width * height * 3);
        for _ in 0..width * height {
            data.extend_from_slice(&rgb);
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Copies `src` with its top-left corner at (`x`, `y`); `src` must fit.
    pub fn paste(&mut self, src: &RgbImage, x: usize, y: usize) {
        assert!(x + src.width <= self.width && y + src.height <= self.height);
        let row = src.width * 3;
        for r in 0..src.height {
            let at = ((y + r) * self.width + x) * 3;
            self.data[at..at + row].copy_from_slice(&src.data[r * row..(r + 1) * row]);
        }
    }
}

impl std::fmt::Debug for RgbImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RgbImage({}x{})", self.width, self.height)
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// `bicubic_filter` with a = -0.5.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = if x < 0.0 { -x } else { x };
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// One pass's filter: each output index reads `len` inputs from `start`, weighted by
/// `k[i * ksize..][..len]`.
struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    k: Vec<i32>,
}

/// `precompute_coeffs` for the box (0, `in_size`), then `normalize_coeffs_8bpc`.
fn coeffs(in_size: usize, out_size: usize) -> Coeffs {
    const SUPPORT: f64 = 2.0;
    // The box edges are C floats; (in1 - in0) is exact for any size the decoder accepts.
    let (in0, in1) = (0.0_f32, in_size as f32);
    let scale = f64::from(in1 - in0) / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut pre = vec![0.0_f64; out_size * ksize];
    let mut bounds = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = f64::from(in0) + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C's (int) cast truncates toward zero; a negative start clamps to 0 either way.
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) - xmin;
        let k = &mut pre[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0;
        for x in 0..xmax {
            let w = bicubic(((x + xmin) as f64 - center + 0.5) * ss);
            k[x as usize] = w;
            ww += w;
        }
        if ww != 0.0 {
            for w in &mut k[..xmax.max(0) as usize] {
                *w /= ww;
            }
        }
        bounds.push((xmin as usize, xmax.max(0) as usize));
    }
    let one = f64::from(1_u32 << PRECISION_BITS);
    let k = pre
        .iter()
        .map(|&w| {
            if w < 0.0 {
                (-0.5 + w * one) as i32
            } else {
                (0.5 + w * one) as i32
            }
        })
        .collect();
    Coeffs { ksize, bounds, k }
}

fn clip8(acc: i32) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `Image.resize((width, height), BICUBIC)` on an RGB image.
///
/// Both target sides must be at least 1. Equal sizes return a copy, as Pillow does.
#[must_use]
pub fn resize_bicubic(src: &RgbImage, width: usize, height: usize) -> RgbImage {
    assert!(width > 0 && height > 0, "resize to {width}x{height}");
    let mut img = src.clone();
    if width != src.width {
        let c = coeffs(src.width, width);
        let mut out = vec![0_u8; width * img.height * 3];
        for y in 0..img.height {
            let row = &img.data[y * img.width * 3..(y + 1) * img.width * 3];
            let dst = &mut out[y * width * 3..(y + 1) * width * 3];
            for (xx, &(xmin, len)) in c.bounds.iter().enumerate() {
                let k = &c.k[xx * c.ksize..xx * c.ksize + len];
                let mut ss = [1_i32 << (PRECISION_BITS - 1); 3];
                for (x, &kx) in k.iter().enumerate() {
                    let px = &row[(xmin + x) * 3..(xmin + x) * 3 + 3];
                    for ch in 0..3 {
                        ss[ch] += i32::from(px[ch]) * kx;
                    }
                }
                for ch in 0..3 {
                    dst[xx * 3 + ch] = clip8(ss[ch]);
                }
            }
        }
        img = RgbImage {
            width,
            height: img.height,
            data: out,
        };
    }
    if height != src.height {
        let c = coeffs(src.height, height);
        let stride = img.width * 3;
        let mut out = vec![0_u8; stride * height];
        for (yy, &(ymin, len)) in c.bounds.iter().enumerate() {
            let k = &c.k[yy * c.ksize..yy * c.ksize + len];
            let dst = &mut out[yy * stride..(yy + 1) * stride];
            for (i, d) in dst.iter_mut().enumerate() {
                let mut ss = 1_i32 << (PRECISION_BITS - 1);
                for (y, &ky) in k.iter().enumerate() {
                    ss += i32::from(img.data[(ymin + y) * stride + i]) * ky;
                }
                *d = clip8(ss);
            }
        }
        img = RgbImage {
            width: img.width,
            height,
            data: out,
        };
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weights_of_each_output_sum_to_one_in_fixed_point() {
        for (i, o) in [
            (10, 3),
            (3, 10),
            (1920, 1708),
            (2, 70),
            (8192, 42882),
            (7, 7),
        ] {
            let c = coeffs(i, o);
            for (xx, &(_, len)) in c.bounds.iter().enumerate() {
                let sum: i64 = c.k[xx * c.ksize..xx * c.ksize + len]
                    .iter()
                    .map(|&k| i64::from(k))
                    .sum();
                // Each weight rounds by at most half a unit.
                assert!(
                    (sum - (1 << PRECISION_BITS)).abs() <= len as i64,
                    "{i}->{o}"
                );
            }
        }
    }

    #[test]
    fn a_flat_image_stays_flat() {
        let img = RgbImage::filled(37, 23, [200, 7, 127]);
        let out = resize_bicubic(&img, 101, 9);
        assert_eq!(out, RgbImage::filled(101, 9, [200, 7, 127]));
    }

    #[test]
    fn the_same_size_is_a_copy() {
        let mut img = RgbImage::filled(5, 4, [0, 0, 0]);
        img.data[7] = 99;
        assert_eq!(resize_bicubic(&img, 5, 4), img);
    }
}
