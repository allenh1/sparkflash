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

//! The fixtures from `tests/fixtures/generate.py`: canvases the reference `load_image` built with
//! Pillow, and small files of each format with what they must decode to.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use sparkflash_vision::prep::pad;
use sparkflash_vision::{DecodeLimits, Geometry, Grid, RgbImage, decode, plan_grid};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn lines(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(str::to_owned).collect())
        .collect()
}

/// `procedural()` in generate.py.
fn procedural(width: usize, height: usize, seed: usize) -> RgbImage {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let v = x * (3 + c)
                    + y * (5 + 2 * c)
                    + ((x * y) >> 5)
                    + ((x ^ y) & 63) * (c + 1)
                    + seed * 17
                    + c * 85;
                data.push((v & 255) as u8);
            }
        }
    }
    RgbImage {
        width,
        height,
        data,
    }
}

#[test]
fn canvases_match_the_reference_load_image() {
    let cases = lines(&fixtures().join("prep_cases.txt"));
    assert!(cases.len() >= 20);
    for case in cases {
        let n: Vec<usize> = case[..7].iter().map(|v| v.parse().unwrap()).collect();
        let (w, h, seed) = (n[0], n[1], n[2]);
        let want = Grid {
            n_llm_h: n[3],
            n_llm_w: n[4],
            best_height: n[5],
            best_width: n[6],
        };
        let grid = plan_grid(w, h, &Geometry::V41);
        assert_eq!(grid, want, "{w}x{h}");
        let canvas = pad(&procedural(w, h, seed), grid.best_width, grid.best_height);
        assert_eq!(hex(&canvas.data), case[7], "{w}x{h} seed {seed}");
    }
}

/// Largest per-sample difference allowed against Pillow's decode of a lossy file. zune-jpeg's IDCT
/// and 4:2:0 chroma upsampling are not libjpeg-turbo's (measured: 2 at 4:4:4, 4 at 4:2:0, mean
/// 0.4); image-webp matches libwebp exactly on the lossy file.
const LOSSY_MAX_DIFF: u8 = 4;
/// Largest mean difference allowed against Pillow's decode of a lossy file.
const LOSSY_MAX_MEAN: f64 = 0.5;

#[test]
fn files_decode_to_the_recipe_pixels() {
    let dir = fixtures().join("decode");
    let cases = lines(&dir.join("manifest.txt"));
    assert!(cases.len() >= 17);
    let limits = DecodeLimits::RECIPE;
    for case in cases {
        let name = &case[0];
        let bytes = std::fs::read(dir.join(name)).unwrap();
        let img = decode(&bytes, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
        let size: (usize, usize) = (case[2].parse().unwrap(), case[3].parse().unwrap());
        assert_eq!((img.width, img.height), size, "{name}");
        match case[1].as_str() {
            "exact" => assert_eq!(hex(&img.data), case[4], "{name}"),
            "lossy" => {
                let reference = decode(&std::fs::read(dir.join(&case[4])).unwrap(), &limits)
                    .unwrap_or_else(|e| panic!("{}: {e}", case[4]));
                assert_eq!((reference.width, reference.height), size);
                let diffs: Vec<u8> = img
                    .data
                    .iter()
                    .zip(&reference.data)
                    .map(|(a, b)| a.abs_diff(*b))
                    .collect();
                let max = diffs.iter().copied().max().unwrap();
                let mean = diffs.iter().map(|&d| f64::from(d)).sum::<f64>() / diffs.len() as f64;
                assert!(
                    max <= LOSSY_MAX_DIFF && mean <= LOSSY_MAX_MEAN,
                    "{name}: off by up to {max} (mean {mean:.3}) from Pillow"
                );
            }
            kind => panic!("{name}: unknown kind {kind}"),
        }
    }
}

#[test]
fn oversized_and_broken_files_are_refused() {
    use sparkflash_vision::VisionError;
    let dir = fixtures().join("decode");
    let png = std::fs::read(dir.join("rgb8.png")).unwrap();
    let small = DecodeLimits {
        max_bytes: 1 << 20,
        max_dimension: 36,
    };
    assert_eq!(
        decode(&png, &small),
        Err(VisionError::TooLarge {
            width: 37,
            height: 23,
            limit: 36
        })
    );
    let tiny = DecodeLimits {
        max_bytes: 10,
        max_dimension: 8192,
    };
    assert!(matches!(
        decode(&png, &tiny),
        Err(VisionError::TooManyBytes { .. })
    ));
    for name in ["rgb8.png", "q75_420.jpg", "anim.gif", "lossy.webp"] {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        let cut = &bytes[..bytes.len() / 2];
        // A truncated file is refused or, where the decoder is lenient, still the right size.
        if let Ok(img) = decode(cut, &DecodeLimits::RECIPE) {
            assert_eq!(img.data.len(), img.width * img.height * 3, "{name}");
        }
        let header = &bytes[..12.min(bytes.len())];
        assert!(decode(header, &DecodeLimits::RECIPE).is_err(), "{name}");
    }
}
