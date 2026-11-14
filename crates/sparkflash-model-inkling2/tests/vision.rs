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

//! The vision tower against a tiny synthetic checkpoint: shapes, token count, determinism, and
//! the two properties that would be silent if they broke - the class token reaching the output
//! and the axial rope depending on both axes. Then the device tower against the host reference at
//! the released geometry, and the span check every rank of a group runs before it prefills.

use std::collections::BTreeMap;
use std::sync::Arc;

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::images::inkling_cubin;
use sparkflash_model::vision::{ImageGrid, ImageSpan, VisionImage};
use sparkflash_model_inkling2::InklingVisionTower;
use sparkflash_model_inkling2::config::InklingVisionConfig;
use sparkflash_model_inkling2::vision::{
    IMAGE_TOKEN_ID, InklingVision, MAX_MERGED_TOKENS, SpanCheck, patch_position, patch_values,
    plan_cells, plan_grid,
};
use sparkflash_weights::{DType, TensorSpec, WeightError, WeightSource};

/// A source whose tensors are the shapes the tower asks for, filled from a seed.
struct Filled {
    specs: BTreeMap<String, TensorSpec>,
    data: BTreeMap<String, Vec<u8>>,
}

impl Filled {
    fn new() -> Self {
        Self {
            specs: BTreeMap::new(),
            data: BTreeMap::new(),
        }
    }

    fn put(&mut self, name: &str, shape: Vec<usize>, scale: f32) {
        let elements: usize = shape.iter().product();
        let mut values = Vec::with_capacity(elements);
        for index in 0..elements {
            let seed = name.bytes().fold(1u64, |state, byte| {
                state.wrapping_mul(31).wrapping_add(u64::from(byte))
            });
            let mixed = seed
                .wrapping_add(index as u64)
                .wrapping_mul(6_364_136_223_846_793_005);
            values.push(((mixed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale);
        }
        self.put_values(name, shape, &values);
    }

    fn put_values(&mut self, name: &str, shape: Vec<usize>, values: &[f32]) {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| sparkflash_weights::quant::f32_to_bf16_bits(*value).to_le_bytes())
            .collect();
        self.specs.insert(
            name.to_owned(),
            TensorSpec {
                name: name.to_owned(),
                dtype: DType::BF16,
                shape,
            },
        );
        self.data.insert(name.to_owned(), bytes);
    }
}

impl WeightSource for Filled {
    fn names(&self) -> Vec<String> {
        self.specs.keys().cloned().collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.specs
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        self.data
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }
}

fn config() -> InklingVisionConfig {
    InklingVisionConfig {
        depth: 2,
        hidden_size: 8,
        heads: 2,
        patch_size: 2,
        temporal_patch_size: 1,
        adapter_merge_size: 2,
        projection_hidden_dim: 16,
        posemb_grid_size: 4,
        layer_norm_eps: 1e-5,
        out_hidden_size: 12,
    }
}

fn tower_source(config: &InklingVisionConfig) -> Filled {
    tower_source_scaled(config, |_, scale| scale)
}

/// The fixture's tensors with each matrix's scale chosen by `matrix(fan_in, fixture_scale)`.
fn tower_source_scaled(config: &InklingVisionConfig, matrix: impl Fn(usize, f32) -> f32) -> Filled {
    let mut source = Filled::new();
    let width = config.hidden_size;
    let intermediate = sparkflash_model_inkling2::vision::mlp_width(width);
    let prefix = "model.visual";
    source.put(
        &format!("{prefix}.backbone.embeddings.patch_embedding.weight"),
        vec![width, patch_values(config)],
        matrix(patch_values(config), 0.4),
    );
    source.put(
        &format!("{prefix}.backbone.embeddings.position_embedding"),
        vec![config.posemb_grid_size * config.posemb_grid_size, width],
        0.4,
    );
    source.put(
        &format!("{prefix}.backbone.embeddings.class_embedding"),
        vec![width],
        0.4,
    );
    source.put(
        &format!("{prefix}.backbone.embeddings.cls_position_embedding"),
        vec![width],
        0.4,
    );
    {
        let name = "ln_pre";
        source.put(&format!("{prefix}.backbone.{name}.weight"), vec![width], 1.0);
        source.put(&format!("{prefix}.backbone.{name}.bias"), vec![width], 0.1);
    }
    for layer in 0..config.depth {
        let p = format!("{prefix}.backbone.layers.{layer}");
        for name in ["ln_1", "ln_2"] {
            source.put(&format!("{p}.{name}.weight"), vec![width], 1.0);
            source.put(&format!("{p}.{name}.bias"), vec![width], 0.1);
        }
        for name in ["ls_1", "ls_2"] {
            source.put(&format!("{p}.{name}.gamma"), vec![width], 0.9);
        }
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            source.put(
                &format!("{p}.self_attn.{name}.weight"),
                vec![width, width],
                matrix(width, 0.5),
            );
            source.put(&format!("{p}.self_attn.{name}.bias"), vec![width], 0.05);
        }
        source.put(
            &format!("{p}.mlp.c_fc.weight"),
            vec![intermediate, width],
            matrix(width, 0.5),
        );
        source.put(&format!("{p}.mlp.c_fc.bias"), vec![intermediate], 0.05);
        source.put(
            &format!("{p}.mlp.c_proj.weight"),
            vec![width, intermediate],
            matrix(intermediate, 0.5),
        );
        source.put(&format!("{p}.mlp.c_proj.bias"), vec![width], 0.05);
    }
    let merge = config.adapter_merge_size;
    source.put(
        &format!("{prefix}.proj_in.weight"),
        vec![config.projection_hidden_dim, width * merge * merge],
        matrix(width * merge * merge, 0.4),
    );
    source.put(
        &format!("{prefix}.proj_out.weight"),
        vec![config.out_hidden_size, config.projection_hidden_dim],
        matrix(config.projection_hidden_dim, 0.4),
    );
    source
}

fn tower() -> (InklingVision, InklingVisionConfig) {
    let config = config();
    let source = tower_source(&config);
    let tower = InklingVision::load(&config, Arc::new(source)).expect("the tower loads");
    (tower, config)
}

/// A 40x40 image is a 20x20 patch grid at the fixture's 2-pixel patch, and a 2x2 merge makes it
/// 100 image tokens of 12 values each.
#[test]
fn the_tower_encodes_one_canvas_to_merged_tokens() {
    let (tower, config) = tower();
    let (grid_h, grid_w) = plan_grid(40, 40, &config).expect("the grid plans");
    assert_eq!((grid_h, grid_w), (20, 20));
    let pixels = vec![0.1f32; grid_h * grid_w * patch_values(&config)];
    let encoded = tower.encode(&pixels, grid_h, grid_w).expect("the encode");
    let merged = grid_h * grid_w / (config.adapter_merge_size * config.adapter_merge_size);
    assert_eq!(encoded.len(), merged * config.out_hidden_size);
    assert!(encoded.iter().all(|value| value.is_finite()));
    assert!(
        encoded.iter().any(|value| value.abs() > 1e-6),
        "the tower produced only zeros"
    );
}

/// The same canvas twice is the same tokens: no hash ordering, no uninitialised memory.
#[test]
fn the_tower_is_deterministic() {
    let (tower, config) = tower();
    let pixels: Vec<f32> = (0..6 * 6 * patch_values(&config))
        .map(|index| ((index * 37 % 101) as f32 - 50.0) / 64.0)
        .collect();
    let first = tower.encode(&pixels, 6, 6).expect("the first encode");
    let second = tower.encode(&pixels, 6, 6).expect("the second encode");
    assert_eq!(first, second);
}

/// The class token must reach the output: a tower that dropped it would still return the right
/// shape and the right count.
#[test]
fn the_class_token_reaches_the_image_tokens() {
    let (tower, config) = tower();
    let pixels: Vec<f32> = (0..6 * 6 * patch_values(&config))
        .map(|index| ((index * 13 % 71) as f32 - 35.0) / 48.0)
        .collect();
    let first = tower.encode(&pixels, 6, 6).expect("the encode");

    // Swap the class embedding for a different one and the tokens must move.
    let config2 = config.clone();
    let mut source = tower_source(&config2);
    let name = "model.visual.backbone.embeddings.class_embedding";
    let width = config2.hidden_size;
    let values: Vec<f32> = (0..width).map(|index| 0.25 * (index as f32 + 1.0)).collect();
    source.put_values(name, vec![width], &values);
    let other = InklingVision::load(&config2, Arc::new(source)).expect("the second tower");
    let second = other.encode(&pixels, 6, 6).expect("the second encode");
    assert_ne!(first, second, "the class token did not reach the output");
}

/// Axial rope means a patch's row and column both matter: a bright patch one row down and one
/// column right of the corner give different tokens from the corner and from each other.
#[test]
fn the_axial_rope_depends_on_both_axes() {
    let (tower, config) = tower();
    let patch = patch_values(&config);
    let merge = config.adapter_merge_size;
    let bright = |row: usize, column: usize| {
        let index = (0..36)
            .find(|index| patch_position(*index, 6, merge) == (row, column))
            .expect("the patch is on the grid");
        let mut pixels = vec![0.0f32; 6 * 6 * patch];
        for value in &mut pixels[index * patch..(index + 1) * patch] {
            *value = 1.0;
        }
        tower.encode(&pixels, 6, 6).expect("the encode")
    };
    let corner = bright(0, 0);
    let below = bright(1, 0);
    let right = bright(0, 1);
    assert_ne!(corner, below, "a patch one row down made the same tokens");
    assert_ne!(corner, right, "a patch one column right made the same tokens");
    assert_ne!(below, right, "the row and the column are interchangeable");
}

/// The canvas path is the planner, the processor's patches and the encode together: a 5x7 image
/// is padded to 2x2 cells of the fixture's 4 pixels.
#[test]
fn a_canvas_encodes_to_its_padded_cells() {
    let (tower, config) = tower();
    let rgb: Vec<u8> = (0..5 * 7 * 3).map(|index| (index * 11 % 256) as u8).collect();
    let encoded = tower.encode_canvas(&rgb, 5, 7).expect("the canvas encodes");
    assert_eq!(plan_cells(5, 7, &config), Ok((2, 2)));
    assert_eq!(encoded.len(), 4 * config.out_hidden_size);
    assert!(tower.encode_canvas(&rgb[1..], 5, 7).is_err(), "a short canvas");
}

/// The planner pads partial cells and refuses what the tower cannot encode.
#[test]
fn the_planner_pads_and_refuses_bad_canvases() {
    let config = config();
    assert_eq!(plan_grid(160, 160, &config), Ok((80, 80)));
    assert_eq!(plan_grid(160, 98, &config), Ok((80, 50)), "a partial cell pads");
    assert_eq!(plan_grid(1, 1, &config), Ok((2, 2)));
    assert!(plan_grid(0, 160, &config).is_err(), "an empty image");
    // 364 x 364 pixels is 91 x 91 cells of 4, over the 8,192-token cap.
    assert!(plan_grid(364, 364, &config).is_err(), "over the cap");
}

/// A grid over the token cap is refused rather than silently truncated.
#[test]
fn a_grid_over_the_token_cap_is_refused() {
    let (tower, config) = tower();
    let (grid_h, grid_w) = (182, 182);
    assert!(grid_h * grid_w / 4 > MAX_MERGED_TOKENS);
    let pixels = vec![0f32; grid_h * grid_w * patch_values(&config)];
    let error = tower.encode(&pixels, grid_h, grid_w).expect_err("the cap refuses it");
    assert!(error.contains("token cap"), "{error}");
}

/// The released tower's geometry (the width, heads, patch, merge and folded temporal patch of 2
/// the device kernels serve), two blocks deep and with a narrower merger so the test stays small.
fn released_geometry() -> InklingVisionConfig {
    InklingVisionConfig {
        depth: 2,
        hidden_size: 1024,
        heads: 16,
        patch_size: 20,
        temporal_patch_size: 2,
        adapter_merge_size: 2,
        projection_hidden_dim: 512,
        posemb_grid_size: 32,
        layer_norm_eps: 1e-5,
        out_hidden_size: 256,
    }
}

/// The smallest cosine between a device token and its host token.
const MIN_COSINE: f64 = 0.999;
/// The largest device-host difference over every value, relative to the largest host magnitude.
const RELATIVE_TOLERANCE: f64 = 2e-2;

/// The device tower is the host reference as launches. It walks bf16 between stages where the
/// host walks f32, so the two are held to a cosine and a relative bound rather than bit equality;
/// a stage in the wrong order (the rope's axes swapped, the merge read in raster order, a bias or
/// the class row dropped) moves the tokens far past both. The canvas is 70x110 so the bottom and
/// right cells are partly the processor's pad.
#[test]
fn the_device_tower_matches_the_host_reference() {
    if inkling_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Inkling cubin");
        return;
    }
    let config = released_geometry();
    // Matrices at a unit-variance gain, so the activations stay O(1) and attention does not
    // collapse onto one key (which would make the comparison depend on a rounding tie).
    let source = tower_source_scaled(&config, |fan_in, _| (3.0 / fan_in as f32).sqrt());
    let (height, width) = (70usize, 110usize);
    let rgb: Vec<u8> = (0..height * width * 3)
        .map(|index| {
            let (pixel, channel) = (index / 3, index % 3);
            let (y, x) = (pixel / width, pixel % width);
            ((y * 3 + x * 2 + channel * 70 + (y * x) % 17) % 256) as u8
        })
        .collect();

    let cuda = Cuda::load().expect("the driver");
    let mut device = InklingVisionTower::load(&cuda, &config, &source).expect("the device tower");
    let tokens = device.tokens(height, width).expect("the token count");
    assert_eq!(tokens, 2 * 3, "70x110 is 2x3 cells of 40 pixels");
    let values = tokens * config.out_hidden_size;
    let mut total = 0usize;
    let out = cuda.alloc(values * 4, &mut total).expect("the output");
    let made = device
        .encode(&cuda, &rgb, height, width, out)
        .expect("the device encode");
    assert_eq!(made, tokens);
    cuda.synchronize().expect("the encode completes");
    let mut bytes = vec![0u8; values * 4];
    cuda.copy_dtoh(&mut bytes, &out).expect("the read back");
    device.free(&cuda);
    let on_device: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();

    let host = InklingVision::load(&config, Arc::new(source)).expect("the host tower");
    let reference = host
        .encode_canvas(&rgb, height, width)
        .expect("the host encode");
    assert_eq!(reference.len(), values);

    let scale = reference
        .iter()
        .fold(0f64, |most, value| most.max(f64::from(value.abs())));
    let worst = on_device
        .iter()
        .zip(&reference)
        .fold(0f64, |most, (a, b)| most.max(f64::from((a - b).abs())));
    let mut lowest = f64::INFINITY;
    for (a, b) in on_device
        .chunks_exact(config.out_hidden_size)
        .zip(reference.chunks_exact(config.out_hidden_size))
    {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        lowest = lowest.min(dot / (norm(a) * norm(b)));
    }
    eprintln!(
        "device vs host: lowest token cosine {lowest:.6}, largest difference {worst:.4e} of a \
         largest magnitude {scale:.4e} ({:.3e})",
        worst / scale
    );
    assert!(scale > 1e-3, "the host tower produced only zeros");
    assert!(lowest >= MIN_COSINE, "a device token's cosine is {lowest}");
    assert!(
        worst <= RELATIVE_TOLERANCE * scale,
        "the device differs by {worst} at a scale of {scale}"
    );
}

/// The fixture's 5x7 canvas is 2x2 cells, so its span is a start, four placeholders and an end.
fn span_image(start: usize, canvas: bool) -> VisionImage {
    let grid = ImageGrid {
        n_llm_h: 2,
        n_llm_w: 2,
        best_height: 5,
        best_width: 7,
        line_stride: 2,
    };
    VisionImage {
        span: ImageSpan {
            start,
            grid,
            key: [7; 32],
        },
        pixels: if canvas {
            Arc::from(vec![0u8; 5 * 7 * 3])
        } else {
            Arc::from(&[][..])
        },
    }
}

/// A prompt with the fixture image at position 1: text, the span, text.
fn span_prompt() -> Vec<u32> {
    let mut tokens = vec![10, 200_005];
    tokens.extend([IMAGE_TOKEN_ID; 4]);
    tokens.extend([200_010, 11]);
    tokens
}

/// What the check passes: the codec's own spans, with or without the canvas on a rank that
/// does not encode.
#[test]
fn the_span_check_admits_the_codecs_spans() {
    let tokens = span_prompt();
    let encoding = SpanCheck::new(Some(config()), true);
    let mirroring = SpanCheck::new(Some(config()), false);
    encoding
        .check(&tokens, &[span_image(1, true)])
        .expect("rank 0 with the canvas");
    mirroring
        .check(&tokens, &[span_image(1, false)])
        .expect("a worker without it");
    encoding.check(&[1, 2, 3], &[]).expect("a text prompt");
    SpanCheck::new(None, false)
        .check(&[1, 2, 3], &[])
        .expect("a text prompt without a tower");
}

/// What the check refuses, each before any rank has started the request.
#[test]
fn the_span_check_refuses_what_a_rank_could_not_prefill() {
    let tokens = span_prompt();
    let check = SpanCheck::new(Some(config()), true);
    let refused = |tokens: &[u32], images: &[VisionImage], expected: &str| {
        let error = check.check(tokens, images).expect_err(expected);
        assert!(error.contains(expected), "{error} (expected {expected})");
    };
    let error = SpanCheck::new(None, false)
        .check(&tokens, &[span_image(1, false)])
        .expect_err("no tower");
    assert!(error.contains("no vision tower"), "{error}");

    let mut stride = span_image(1, true);
    stride.span.grid.line_stride = 3;
    refused(&tokens, &[stride], "no separator");

    let mut grid = span_image(1, true);
    grid.span.grid.n_llm_w = 3;
    grid.span.grid.line_stride = 3;
    refused(&tokens, &[grid], "makes 2x2");

    refused(&tokens[..6], &[span_image(1, true)], "past the");

    let mut cell = tokens.clone();
    cell[3] = 42;
    refused(&cell, &[span_image(1, true)], "not the placeholder");

    let mut stray = tokens.clone();
    stray[7] = IMAGE_TOKEN_ID;
    refused(&stray, &[span_image(1, true)], "outside every image");
    refused(&tokens, &[], "outside every image");

    let mut twice = tokens.clone();
    twice.extend(span_prompt());
    refused(&twice, &[span_image(1, true), span_image(4, true)], "inside the span before");
    // A second image that starts on the first one's end delimiter overlaps it too.
    refused(&twice, &[span_image(1, true), span_image(6, true)], "inside the span before");
    check
        .check(&twice, &[span_image(1, true), span_image(9, true)])
        .expect("two images, each its own span");

    refused(&tokens, &[span_image(1, false)], "canvas is 0 bytes");
}
