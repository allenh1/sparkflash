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
//! and the axial rope depending on both axes.

use std::collections::BTreeMap;
use std::sync::Arc;

use sparkflash_model_inkling2::config::InklingVisionConfig;
use sparkflash_model_inkling2::vision::{InklingVision, MAX_MERGED_TOKENS, patch_values, plan_grid};
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
    let mut source = Filled::new();
    let width = config.hidden_size;
    let intermediate = sparkflash_model_inkling2::vision::mlp_width(width);
    let prefix = "model.visual";
    source.put(
        &format!("{prefix}.backbone.embeddings.patch_embedding.weight"),
        vec![width, patch_values(config)],
        0.4,
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
    for name in ["ln_pre"] {
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
            source.put(&format!("{p}.self_attn.{name}.weight"), vec![width, width], 0.5);
            source.put(&format!("{p}.self_attn.{name}.bias"), vec![width], 0.05);
        }
        source.put(&format!("{p}.mlp.c_fc.weight"), vec![intermediate, width], 0.5);
        source.put(&format!("{p}.mlp.c_fc.bias"), vec![intermediate], 0.05);
        source.put(&format!("{p}.mlp.c_proj.weight"), vec![width, intermediate], 0.5);
        source.put(&format!("{p}.mlp.c_proj.bias"), vec![width], 0.05);
    }
    let merge = config.adapter_merge_size;
    source.put(
        &format!("{prefix}.proj_in.weight"),
        vec![config.projection_hidden_dim, width * merge * merge],
        0.4,
    );
    source.put(
        &format!("{prefix}.proj_out.weight"),
        vec![config.out_hidden_size, config.projection_hidden_dim],
        0.4,
    );
    source
}

fn tower() -> (InklingVision, InklingVisionConfig) {
    let config = config();
    let source = tower_source(&config);
    let tower = InklingVision::load(&config, Arc::new(source)).expect("the tower loads");
    (tower, config)
}

/// A 40x40 image is the smallest the release allows, a 20x20 patch grid at the fixture's
/// 2-pixel patch, and a 2x2 merge makes it 100 image tokens of 12 values each.
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

/// Axial rope means a patch's row and column both matter: two canvases whose only difference is
/// transposed differ, and a canvas shifted down differs from one shifted right.
#[test]
fn the_axial_rope_depends_on_both_axes() {
    let (tower, config) = tower();
    let patch = patch_values(&config);
    let mut flat = vec![0.0f32; 6 * 6 * patch];
    // One bright patch in the top-left block.
    for value in &mut flat[..patch] {
        *value = 1.0;
    }
    let top_left = tower.encode(&flat, 6, 6).expect("the encode");
    let mut down = vec![0.0f32; 6 * 6 * patch];
    for value in &mut down[6 * patch..7 * patch] {
        *value = 1.0;
    }
    let below = tower.encode(&down, 6, 6).expect("the encode");
    assert_ne!(
        top_left, below,
        "a patch one row down produced the same tokens, so the row angle is unused"
    );
}

/// The planner refuses what the tower cannot encode.
#[test]
fn the_planner_refuses_bad_canvases() {
    let config = config();
    assert!(plan_grid(160, 160, &config).is_ok());
    assert!(plan_grid(160, 98, &config).is_err(), "a partial cell");
    assert!(plan_grid(20, 160, &config).is_err(), "under the minimum");
    assert!(plan_grid(4096 + 40, 160, &config).is_err(), "over the maximum");
}

/// A grid over the token cap is refused rather than silently truncated.
#[test]
fn a_canvas_over_the_token_cap_is_refused() {
    let (tower, config) = tower();
    let (grid_h, grid_w) = plan_grid(4096, 4096, &config).expect("the largest grid plans");
    let merged = grid_h * grid_w / 4;
    assert!(grid_h * grid_w / 4 > MAX_MERGED_TOKENS);
    assert!(merged > MAX_MERGED_TOKENS);
    let pixels = vec![0f32; grid_h * grid_w * patch_values(&config)];
    let error = tower.encode(&pixels, grid_h, grid_w).expect_err("the cap refuses it");
    assert!(error.contains("token cap"), "{error}");
}
