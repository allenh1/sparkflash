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

//! A grouped-query block inside the DeepSeek-shaped stack, on the host.
//!
//! The block's mHC entry, norms and FFN are the stack's; only its token mixer differs: the
//! `gqa_attn_config` heads project the normalised rows through `attn.wq`/`wk`/`wv`, turn query
//! and key by the stack's unscaled `rope_theta` over every float of a head, attend causally over
//! the block's own key/value cache and project back through `attn.wo`. No output gate and no
//! query/key norms: the grouped-query geometry states neither.

use super::weight_view::WeightView;
use crate::config::{GqaGeometry, ModelConfig};
use crate::linalg::{EngineError, ScaleGrid};
use crate::weights::WeightSource;

/// One grouped-query block's cache: every consumed position's turned key and its value,
/// `[positions][kv_heads * head_dim]` each.
#[derive(Debug, Clone, Default)]
pub struct GroupedCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl GroupedCache {
    /// The positions the cache holds.
    #[must_use]
    pub fn len(&self, geometry: &GqaGeometry) -> usize {
        self.keys
            .len()
            .checked_div(geometry.kv_heads * geometry.head_dim)
            .unwrap_or(0)
    }

    /// Whether the cache holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

fn shape(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}

/// Runs block `layer`'s grouped-query attention over `x` (`[rows][dim]`, the normalised block
/// input) at `start_pos`, appending every row's key and value to `cache`, and writes the projected
/// rows into `out` (`[rows][dim]`).
///
/// # Errors
///
/// [EngineError::Shape] when the stack states no grouped-query geometry, the cache does not end at
/// `start_pos`, or a weight is not the geometry's shape; a read error from `source`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouped_attention_forward(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    cache: &mut GroupedCache,
    source: &dyn WeightSource,
    out: &mut Vec<f32>,
    limit: usize,
) -> Result<(), EngineError> {
    let what = format!("layer {layer} grouped-query attention");
    let geometry = config.gqa.as_ref().ok_or_else(|| {
        shape(
            &what,
            "the block is gqa_attention and the stack states no gqa_attn_config".to_owned(),
        )
    })?;
    let dim = config.dim;
    if dim == 0 || !x.len().is_multiple_of(dim) {
        return Err(shape(
            &what,
            format!("{} input values are not whole {dim}-wide rows", x.len()),
        ));
    }
    let rows = x.len() / dim;
    if cache.len(geometry) != start_pos {
        return Err(shape(
            &what,
            format!(
                "the cache holds {} positions and the chunk starts at {start_pos}",
                cache.len(geometry)
            ),
        ));
    }
    let (heads, kv_heads, head_dim) = (geometry.heads, geometry.kv_heads, geometry.head_dim);
    if geometry.rotary > head_dim || !geometry.rotary.is_multiple_of(2) {
        return Err(shape(
            &what,
            format!(
                "the rope turns {} floats of a {head_dim}-wide head",
                geometry.rotary
            ),
        ));
    }
    let query_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let grid = config.dense_grid().unwrap_or(ScaleGrid::Block32);
    let project = |name: &str, outputs: usize, inputs: usize, input: &[f32]| {
        let view = WeightView::read_optional_scale(
            source,
            &format!("{prefix}.attn.{name}"),
            outputs,
            inputs,
            grid,
            limit,
        )?;
        let count = input.len() / inputs;
        let mut projected = vec![0.0f32; count * outputs];
        view.matmul(&mut projected, input, outputs, inputs, count)?;
        Ok::<_, EngineError>(projected)
    };
    let mut queries = project("wq", query_width, dim, x)?;
    let mut keys = project("wk", kv_width, dim, x)?;
    let values = project("wv", kv_width, dim, x)?;
    let theta = geometry.theta.max(1.0) as f32;
    let mut attended = vec![0.0f32; rows * query_width];
    for row in 0..rows {
        let position = start_pos + row;
        let query = &mut queries[row * query_width..(row + 1) * query_width];
        let key = &mut keys[row * kv_width..(row + 1) * kv_width];
        sparkflash_weights::gqa::rotate_half(
            query,
            (heads, head_dim),
            geometry.rotary,
            (theta, 1.0),
            position,
        );
        sparkflash_weights::gqa::rotate_half(
            key,
            (kv_heads, head_dim),
            geometry.rotary,
            (theta, 1.0),
            position,
        );
        cache.keys.extend_from_slice(key);
        cache
            .values
            .extend_from_slice(&values[row * kv_width..(row + 1) * kv_width]);
        let heads_out = sparkflash_weights::gqa::attend(
            query,
            (&cache.keys, &cache.values),
            (heads, kv_heads, head_dim),
        );
        attended[row * query_width..(row + 1) * query_width].copy_from_slice(&heads_out);
    }
    let projected = project("wo", dim, query_width, &attended)?;
    out.clear();
    out.extend_from_slice(&projected);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::SyntheticWeights;
    use crate::tensor::TensorPlan;

    fn grouped_tiny() -> ModelConfig {
        ModelConfig {
            gqa_layers: vec![1],
            gqa: Some(GqaGeometry {
                heads: 4,
                kv_heads: 2,
                head_dim: 16,
                rotary: 16,
                theta: 10_000.0,
            }),
            ..ModelConfig::tiny()
        }
    }

    fn rows(config: &ModelConfig, count: usize) -> Vec<f32> {
        (0..count * config.dim)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) * 0.01)
            .collect()
    }

    /// The writer plans the four projections and none of MLA's tensors at the grouped block.
    #[test]
    fn the_plan_holds_the_grouped_block_s_projections_only() {
        let config = grouped_tiny();
        let plan = TensorPlan::language_model(&config);
        for name in ["wq", "wk", "wv", "wo"] {
            assert!(plan.contains(&format!("layers.1.attn.{name}.weight")));
            assert!(!plan.contains(&format!("layers.0.attn.{name}.weight")));
        }
        for name in ["wq_a.weight", "wkv.weight", "attn_sink", "q_norm.weight"] {
            assert!(!plan.contains(&format!("layers.1.attn.{name}")));
            assert!(plan.contains(&format!("layers.0.attn.{name}")));
        }
        assert!(plan.contains("layers.1.attn_norm.weight"));
    }

    /// One chunk and the same rows a position at a time leave the same bits and the same cache, and
    /// a lone first position attends to its own value only.
    #[test]
    fn chunks_are_causal_and_split_anywhere() {
        let config = grouped_tiny();
        let source = SyntheticWeights::from_plan(&TensorPlan::language_model(&config), 3);
        let limit = 1 << 22;
        let x = rows(&config, 5);
        let mut whole_cache = GroupedCache::default();
        let mut whole = Vec::new();
        grouped_attention_forward(
            &config,
            1,
            "layers.1",
            &x,
            0,
            &mut whole_cache,
            &source,
            &mut whole,
            limit,
        )
        .unwrap();
        let mut split_cache = GroupedCache::default();
        let mut split = Vec::new();
        for (start, end) in [(0, 2), (2, 3), (3, 5)] {
            let mut out = Vec::new();
            grouped_attention_forward(
                &config,
                1,
                "layers.1",
                &x[start * config.dim..end * config.dim],
                start,
                &mut split_cache,
                &source,
                &mut out,
                limit,
            )
            .unwrap();
            split.extend_from_slice(&out);
        }
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&whole), bits(&split));
        assert_eq!(bits(&whole_cache.keys), bits(&split_cache.keys));
        assert_eq!(whole_cache.len(config.gqa.as_ref().unwrap()), 5);
        assert!(whole.iter().any(|value| value.to_bits() != 0));

        // The first position's output is wo over its own values, each query head reading its group's.
        let geometry = config.gqa.unwrap();
        let kv_width = geometry.kv_heads * geometry.head_dim;
        let mut expanded = Vec::new();
        for head in 0..geometry.heads {
            let kv_head = head / (geometry.heads / geometry.kv_heads);
            expanded.extend_from_slice(
                &whole_cache.values[kv_head * geometry.head_dim..(kv_head + 1) * geometry.head_dim],
            );
        }
        assert_eq!(whole_cache.values.len(), 5 * kv_width);
        let wo = WeightView::read_optional_scale(
            &source,
            "layers.1.attn.wo",
            config.dim,
            geometry.heads * geometry.head_dim,
            config.dense_grid().unwrap_or(ScaleGrid::Block32),
            limit,
        )
        .unwrap();
        let mut first = vec![0.0f32; config.dim];
        wo.matmul(&mut first, &expanded, config.dim, expanded.len(), 1)
            .unwrap();
        assert_eq!(bits(&first), bits(&whole[..config.dim]));
    }

    /// A stack holding a grouped-query block prefills in chunks with the bits of one whole pass,
    /// and its decode step reads the cache the prefill left.
    #[test]
    fn a_stack_with_a_grouped_block_prefills_in_any_chunks() {
        use crate::forward::{InferenceEngine, SequenceState};
        let config = grouped_tiny();
        let engine = || {
            let source = SyntheticWeights::from_plan(&TensorPlan::language_model(&config), 9);
            InferenceEngine::new(config.clone(), Box::new(source), 1 << 22).unwrap()
        };
        let prompt = [3u32, 17, 42, 5, 99, 1];
        let whole_engine = engine();
        let mut whole = SequenceState::new(&config);
        let one = whole_engine.prefill(&mut whole, &prompt).unwrap();
        let split_engine = engine();
        let mut split = SequenceState::new(&config);
        split_engine.prefill(&mut split, &prompt[..2]).unwrap();
        let two = split_engine.prefill(&mut split, &prompt[2..]).unwrap();
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&one.logits), bits(&two.logits));
        assert_eq!(
            whole.grouped[1].len(config.gqa.as_ref().unwrap()),
            prompt.len()
        );
        assert!(whole.grouped[0].is_empty());
        let next = whole_engine.decode(&mut whole, 7).unwrap();
        let next_split = split_engine.decode(&mut split, 7).unwrap();
        assert_eq!(bits(&next.logits), bits(&next_split.logits));
    }

    /// A chunk that does not start where the cache ends is refused by name.
    #[test]
    fn a_chunk_past_the_cache_is_refused() {
        let config = grouped_tiny();
        let source = SyntheticWeights::from_plan(&TensorPlan::language_model(&config), 3);
        let mut out = Vec::new();
        let error = grouped_attention_forward(
            &config,
            1,
            "layers.1",
            &rows(&config, 1),
            4,
            &mut GroupedCache::default(),
            &source,
            &mut out,
            1 << 22,
        )
        .unwrap_err();
        assert!(error.to_string().contains("starts at 4"), "{error}");
    }
}
