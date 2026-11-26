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

//! The absorbed (V3-form) MLA block and its pooled sparse indexer, on the host.
//!
//! A stack whose layout states `mla_attention.form = absorbed` ([ModelConfig::absorbed_latent])
//! caches one normed key/value latent per position. A query reads the latent through `wkv_b`'s
//! key rows (the absorbed query), takes its softmax over the positions its indexer selected, and
//! the mixed latent leaves through `wkv_b`'s value rows: the expanded keys and values are never
//! built. The arithmetic is the decided contract of `docs/engine/08-glm-on-the-base.md`: BF16
//! projections and norms (the stated stream type, [super::stated]), the absorbed query, logits,
//! softmax and mixed latent in f32, the probabilities and the head outputs in BF16.
//!
//! The indexer pools its keys (`indexer.pool_compress`): a key is a LayerNorm of `wk(x)`, a
//! pool's key is a per-channel softmax of gate plus position over the pool's positions times
//! their keys, and a query scores each complete pool it can see by its heads' weighted ReLU
//! dots. It selects the best `index_topk / index_pool` pools ([topk::top_pools]) and, when the
//! layout states the tail, the incomplete pool's visible positions. A layer without its own
//! indexer reads the selection of the layer right before it, which must run one
//! (`modeling_glm5_next.py` 1105-1107, 1497-1506); the parser refuses any other order.

use super::stated;
use super::weight_view::WeightView;
use crate::config::ModelConfig;
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::quant;
use crate::weights::WeightSource;
use sparkflash_weights::topk;

/// One absorbed layer's caches, one row a position.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AbsorbedCache {
    /// The normed key/value latents, `kv_lora_rank` a position.
    latent: Vec<f32>,
    /// The indexer's normed keys, `index_head_dim` a position; empty on a sharing layer.
    index_keys: Vec<f32>,
    /// The indexer's pool-gate projections, `index_head_dim` a position.
    index_gates: Vec<f32>,
    /// One pooled key per complete pool, `index_head_dim` each.
    pool_keys: Vec<f32>,
    /// How many positions the caches hold.
    positions: usize,
}

/// Each position's selected positions, which the next layer reads when it shares this one's
/// indexer.
pub(crate) type Selection = Vec<Vec<usize>>;

fn shape(what: impl Into<String>, detail: impl Into<String>) -> EngineError {
    EngineError::Shape {
        what: what.into(),
        detail: detail.into(),
    }
}

fn inverse_sqrt(value: usize) -> f32 {
    (1.0 / (value as f64).sqrt()) as f32
}

/// The block's indexer weights and widths.
struct Indexer {
    keys: Vec<f32>,
    gates: Vec<f32>,
    queries: Vec<f32>,
    head_weights: Vec<f32>,
    ape: Vec<f32>,
    heads: usize,
    width: usize,
    pool: usize,
}

/// Runs one absorbed MLA block over `input` (`positions * dim`, already normed), writing the
/// block output into `out[..positions * dim]`. `carry` is the selection the layer before left;
/// the return is the one this layer leaves for the next (none for a sharing layer, whose
/// successor cannot share again).
///
/// # Errors
/// A stack this form does not run (a rotary width, no pool), a sharing layer with no carry, or
/// a weight read or shape error.
#[allow(clippy::too_many_arguments)] // the block's operands, as the other mixers take them
pub(crate) fn absorbed_attention_forward(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    input: &[f32],
    start_pos: usize,
    cache: &mut AbsorbedCache,
    carry: Option<Selection>,
    source: &dyn WeightSource,
    out: &mut [f32],
    limit: usize,
) -> Result<Option<Selection>, EngineError> {
    let what = format!("layer {layer} absorbed attention");
    let latent_form = config
        .absorbed_latent
        .ok_or_else(|| shape(&what, "the stack states no absorbed latent"))?;
    if config.rope_head_dim != 0 {
        return Err(shape(
            &what,
            format!(
                "the absorbed form here reads keys without a rotary part; the stack has a {}-wide one",
                config.rope_head_dim
            ),
        ));
    }
    if cache.positions != start_pos {
        return Err(shape(
            &what,
            format!(
                "the caches hold {} positions and the chunk starts at {start_pos}",
                cache.positions
            ),
        ));
    }
    let dim = config.dim;
    let positions = input.len() / dim.max(1);
    let (heads, head_dim) = (config.n_heads, config.head_dim);
    let (rank, value_dim) = (latent_form.kv_lora_rank, latent_form.v_head_dim);
    let q_lora = config.q_lora_rank;
    let eps = config.norm_eps as f32;
    let grid = config.dense_grid().unwrap_or(ScaleGrid::Block32);
    let attn = format!("{prefix}.attn");
    let norm = |name: &str| linalg::decode_by_name(source, &format!("{attn}.{name}"), limit);

    let mut query_residual = stated::project(
        config,
        source,
        &format!("{attn}.wq_a"),
        q_lora,
        dim,
        input,
        grid,
        limit,
    )?;
    stated::rms_norm(config, &mut query_residual, &norm("q_norm.weight")?, eps)?;
    let queries = stated::project(
        config,
        source,
        &format!("{attn}.wq_b"),
        heads * head_dim,
        q_lora,
        &query_residual,
        grid,
        limit,
    )?;
    let mut latent = stated::project(
        config,
        source,
        &format!("{attn}.wkv"),
        rank,
        dim,
        input,
        grid,
        limit,
    )?;
    stated::rms_norm(config, &mut latent, &norm("kv_norm.weight")?, eps)?;
    cache.latent.extend_from_slice(&latent);
    let expand_rows = heads * (head_dim + value_dim);
    let expand = WeightView::read_optional_scale(
        source,
        &format!("{attn}.wkv_b"),
        expand_rows,
        rank,
        grid,
        limit,
    )?
    .to_f32(expand_rows, rank)?;

    // The selection: this layer's own indexer, or the carry of the layer before.
    let own = config.index_source_layers.contains(&layer);
    let selection = if own {
        let indexer = read_indexer(config, &attn, input, &query_residual, source, grid, limit)?;
        select(config, &what, cache, &indexer, start_pos, positions)?
    } else {
        let carried = carry.ok_or_else(|| {
            shape(
                &what,
                "the layer shares an indexer, but the layer before it left no selection",
            )
        })?;
        if carried.len() != positions {
            return Err(shape(
                &what,
                format!(
                    "the carried selection covers {} positions, the chunk {positions}",
                    carried.len()
                ),
            ));
        }
        carried
    };
    cache.positions += positions;

    // The attention through the latent, per position and head.
    let scale = inverse_sqrt(head_dim + config.rope_head_dim);
    let mut mixed_heads = vec![0.0f32; positions * heads * value_dim];
    if latent_form.expanded {
        // The producer's own order (D125's parity mode): every cached position's latent is
        // expanded through `wkv_b` once into per-head keys and values, and each lands in the
        // producer's type, BF16, because `kv_b` is a module output. The absorbed form below is
        // the same linear map evaluated in another order; it never makes these two roundings.
        let total = cache.positions;
        let mut keys = vec![0.0f32; total * heads * head_dim];
        let mut values = vec![0.0f32; total * heads * value_dim];
        for position in 0..total {
            let cached = &cache.latent[position * rank..][..rank];
            for head in 0..heads {
                let rows = &expand[head * (head_dim + value_dim) * rank..][..head_dim * rank];
                let key = &mut keys[(position * heads + head) * head_dim..][..head_dim];
                for (slot, row) in key.iter_mut().zip(rows.chunks(rank)) {
                    *slot = quant::bf16_round(row.iter().zip(cached).map(|(w, c)| w * c).sum());
                }
                let rows =
                    &expand[(head * (head_dim + value_dim) + head_dim) * rank..][..value_dim * rank];
                let value = &mut values[(position * heads + head) * value_dim..][..value_dim];
                for (slot, row) in value.iter_mut().zip(rows.chunks(rank)) {
                    *slot = quant::bf16_round(row.iter().zip(cached).map(|(w, c)| w * c).sum());
                }
            }
        }
        for (row, chosen) in selection.iter().enumerate() {
            let mut logits = vec![0.0f32; chosen.len()];
            for head in 0..heads {
                let query = &queries[(row * heads + head) * head_dim..][..head_dim];
                for (logit, &position) in logits.iter_mut().zip(chosen) {
                    let key = &keys[(position * heads + head) * head_dim..][..head_dim];
                    // The producer's own types (`modeling_glm5_next.py` 1089): a BF16 matmul
                    // rounds its output, and the scale multiplies that BF16 value, so the
                    // logit the softmax sees is `bf16(bf16(q.k) * scale)`.
                    let dot = quant::bf16_round(
                        query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>(),
                    );
                    *logit = quant::bf16_round(dot * scale);
                }
                linalg::softmax(&mut logits);
                let head_out = &mut mixed_heads[(row * heads + head) * value_dim..][..value_dim];
                head_out.fill(0.0);
                for (probability, &position) in logits.iter().zip(chosen) {
                    let probability = if config.stated().bf16_stream {
                        quant::bf16_round(*probability)
                    } else {
                        *probability
                    };
                    let value = &values[(position * heads + head) * value_dim..][..value_dim];
                    for (slot, v) in head_out.iter_mut().zip(value) {
                        *slot += probability * v;
                    }
                }
            }
        }
        stated::hold(config, &mut mixed_heads);
        let projected = stated::project(
            config,
            source,
            &format!("{attn}.wo"),
            dim,
            heads * value_dim,
            &mixed_heads,
            grid,
            limit,
        )?;
        out[..projected.len()].copy_from_slice(&projected);
        return Ok(own.then_some(selection));
    }
    let mut absorbed = vec![0.0f32; rank];
    let mut mixed = vec![0.0f32; rank];
    for (row, chosen) in selection.iter().enumerate() {
        let mut logits = vec![0.0f32; chosen.len()];
        for head in 0..heads {
            let query = &queries[(row * heads + head) * head_dim..][..head_dim];
            let key_rows = &expand[head * (head_dim + value_dim) * rank..][..head_dim * rank];
            absorbed.fill(0.0);
            for (q, key_row) in query.iter().zip(key_rows.chunks(rank)) {
                for (slot, weight) in absorbed.iter_mut().zip(key_row) {
                    *slot += q * weight;
                }
            }
            for (logit, &position) in logits.iter_mut().zip(chosen) {
                let cached = &cache.latent[position * rank..][..rank];
                let dot: f32 = absorbed.iter().zip(cached).map(|(a, c)| a * c).sum();
                *logit = dot * scale;
            }
            linalg::softmax(&mut logits);
            mixed.fill(0.0);
            for (probability, &position) in logits.iter().zip(chosen) {
                let probability = if config.stated().bf16_stream {
                    quant::bf16_round(*probability)
                } else {
                    *probability
                };
                let cached = &cache.latent[position * rank..][..rank];
                for (slot, value) in mixed.iter_mut().zip(cached) {
                    *slot += probability * value;
                }
            }
            let value_rows =
                &expand[(head * (head_dim + value_dim) + head_dim) * rank..][..value_dim * rank];
            let head_out = &mut mixed_heads[(row * heads + head) * value_dim..][..value_dim];
            for (slot, value_row) in head_out.iter_mut().zip(value_rows.chunks(rank)) {
                *slot = value_row.iter().zip(&mixed).map(|(w, m)| w * m).sum();
            }
        }
    }
    stated::hold(config, &mut mixed_heads);
    let projected = stated::project(
        config,
        source,
        &format!("{attn}.wo"),
        dim,
        heads * value_dim,
        &mixed_heads,
        grid,
        limit,
    )?;
    out[..projected.len()].copy_from_slice(&projected);
    Ok(own.then_some(selection))
}

/// The indexer's projections over the chunk: the normed keys, the pool-gate rows, the queries
/// and the head weights.
fn read_indexer(
    config: &ModelConfig,
    attn: &str,
    input: &[f32],
    query_residual: &[f32],
    source: &dyn WeightSource,
    grid: ScaleGrid,
    limit: usize,
) -> Result<Indexer, EngineError> {
    let what = format!("{attn}.indexer");
    let wiring = config.stated();
    let pool = wiring.index_pool;
    if pool < 2 {
        return Err(shape(
            &what,
            "the absorbed form here runs a pooled indexer; the stack states no pool",
        ));
    }
    let eps = config.index_key_norm_eps as f32;
    if eps <= 0.0 {
        return Err(shape(
            &what,
            "the layout states no key norm epsilon (sparse_indexer.key_norm_eps)",
        ));
    }
    let (dim, heads, width) = (config.dim, config.index_n_heads, config.index_head_dim);
    let base = format!("{attn}.indexer");
    let mut keys = stated::project(
        config,
        source,
        &format!("{base}.wk"),
        width,
        dim,
        input,
        grid,
        limit,
    )?;
    let key_weight = linalg::decode_by_name(source, &format!("{base}.k_norm.weight"), limit)?;
    let key_bias = linalg::decode_by_name(source, &format!("{base}.k_norm.bias"), limit)?;
    linalg::layer_norm(&mut keys, &key_weight, &key_bias, width, eps)?;
    stated::hold(config, &mut keys);
    let gate = WeightView::read_plain(source, &format!("{base}.kpool_gate"), width, dim, limit)?;
    let mut gates = vec![0.0f32; input.len() / dim.max(1) * width];
    stated::project_view(
        config,
        source,
        &gate,
        &format!("{base}.kpool_gate"),
        (width, dim),
        input,
        &mut gates,
    )?;
    let queries = stated::project(
        config,
        source,
        &format!("{base}.wq_b"),
        heads * width,
        config.q_lora_rank,
        query_residual,
        grid,
        limit,
    )?;
    let head_scale = inverse_sqrt(heads);
    let head_weights = stated::project(
        config,
        source,
        &format!("{base}.weights_proj"),
        heads,
        dim,
        input,
        grid,
        limit,
    )?
    .into_iter()
    .map(|weight| weight * head_scale)
    .collect();
    let ape = linalg::decode_by_name(source, &format!("{base}.kpool_ape"), limit)?;
    if ape.len() != pool * width {
        return Err(shape(
            &what,
            format!(
                "kpool_ape holds {} values, not {pool} positions of {width}",
                ape.len()
            ),
        ));
    }
    Ok(Indexer {
        keys,
        gates,
        queries,
        head_weights,
        ape,
        heads,
        width,
        pool,
    })
}

/// Caches the chunk's index keys, pools each pool a position completes, and selects every
/// position's attended positions.
fn select(
    config: &ModelConfig,
    what: &str,
    cache: &mut AbsorbedCache,
    indexer: &Indexer,
    start_pos: usize,
    positions: usize,
) -> Result<Selection, EngineError> {
    let Indexer {
        heads, width, pool, ..
    } = *indexer;
    let budget = config.index_topk / pool;
    if budget == 0 {
        return Err(shape(
            what,
            format!(
                "index_topk {} selects no whole {pool}-position pool",
                config.index_topk
            ),
        ));
    }
    let bf16 = config.stated().bf16_stream;
    let score_scale = inverse_sqrt(width);
    let mut selection = Vec::with_capacity(positions);
    let mut probabilities = vec![0.0f32; pool];
    let mut scores = Vec::new();
    let mut chosen = Vec::new();
    for row in 0..positions {
        let position = start_pos + row;
        cache
            .index_keys
            .extend_from_slice(&indexer.keys[row * width..][..width]);
        cache
            .index_gates
            .extend_from_slice(&indexer.gates[row * width..][..width]);
        if (position + 1).is_multiple_of(pool) {
            let first = position + 1 - pool;
            for channel in 0..width {
                for (offset, probability) in probabilities.iter_mut().enumerate() {
                    *probability = cache.index_gates[(first + offset) * width + channel]
                        + indexer.ape[offset * width + channel];
                }
                linalg::softmax(&mut probabilities);
                let mut pooled = 0.0f32;
                for (offset, probability) in probabilities.iter().enumerate() {
                    let key = cache.index_keys[(first + offset) * width + channel];
                    pooled += if bf16 {
                        quant::bf16_round(quant::bf16_round(*probability) * key)
                    } else {
                        probability * key
                    };
                }
                cache.pool_keys.push(if bf16 {
                    quant::bf16_round(pooled)
                } else {
                    pooled
                });
            }
        }
        let query = &indexer.queries[row * heads * width..][..heads * width];
        let weights = &indexer.head_weights[row * heads..][..heads];
        scores.clear();
        for pooled in cache.pool_keys.chunks(width) {
            let score: f32 = weights
                .iter()
                .zip(query.chunks(width))
                .map(|(weight, head)| {
                    let dot: f32 = head.iter().zip(pooled).map(|(q, k)| q * k).sum();
                    weight * (dot * score_scale).max(0.0)
                })
                .sum();
            scores.push(score);
        }
        topk::top_pools(
            &scores,
            pool,
            budget,
            position,
            config.stated().index_tail,
            &mut chosen,
        );
        selection.push(chosen.clone());
    }
    Ok(selection)
}
