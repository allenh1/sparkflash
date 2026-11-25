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

//! A described checkpoint's values: the model-wide value object filled from the facts its
//! description carries, so the device forward and the host reference open any checkpoint a layout
//! describes from one reading.

use std::collections::BTreeMap;

use sparkflash_model::format::StoredFormat;
use sparkflash_model::{Architecture, FactValue};

use crate::config::ModelConfig;

/// A fact the description does not carry, or one of the wrong type, named.
pub type ConfigError = String;

fn int(facts: &BTreeMap<String, FactValue>, name: &str) -> Result<usize, ConfigError> {
    facts
        .get(name)
        .and_then(FactValue::as_integer)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("the plan carries no non-negative integer fact `{name}`"))
}

fn opt_int(facts: &BTreeMap<String, FactValue>, name: &str) -> Option<usize> {
    facts
        .get(name)
        .and_then(FactValue::as_integer)
        .and_then(|value| usize::try_from(value).ok())
}

fn float(facts: &BTreeMap<String, FactValue>, name: &str) -> Result<f64, ConfigError> {
    facts
        .get(name)
        .and_then(FactValue::as_float)
        .ok_or_else(|| format!("the plan carries no numeric fact `{name}`"))
}

fn opt_float(facts: &BTreeMap<String, FactValue>, name: &str) -> Option<f64> {
    facts.get(name).and_then(FactValue::as_float)
}

fn text(facts: &BTreeMap<String, FactValue>, name: &str) -> Result<String, ConfigError> {
    facts
        .get(name)
        .and_then(FactValue::as_text)
        .map(str::to_owned)
        .ok_or_else(|| format!("the plan carries no text fact `{name}`"))
}

fn integers(facts: &BTreeMap<String, FactValue>, name: &str) -> Result<Vec<usize>, ConfigError> {
    match facts.get(name) {
        Some(FactValue::Integers(values)) => values
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .map_err(|_| format!("the fact `{name}` carries a negative value"))
            })
            .collect(),
        _ => Err(format!("the plan carries no integer-list fact `{name}`")),
    }
}

fn boolean(facts: &BTreeMap<String, FactValue>, name: &str, default: bool) -> bool {
    match facts.get(name) {
        Some(FactValue::Bool(value)) => *value,
        _ => default,
    }
}

/// Fill the model-wide value object from a description's facts
/// ([Architecture::model_facts]) and its dense class's stored format.
///
/// Every field is read from the facts; no literal model value survives here.
///
/// # Errors
/// A fact the plan does not carry, or one of the wrong type. The forward refuses to open rather
/// than inventing a value for a field the description left out.
pub fn from_facts(
    facts: &BTreeMap<String, FactValue>,
    architecture: &Architecture,
    dense: Option<StoredFormat>,
) -> Result<ModelConfig, ConfigError> {
    // A fact is required only when a part that reads it is planned: the rotary base and its
    // YaRN bounds for a rotated attention, the grouped output's widths for an MLA with groups,
    // the compressor's base for a compressed layer, the indexer's for an indexed one. A stack
    // without the part carries zero, as the open manifest records an absent width.
    let rotated = int(facts, "rope_head_dim")? > 0;
    let grouped = architecture.layers.iter().any(|layer| {
        matches!(
            layer.attention.kind,
            sparkflash_model::architecture::AttentionKind::Mla {
                groups: Some(_),
                ..
            }
        )
    });
    let indexed = !integers(facts, "index_source_layers")?.is_empty();
    let compress_ratios = integers(facts, "compress_ratios")?;
    let compressed = compress_ratios.iter().any(|ratio| *ratio > 0);
    let by_part_int = |planned: bool, name: &str| -> Result<usize, ConfigError> {
        if planned {
            int(facts, name)
        } else {
            Ok(opt_int(facts, name).unwrap_or(0))
        }
    };
    let by_part_float = |planned: bool, name: &str| -> Result<f64, ConfigError> {
        if planned {
            float(facts, name)
        } else {
            Ok(opt_float(facts, name).unwrap_or(0.0))
        }
    };
    let absorbed_latent = match facts
        .get("language.mla_attention.form")
        .and_then(FactValue::as_text)
    {
        Some("absorbed") => Some(crate::config::AbsorbedLatent {
            kv_lora_rank: int(facts, "kv_lora_rank")?,
            v_head_dim: int(facts, "v_head_dim")?,
        }),
        Some("latent") | None => None,
        Some(other) => {
            return Err(format!(
                "the layout states the MLA form `{other}`; the base knows `latent` (one key/value latent every head reads) and `absorbed` (the V3 form through kv_b)"
            ));
        }
    };
    let n_layers = int(facts, "n_layers")?;
    let n_mtp_layers = opt_int(facts, "n_mtp_layers").unwrap_or(0);
    if compress_ratios.len() != n_layers + n_mtp_layers {
        return Err(format!(
            "the plan's compress ratios have {} entries for {n_layers}+{n_mtp_layers} layers",
            compress_ratios.len()
        ));
    }
    Ok(ModelConfig {
        vocab_size: int(facts, "vocab_size")?,
        dim: int(facts, "dim")?,
        moe_inter_dim: int(facts, "moe_inter_dim")?,
        dense_ffn_layers: match facts.get("dense_ffn_layers") {
            Some(_) => integers(facts, "dense_ffn_layers")?,
            None => Vec::new(),
        },
        dense_inter_dim: opt_int(facts, "dense_inter_dim").unwrap_or(0),
        gqa_layers: match facts.get("gqa_layers") {
            Some(_) => integers(facts, "gqa_layers")?,
            None => Vec::new(),
        },
        gqa: match facts.get("gqa_layers") {
            Some(_) => Some(crate::config::GqaGeometry {
                heads: int(facts, "gqa_heads")?,
                kv_heads: int(facts, "gqa_kv_heads")?,
                head_dim: int(facts, "gqa_head_dim")?,
                rotary: int(facts, "gqa_rotary")?,
                theta: float(facts, "gqa_rope_theta")?,
            }),
            None => None,
        },
        kda_layers: match facts.get("kda_layers") {
            Some(_) => integers(facts, "kda_layers")?,
            None => Vec::new(),
        },
        kda: match facts.get("kda_layers") {
            Some(_) => Some(crate::config::KdaGeometry {
                heads: int(facts, "kda_heads")?,
                head_dim: int(facts, "kda_head_dim")?,
                conv: int(facts, "kda_conv")?,
                lower_bound: float(facts, "kda_lower_bound")?,
            }),
            None => None,
        },
        absorbed_latent,
        gdn_layers: match facts.get("gdn_layers") {
            Some(_) => integers(facts, "gdn_layers")?,
            None => Vec::new(),
        },
        gdn: match facts.get("gdn_layers") {
            Some(_) => Some(crate::config::GdnGeometry {
                key_heads: int(facts, "gdn_key_heads")?,
                value_heads: int(facts, "gdn_value_heads")?,
                key_dim: int(facts, "gdn_key_dim")?,
                value_dim: int(facts, "gdn_value_dim")?,
                conv: int(facts, "gdn_conv")?,
            }),
            None => None,
        },
        n_layers,
        n_mtp_layers,
        dspark_block_size: opt_int(facts, "dspark_block_size").unwrap_or(0),
        dspark_noise_token_id: u32::try_from(opt_int(facts, "dspark_noise_token_id").unwrap_or(0))
            .map_err(|_| "the plan's noise token does not fit u32".to_owned())?,
        dspark_target_layer_ids: integers(facts, "dspark_target_layer_ids").unwrap_or_default(),
        dspark_markov_rank: int(facts, "dspark_markov_rank")?,
        dspark_n_routed_experts: opt_int(facts, "dspark_n_routed_experts").unwrap_or_default(),
        dspark_n_activated_experts: opt_int(facts, "dspark_n_activated_experts")
            .unwrap_or_default(),
        n_heads: int(facts, "n_heads")?,
        n_routed_experts: int(facts, "n_routed_experts")?,
        n_shared_experts: int(facts, "n_shared_experts")?,
        n_activated_experts: int(facts, "n_activated_experts")?,
        score_func: text(facts, "score_func")?,
        route_scale: float(facts, "route_scale")?,
        swiglu_limit: float(facts, "swiglu_limit")?,
        q_lora_rank: int(facts, "q_lora_rank")?,
        head_dim: int(facts, "head_dim")?,
        rope_head_dim: int(facts, "rope_head_dim")?,
        norm_eps: float(facts, "norm_eps")?,
        o_groups: by_part_int(grouped, "o_groups")?,
        o_lora_rank: by_part_int(grouped, "o_lora_rank")?,
        window_size: opt_int(facts, "window_size").unwrap_or(0),
        kv_source_layers: integers(facts, "kv_source_layers")?,
        index_source_layers: integers(facts, "index_source_layers")?,
        original_seq_len: by_part_int(rotated, "original_seq_len")?,
        rope_theta: by_part_float(rotated, "rope_theta")?,
        rope_factor: by_part_float(rotated, "rope_factor")?,
        beta_fast: by_part_float(rotated, "beta_fast")?,
        beta_slow: by_part_float(rotated, "beta_slow")?,
        index_n_heads: by_part_int(indexed, "index_n_heads")?,
        index_head_dim: by_part_int(indexed, "index_head_dim")?,
        index_topk: by_part_int(indexed, "index_topk")?,
        candidate_source_layer: int(facts, "candidate_source_layer")?,
        candidate_topk_blocks: int(facts, "candidate_topk_blocks")?,
        candidate_block_size: int(facts, "candidate_block_size")?,
        hc_mult: int(facts, "hc_mult")?,
        hc_sinkhorn_iters: int(facts, "hc_sinkhorn_iters")?,
        hc_eps: float(facts, "hc_eps")?,
        engram_layer_ids: integers(facts, "engram_layer_ids")?,
        engram_vocab_size: opt_int(facts, "engram_vocab_size").unwrap_or(0),
        engram_num_embeddings: integers(facts, "engram_num_embeddings").unwrap_or_default(),
        engram_max_ngram_size: opt_int(facts, "engram_max_ngram_size").unwrap_or(0),
        engram_pad_id: u32::try_from(opt_int(facts, "engram_pad_id").unwrap_or(0))
            .map_err(|_| "the plan's Engram pad id does not fit u32".to_owned())?,
        engram_compressed_vocab_size: opt_int(facts, "engram_compressed_vocab_size").unwrap_or(0),
        engram_n_heads: opt_int(facts, "engram_n_heads").unwrap_or(0),
        engram_head_dim: opt_int(facts, "engram_head_dim").unwrap_or(0),
        dtype: text(facts, "dtype")?,
        expert_dtype: text(facts, "expert_dtype")?,
        // The block and the scale values are the plan's own statement of its linear format, not
        // facts the released manifest records: read them from the resolved format.
        weight_block: dense.map_or(32, |format| format.block()),
        weight_scale: dense.map_or_else(
            || "ue8m0".to_owned(),
            |format| format.scale_values().name().to_owned(),
        ),
        compress_rope_theta: by_part_float(compressed, "compress_rope_theta")?,
        compress_ratios,
        image_token_id: u32::try_from(opt_int(facts, "image_token_id").unwrap_or(0))
            .map_err(|_| "the plan's image token does not fit u32".to_owned())?,
        gate_temp: float(facts, "gate_temp")?,
        norm_topk_prob: boolean(facts, "norm_topk_prob", true),
        vision_n_layers: opt_int(facts, "vision_n_layers"),
        vision_dim: opt_int(facts, "vision_dim"),
        vision_n_heads: opt_int(facts, "vision_n_heads"),
        vision_inter_dim: opt_int(facts, "vision_inter_dim"),
        vision_patch_size: opt_int(facts, "vision_patch_size"),
        vision_downsample_ratio: opt_int(facts, "vision_downsample_ratio"),
        vision_max_n_token: opt_int(facts, "vision_max_n_token"),
        vision_min_pixels: opt_int(facts, "vision_min_pixels"),
        vision_max_wh_ratio: opt_float(facts, "vision_max_wh_ratio"),
        vision_rope_theta: opt_float(facts, "vision_rope_theta"),
    })
}
