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

//! The model-wide facts a description carries, read from the typed architecture alone: the
//! engine plan publishes them, and a host reference opens a described checkpoint from them
//! without planning it.

use std::collections::BTreeMap;

use crate::layout::LayoutValue;
use crate::{Architecture, FactValue};

/// A layout fact as a plan fact. A list of names has no plan form of its own and is published
/// as its names joined by commas, in the layout's order.
fn layout_fact(value: &LayoutValue) -> FactValue {
    match value {
        LayoutValue::Bool(value) => FactValue::Bool(*value),
        LayoutValue::Integer(value) => FactValue::Integer(*value),
        LayoutValue::Float(value) => FactValue::Float(*value),
        LayoutValue::String(value) => FactValue::Text(value.clone()),
        LayoutValue::Integers(values) => FactValue::Integers(values.clone()),
        LayoutValue::Strings(values) => FactValue::Text(values.join(",")),
    }
}

impl Architecture {
    /// The model-wide facts the open manifest names, read from the typed architecture and the
    /// layout's config vars.
    ///
    /// The names are the baseline manifest's names, not a second vocabulary: a value the forward
    /// reads at open is carried here once, on the plan, so a backend never re-parses the checkpoint.
    #[must_use]
    pub fn model_facts(&self) -> BTreeMap<String, FactValue> {
        model_facts(self)
    }
}

fn model_facts(architecture: &Architecture) -> BTreeMap<String, FactValue> {
    let mut facts = BTreeMap::new();
    use crate::architecture::AttentionKind;
    // A grouped-query layer inside an MLA stack carries its own geometry (the `gqa_*` facts
    // below); the stack-wide attention facts are the MLA layers'. No other stack mixes the two.
    let mla_stack = architecture
        .layers
        .iter()
        .any(|layer| matches!(layer.attention.kind, AttentionKind::Mla { .. }));
    let foreign_gqa = |layer: &crate::architecture::Layer| {
        mla_stack && matches!(layer.attention.kind, AttentionKind::Gqa { .. })
    };
    // A recurrent layer (Gated DeltaNet or Kimi Delta Attention) in an MLA stack likewise
    // carries its own geometry (`gdn_*`, `kda_*`), so the stack-wide attention facts are the MLA
    // layers'. Whether such a block runs split or whole is the layout's statement
    // ([whole_recurrent_blocks]), not a fact of its kind.
    let foreign_gdn = |layer: &crate::architecture::Layer| {
        mla_stack && matches!(layer.attention.kind, AttentionKind::Gdn(_))
    };
    let foreign_kda = |layer: &crate::architecture::Layer| {
        mla_stack && matches!(layer.attention.kind, AttentionKind::Kda(_))
    };
    let first = architecture
        .layers
        .iter()
        .find(|layer| !foreign_gqa(layer) && !foreign_gdn(layer) && !foreign_kda(layer));
    let first_moe = architecture.first_moe();
    let first_hyper = architecture
        .layers
        .iter()
        .find_map(|layer| match &layer.residual {
            crate::architecture::Residual::Hyper(hyper) => Some(hyper),
            _ => None,
        });
    let first_mla = architecture
        .layers
        .iter()
        .find_map(|layer| match &layer.attention.kind {
            crate::architecture::AttentionKind::Mla {
                q_lora_rank,
                groups,
                ..
            } => Some((*q_lora_rank, *groups)),
            _ => None,
        });
    let mut put = |name: &str, value: FactValue| {
        facts.insert(name.to_owned(), value);
    };
    put(
        "vocab_size",
        FactValue::Integer(architecture.vocab_size as i64),
    );
    put("dim", FactValue::Integer(architecture.hidden_size as i64));
    put(
        "n_layers",
        FactValue::Integer(architecture.layers.len() as i64),
    );
    put("norm_eps", FactValue::Float(architecture.norm_epsilon));
    put(
        "image_token_id",
        architecture
            .tokens
            .image_token_id
            .map_or(FactValue::Null, |id| FactValue::Integer(i64::from(id))),
    );
    put(
        "dtype",
        FactValue::Text(
            architecture
                .quantization
                .method
                .clone()
                .unwrap_or_else(|| "bf16".to_owned()),
        ),
    );
    // A release that states one quantisation method and no expert format of its own (GLM's
    // `quant_method: fp8`) stores its experts in that method, as it stores its dense linears.
    put(
        "expert_dtype",
        FactValue::Text(
            architecture
                .quantization
                .expert
                .clone()
                .or_else(|| architecture.quantization.method.clone())
                .unwrap_or_else(|| "bf16".to_owned()),
        ),
    );
    if let Some(attention) = first.map(|layer| &layer.attention) {
        put("n_heads", FactValue::Integer(attention.heads as i64));
        put("head_dim", FactValue::Integer(attention.head_dim as i64));
        put(
            "rope_head_dim",
            FactValue::Integer(attention.rope.rotated_dims as i64),
        );
        put("rope_theta", FactValue::Float(attention.rope.theta));
        put(
            "window_size",
            attention
                .window
                .map_or(FactValue::Null, |window| FactValue::Integer(window as i64)),
        );
    }
    if let Some(scaling) = architecture
        .layers
        .iter()
        .find_map(|layer| layer.attention.rope.scaling.as_ref())
    {
        put(
            "original_seq_len",
            FactValue::Integer(scaling.original_seq_len as i64),
        );
        put("rope_factor", FactValue::Float(scaling.factor));
        put("beta_fast", FactValue::Float(scaling.beta_fast));
        put("beta_slow", FactValue::Float(scaling.beta_slow));
    }
    if let Some((q_lora, groups)) = first_mla {
        put(
            "q_lora_rank",
            q_lora.map_or(FactValue::Null, |rank| FactValue::Integer(rank as i64)),
        );
        if let Some(groups) = groups {
            put("o_groups", FactValue::Integer(groups.groups as i64));
            put("o_lora_rank", FactValue::Integer(groups.rank as i64));
        }
    }
    // The layers whose FFN is one dense gated MLP, and its width: a stack that mixes them with
    // MoE layers (`first_k_dense_replace`) runs them as their own part.
    let dense_layers: Vec<i64> = architecture
        .layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| matches!(layer.ffn, crate::architecture::Ffn::Dense(_)))
        .map(|(index, _)| index as i64)
        .collect();
    if let Some(dense) = architecture
        .layers
        .iter()
        .find_map(|layer| match &layer.ffn {
            crate::architecture::Ffn::Dense(dense) => Some(dense),
            crate::architecture::Ffn::Moe(_) => None,
        })
    {
        put("dense_ffn_layers", FactValue::Integers(dense_layers));
        put(
            "dense_inter_dim",
            FactValue::Integer(dense.intermediate as i64),
        );
    }
    // The grouped-query layers of an MLA stack, and their one geometry.
    let gqa_layers: Vec<&crate::architecture::Layer> = architecture
        .layers
        .iter()
        .filter(|layer| foreign_gqa(layer))
        .collect();
    if let Some(gqa) = gqa_layers.first() {
        let indices = gqa_layers.iter().map(|layer| layer.index as i64).collect();
        put("gqa_layers", FactValue::Integers(indices));
        put("gqa_heads", FactValue::Integer(gqa.attention.heads as i64));
        if let AttentionKind::Gqa { kv_heads, .. } = gqa.attention.kind {
            put("gqa_kv_heads", FactValue::Integer(kv_heads as i64));
        }
        put(
            "gqa_head_dim",
            FactValue::Integer(gqa.attention.head_dim as i64),
        );
        put(
            "gqa_rotary",
            FactValue::Integer(gqa.attention.rope.rotated_dims as i64),
        );
        put("gqa_rope_theta", FactValue::Float(gqa.attention.rope.theta));
    }
    let gdn_layers: Vec<&crate::architecture::Layer> = architecture
        .layers
        .iter()
        .filter(|layer| foreign_gdn(layer))
        .collect();
    if let Some(AttentionKind::Gdn(gdn)) = gdn_layers.first().map(|layer| &layer.attention.kind) {
        let indices = gdn_layers.iter().map(|layer| layer.index as i64).collect();
        put("gdn_layers", FactValue::Integers(indices));
        put("gdn_key_heads", FactValue::Integer(gdn.key_heads as i64));
        put(
            "gdn_value_heads",
            FactValue::Integer(gdn.value_heads as i64),
        );
        put("gdn_key_dim", FactValue::Integer(gdn.key_dim as i64));
        put("gdn_value_dim", FactValue::Integer(gdn.value_dim as i64));
        put("gdn_conv", FactValue::Integer(gdn.conv_kernel as i64));
    }
    let kda_layers: Vec<&crate::architecture::Layer> = architecture
        .layers
        .iter()
        .filter(|layer| foreign_kda(layer))
        .collect();
    if let Some(AttentionKind::Kda(kda)) = kda_layers.first().map(|layer| &layer.attention.kind) {
        let indices = kda_layers.iter().map(|layer| layer.index as i64).collect();
        put("kda_layers", FactValue::Integers(indices));
        put("kda_heads", FactValue::Integer(kda.heads as i64));
        put("kda_head_dim", FactValue::Integer(kda.head_dim as i64));
        put("kda_conv", FactValue::Integer(kda.conv_kernel as i64));
        put("kda_lower_bound", FactValue::Float(kda.lower_bound));
    }
    // The MLA latent's widths: the absorbed form's key/value latent and value head, which the
    // layout's `language.mla_attention.form` says it reads through `kv_b`.
    if let Some(AttentionKind::Mla {
        kv_lora_rank,
        v_head_dim,
        ..
    }) = first.map(|layer| &layer.attention.kind)
    {
        put("kv_lora_rank", FactValue::Integer(*kv_lora_rank as i64));
        put("v_head_dim", FactValue::Integer(*v_head_dim as i64));
    }
    // The wiring facts the layout states and no typed field holds, under the layout's path. A
    // stack whose layout does not state one carries no such fact.
    for (name, value) in &architecture.layout_facts {
        put(name, layout_fact(value));
    }
    if let Some(moe) = first_moe {
        put("moe_inter_dim", FactValue::Integer(moe.intermediate as i64));
        put(
            "n_routed_experts",
            FactValue::Integer(moe.routed_experts as i64),
        );
        put(
            "n_shared_experts",
            FactValue::Integer(moe.shared_experts as i64),
        );
        put(
            "n_activated_experts",
            FactValue::Integer(moe.active_experts as i64),
        );
        put(
            "score_func",
            FactValue::Text(
                match moe.score {
                    crate::architecture::Score::Softmax => "softmax",
                    crate::architecture::Score::Sigmoid => "sigmoid",
                    crate::architecture::Score::SqrtSoftPlus => "sqrtsoftplus",
                }
                .to_owned(),
            ),
        );
        put("route_scale", FactValue::Float(moe.routed_scaling));
        put("norm_topk_prob", FactValue::Bool(moe.norm_topk));
        put(
            "swiglu_limit",
            moe.clamp.map_or(FactValue::Null, FactValue::Float),
        );
    }
    if let Some(hyper) = first_hyper {
        put("hc_mult", FactValue::Integer(hyper.streams as i64));
        put(
            "hc_sinkhorn_iters",
            FactValue::Integer(hyper.sinkhorn_iters as i64),
        );
        put("hc_eps", FactValue::Float(hyper.epsilon));
    }
    let kv_sources: Vec<i64> = architecture
        .layers
        .iter()
        .filter(|layer| {
            layer
                .attention
                .compression
                .as_ref()
                .is_some_and(|compression| compression.owns_keys)
        })
        .map(|layer| layer.index as i64)
        .collect();
    let index_sources: Vec<i64> = architecture
        .layers
        .iter()
        .filter(|layer| layer.attention.indexer.is_some())
        .map(|layer| layer.index as i64)
        .collect();
    put("kv_source_layers", FactValue::Integers(kv_sources));
    put("index_source_layers", FactValue::Integers(index_sources));
    let pool = architecture
        .layers
        .iter()
        .find_map(|layer| layer.attention.indexer.as_ref()?.candidate_pool.as_ref());
    put(
        "candidate_source_layer",
        pool.map_or(FactValue::Integer(0), |pool| {
            FactValue::Integer(pool.source_layer as i64)
        }),
    );
    put(
        "candidate_topk_blocks",
        pool.map_or(FactValue::Integer(0), |pool| {
            FactValue::Integer(pool.blocks as i64)
        }),
    );
    put(
        "candidate_block_size",
        pool.map_or(FactValue::Integer(0), |pool| {
            FactValue::Integer(pool.block_size as i64)
        }),
    );
    let indexer = architecture
        .layers
        .iter()
        .find_map(|layer| layer.attention.indexer.as_ref());
    if let Some(indexer) = indexer {
        put("index_n_heads", FactValue::Integer(indexer.heads as i64));
        put(
            "index_head_dim",
            FactValue::Integer(indexer.head_dim as i64),
        );
        put("index_topk", FactValue::Integer(indexer.topk as i64));
    }
    let compressed = architecture.layers.iter().find_map(|layer| {
        layer
            .attention
            .compression
            .as_ref()
            .filter(|compression| compression.ratio > 0)
            .map(|_| layer.attention.rope.theta)
    });
    put(
        "compress_rope_theta",
        compressed.map_or(FactValue::Null, FactValue::Float),
    );
    let mut ratios = Vec::with_capacity(architecture.layers.len());
    for layer in &architecture.layers {
        ratios.push(
            layer
                .attention
                .compression
                .as_ref()
                .map_or(0, |compression| compression.ratio) as i64,
        );
    }
    if let Some(mtp) = &architecture.mtp {
        ratios.extend(std::iter::repeat_n(0, mtp.stages));
        put("n_mtp_layers", FactValue::Integer(mtp.stages as i64));
        put(
            "dspark_block_size",
            mtp.block_size
                .map_or(FactValue::Null, |size| FactValue::Integer(size as i64)),
        );
        put(
            "dspark_target_layer_ids",
            FactValue::Integers(
                mtp.target_layers
                    .iter()
                    .map(|layer| *layer as i64)
                    .collect(),
            ),
        );
        put(
            "dspark_n_routed_experts",
            mtp.routed_experts
                .map_or(FactValue::Null, |count| FactValue::Integer(count as i64)),
        );
        put(
            "dspark_n_activated_experts",
            mtp.active_experts
                .map_or(FactValue::Null, |count| FactValue::Integer(count as i64)),
        );
    }
    put("compress_ratios", FactValue::Integers(ratios));
    let engram_layers: Vec<i64> = architecture
        .layers
        .iter()
        .filter(|layer| {
            layer
                .sides
                .iter()
                .any(|side| matches!(side, crate::architecture::Side::Engram(_)))
        })
        .map(|layer| layer.index as i64)
        .collect();
    put("engram_layer_ids", FactValue::Integers(engram_layers));
    if let Some(engram) = architecture.layers.iter().find_map(|layer| {
        layer.sides.iter().find_map(|side| match side {
            crate::architecture::Side::Engram(engram) => Some(engram),
            crate::architecture::Side::Ple(_) => None,
        })
    }) {
        put(
            "engram_num_embeddings",
            FactValue::Integers(engram.tables.iter().map(|rows| *rows as i64).collect()),
        );
        put(
            "engram_max_ngram_size",
            FactValue::Integer(engram.max_ngram as i64),
        );
        put(
            "engram_pad_id",
            FactValue::Integer(i64::from(engram.pad_id)),
        );
        put(
            "engram_compressed_vocab_size",
            FactValue::Integer(engram.compressed_vocab as i64),
        );
        put("engram_n_heads", FactValue::Integer(engram.heads as i64));
        put(
            "engram_head_dim",
            FactValue::Integer(engram.head_dim as i64),
        );
        put(
            "engram_vocab_size",
            FactValue::Integer(engram.vocab_size as i64),
        );
    }
    put(
        "dspark_markov_rank",
        FactValue::Integer(
            architecture
                .layout_vars
                .get("markov_rank")
                .copied()
                .unwrap_or(0),
        ),
    );
    put(
        "dspark_noise_token_id",
        FactValue::Integer(
            architecture
                .layout_vars
                .get("noise_token")
                .copied()
                .unwrap_or(0),
        ),
    );
    // The layout states the producer's ModelArgs default (model.py:67-68); the release stores
    // no key for it, so the fallback is the one the reference constructor uses.
    put(
        "gate_temp",
        FactValue::Float(
            architecture
                .layout_vars
                .get("gate_temp")
                .copied()
                .unwrap_or(1) as f64,
        ),
    );
    if let Some(vision) = &architecture.vision {
        put("vision_n_layers", FactValue::Integer(vision.depth as i64));
        put("vision_dim", FactValue::Integer(vision.hidden_size as i64));
        put("vision_n_heads", FactValue::Integer(vision.heads as i64));
        put(
            "vision_inter_dim",
            FactValue::Integer(vision.intermediate as i64),
        );
        put(
            "vision_patch_size",
            FactValue::Integer(vision.patch_size as i64),
        );
        put(
            "vision_downsample_ratio",
            FactValue::Integer(vision.spatial_merge as i64),
        );
        put(
            "vision_max_n_token",
            FactValue::Integer(
                architecture
                    .layout_vars
                    .get("vision_max_tokens")
                    .copied()
                    .unwrap_or(0),
            ),
        );
        put(
            "vision_min_pixels",
            FactValue::Integer(vision.min_pixels as i64),
        );
        // The released document stores `null`; the plan says so rather than inventing a cap.
        put("vision_max_wh_ratio", FactValue::Null);
        put("vision_rope_theta", FactValue::Float(vision.rope_theta));
    }
    facts
}
