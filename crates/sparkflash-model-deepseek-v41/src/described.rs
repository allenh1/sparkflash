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

/// The wiring facts a layout states that are not part sizes: how a part computes (its form, the
/// type a value is held in, the fold at the head). An arm that runs none of a stated value must
/// refuse the description by that name, never run its own arithmetic in the model's place.
const WIRING_FLAGS: [&str; 2] = [
    "language.indexer.pool_compress",
    "language.draft.indexer_shared",
];

fn wiring_value(value: &FactValue) -> String {
    match value {
        FactValue::Text(text) => text.clone(),
        FactValue::Bool(flag) => flag.to_string(),
        other => format!("{other:?}"),
    }
}

/// The wiring the base runs today, on the host and on the device: the single-latent MLA, each
/// sublayer's collapse through the previous sublayer's mix and the head fold through the last
/// pre-mix, the shared f32 KDA body, f32 norm and router sums.
pub const BASE_RUNS: [(&str, &str); 9] = [
    ("language.mla_attention.form", "latent"),
    (
        "language.manifold_hyper_connection.collapse",
        "previous_mix",
    ),
    (
        "language.manifold_hyper_connection.head_fold",
        "last_pre_mix",
    ),
    ("language.kda_scan.decay", "f32"),
    ("language.kda_scan.recurrence", "f32"),
    ("language.kda_scan.state", "f32"),
    ("language.kda_scan.output_norm", "f32"),
    ("language.rms_norm.accumulate", "f32"),
    ("language.moe.router_dtype", "float32"),
];

/// The wiring the host's whole-model forward runs: the base's, and the described forms it has
/// built beside them ([crate::forward]'s `stated` and `absorbed` modules). Each sublayer's
/// own-mix collapse, the mean head fold, the BF16 stream (every module output, norm and stream
/// update in BF16, f32 where the producer casts), f32 mix inputs and coefficients, the absorbed
/// MLA with its f32 softmax and query-typed probabilities, and the pooled indexer with f32
/// scores and pool softmax. `draft.indexer_shared` is the draft's selection; the host drafts
/// only through DSpark target layers, and a stack without them drafts nothing here.
pub const HOST_RUNS: [(&str, &str); 21] = [
    BASE_RUNS[0],
    BASE_RUNS[1],
    BASE_RUNS[2],
    BASE_RUNS[3],
    BASE_RUNS[4],
    BASE_RUNS[5],
    BASE_RUNS[6],
    BASE_RUNS[7],
    BASE_RUNS[8],
    ("language.mla_attention.form", "absorbed"),
    ("language.mla_attention.probabilities", "query"),
    ("language.mla_attention.softmax", "f32"),
    ("language.manifold_hyper_connection.collapse", "own_mix"),
    ("language.manifold_hyper_connection.head_fold", "mean"),
    ("language.manifold_hyper_connection.coefficients", "f32"),
    ("language.manifold_hyper_connection.mix_input", "f32"),
    (
        "language.manifold_hyper_connection.stream_update",
        "bfloat16",
    ),
    ("language.sparse_indexer.scores", "f32"),
    ("language.sparse_indexer.pool_softmax", "f32"),
    ("language.indexer.pool_compress", "true"),
    ("language.draft.indexer_shared", "true"),
];

/// Refuse every stated wiring fact whose value is not in `runs` (`(fact, value)` pairs): a
/// `language.<part>.<name>` fact, and the pooled indexer's and the shared draft selection's flags
/// when they are on. An arm passes what it computes; a fact it has not built is named, with the
/// value the model states.
///
/// # Errors
/// The facts the arm does not run, joined into one refusal.
pub fn refuse_unrun_wiring(
    facts: &BTreeMap<String, FactValue>,
    arm: &str,
    runs: &[(&str, &str)],
) -> Result<(), ConfigError> {
    let mut unrun = Vec::new();
    for (name, value) in facts {
        let wiring = WIRING_FLAGS.contains(&name.as_str())
            || name
                .strip_prefix("language.")
                .and_then(|rest| rest.split_once('.'))
                .is_some_and(|(part, _)| sparkflash_model::Part::parse(part).is_some());
        // A number is a part's parameter, read through the value table by the arm that runs
        // the part; only how a part computes is wiring.
        if !wiring
            || matches!(
                value,
                FactValue::Bool(false) | FactValue::Integer(_) | FactValue::Float(_)
            )
        {
            continue;
        }
        let value = wiring_value(value);
        if !runs
            .iter()
            .any(|(fact, ran)| *fact == name.as_str() && *ran == value)
        {
            unrun.push(format!("{name} = {value}"));
        }
    }
    if unrun.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{arm} does not run what the description states: {}",
            unrun.join(", ")
        ))
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
    let wiring = stated_wiring(facts, architecture)?;
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
        index_key_norm_eps: opt_float(facts, "language.sparse_indexer.key_norm_eps").unwrap_or(0.0),
        wiring: (!wiring.is_release()).then_some(wiring),
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

/// The wiring the facts state beside the release's ([crate::config::StatedWiring]): the mHC
/// schedule, the head fold, the stream's type map, the dynamic activation tile and the pooled
/// indexer. A value outside the ones the base knows is refused by name.
fn stated_wiring(
    facts: &BTreeMap<String, FactValue>,
    architecture: &Architecture,
) -> Result<crate::config::StatedWiring, ConfigError> {
    use sparkflash_model::architecture::WeightFormat;
    let fact = |name: &str| facts.get(name).and_then(FactValue::as_text);
    let choose = |name: &str, release: &str, other: &str| -> Result<bool, ConfigError> {
        match fact(name) {
            None => Ok(false),
            Some(value) if value == release => Ok(false),
            Some(value) if value == other => Ok(true),
            Some(value) => Err(format!(
                "the layout states {name} = {value}; the base knows {release} and {other}"
            )),
        }
    };
    let own_mix = choose(
        "language.manifold_hyper_connection.collapse",
        "previous_mix",
        "own_mix",
    )?;
    let mean_fold = choose(
        "language.manifold_hyper_connection.head_fold",
        "last_pre_mix",
        "mean",
    )?;
    let bf16_stream = match fact("language.manifold_hyper_connection.stream_update") {
        None => false,
        Some("bfloat16" | "bf16") => true,
        Some(other) => {
            return Err(format!(
                "the layout states the mHC stream update in {other}; the base runs a BF16 stream"
            ));
        }
    };
    // The dynamic recipe is part of the producer's type map: a description that states its
    // stream type runs its FP8 linears on the quantised input. The release's f32 reference path,
    // and the synthetic variants on it, keep their own activation handling.
    let activation_tile = match architecture.quantization.dense_format {
        WeightFormat::Fp8Tile {
            block_n,
            dynamic_activations: true,
            ..
        } if bf16_stream => block_n,
        _ => 0,
    };
    let pooled = matches!(
        facts.get("language.indexer.pool_compress"),
        Some(FactValue::Bool(true))
    );
    let mut pools = architecture
        .layers
        .iter()
        .filter_map(|layer| layer.attention.indexer.as_ref())
        .map(|indexer| (indexer.pool, indexer.always_select_tail));
    let (index_pool, index_tail) = match (pooled, pools.next()) {
        (true, Some(first)) => {
            if let Some(other) = pools.find(|other| *other != first) {
                return Err(format!(
                    "the pooled indexers disagree: a pool of {} (tail {}) and one of {} (tail {})",
                    first.0, first.1, other.0, other.1
                ));
            }
            if first.0 < 2 {
                return Err(format!(
                    "the layout states pooled indexer keys and a pool of {}",
                    first.0
                ));
            }
            first
        }
        (true, None) => {
            return Err(
                "the layout states pooled indexer keys and no layer runs an indexer".to_owned(),
            );
        }
        (false, _) => (0, false),
    };
    Ok(crate::config::StatedWiring {
        own_mix,
        mean_fold,
        bf16_stream,
        activation_tile,
        index_pool,
        index_tail,
    })
}

/// A described checkpoint's values from its `config.json` text alone: the architecture, its
/// facts, and the dense class it states. The host reference opens through this ([host_config]), and the device
/// forward reads the same facts through its plan (the plan's per-part formats agree with the
/// architecture's on every carried description; ops' engine_plan tests hold the two routes
/// equal).
///
/// # Errors
/// The config does not parse, or a fact the value table needs is missing.
pub fn from_config(text: &str) -> Result<ModelConfig, ConfigError> {
    let architecture = Architecture::from_json(text).map_err(|error| error.to_string())?;
    let facts = architecture.model_facts();
    from_facts(
        &facts,
        &architecture,
        architecture
            .stored_format(sparkflash_model::Part::DenseMlp)
            .map(sparkflash_model::architecture::WeightFormat::stored_format),
    )
}

/// A checkpoint's `config.json` text as the base reads it, and whether it was described: the
/// release's own spelling as it always parsed ([ModelConfig::from_json]), else another
/// architecture's spelling through its description ([host_config]).
///
/// # Errors
/// Neither reading opens the text; both refusals are named.
pub fn checkpoint_config(text: &str) -> Result<(ModelConfig, bool), ConfigError> {
    match ModelConfig::from_json(text) {
        Ok(config) => Ok((config, false)),
        Err(typed) => host_config(text)
            .map(|config| (config, true))
            .map_err(|described| {
                format!(
                    "as the release's spelling, {typed}; as a described architecture, {described}"
                )
            }),
    }
}

/// The host forward's open: [from_config], refused first for every wiring fact the host does not
/// run ([refuse_unrun_wiring] at [HOST_RUNS]).
///
/// # Errors
/// As [from_config], and the wiring the host has not built, by name.
pub fn host_config(text: &str) -> Result<ModelConfig, ConfigError> {
    let architecture = Architecture::from_json(text).map_err(|error| error.to_string())?;
    refuse_unrun_wiring(&architecture.model_facts(), "the host forward", &HOST_RUNS)?;
    from_config(text)
}
