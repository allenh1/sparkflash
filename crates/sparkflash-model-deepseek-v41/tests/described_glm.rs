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

//! GLM-5.3-Flash's description on the base's host forward, at tiny widths.
//!
//! The checkpoint is written from the layout (`write_described_checkpoint`) under GLM's own
//! stored names, and the base opens it through the description alone (`described::host_config`).
//! The forward runs GLM's stated wiring: KDA blocks, absorbed MLA with the pooled indexer and a
//! shared selection, the dense prefix and FP8-tile experts, each sublayer's own-mix collapse and
//! the mean head fold, all on the producer's BF16 type map.
//!
//! The pinned ids are the greedy tokens of every position on the type map. Before the GLM crate
//! was deleted its host reference (f32 streams) ran the same checkpoint: with the BF16 rounds and
//! the dynamic FP8 activation quantiser switched off in both, every logit of every position
//! agreed within 3e-6. With the quantiser on, an E4M3 code can flip on a 1e-7 input difference,
//! and on the type map 18 of the 20 positions keep the crate's argmax.

use std::path::PathBuf;

use sparkflash_model_deepseek_v41::checkpoint::write_described_checkpoint;
use sparkflash_model_deepseek_v41::described::host_config;
use sparkflash_model_deepseek_v41::forward::{InferenceEngine, SequenceState};

const CONFIG: &str = include_str!("fixtures/glm-tiny/config.json");

/// Twenty tokens: five complete pools of four, past the tiny budget of two pools, so the
/// ranking and the tail both decide what a query reads.
const TOKENS: [u32; 20] = [
    3, 17, 42, 5, 63, 0, 28, 9, 11, 200, 7, 33, 21, 60, 14, 90, 2, 150, 77, 8,
];

/// Every position's greedy token, under the fixture's current recipes (doc 08, "the fixture is made
/// decisive"): the head's first rows are drawn wider, so a handful of logits carry the row and the
/// argmax is decided by a margin rather than by a 512-way near-tie.
const PINNED: [usize; 20] = [
    75, 5, 4, 5, 2, 3, 2, 1, 1, 318, 1, 4, 3, 3, 4, 3, 2, 2, 6, 1,
];

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn engine() -> (Scratch, InferenceEngine) {
    engine_in_mla_form(false)
}

/// The engine over the fixture's checkpoint in one of the MLA's two host orders: `absorbed` (the
/// form a rank serves) or `expanded` (the producer's own, D125's parity mode).
fn engine_in_mla_form(expanded: bool) -> (Scratch, InferenceEngine) {
    engine_in_mla_form_over(CONFIG, expanded)
}

/// [engine_in_mla_form] over a stated configuration document.
fn engine_in_mla_form_over(config_text: &str, expanded: bool) -> (Scratch, InferenceEngine) {
    let directory = std::env::temp_dir().join(format!(
        "glm-tiny-{}-{expanded}-{}",
        std::process::id(),
        config_text.len()
    ));
    let scratch = Scratch(directory.clone());
    std::fs::create_dir_all(&directory).expect("a scratch directory");
    std::fs::write(directory.join("config.json"), config_text).expect("the config writes");
    let architecture =
        sparkflash_model::Architecture::from_json(config_text).expect("GLM parses");
    write_described_checkpoint(&directory, &architecture, 7, 2).expect("the checkpoint writes");
    let mut config = host_config(config_text).expect("the host runs GLM's description");
    if let Some(latent) = config.absorbed_latent.as_mut() {
        latent.expanded = expanded;
    }
    let source = sparkflash_model::layout::open_checkpoint(
        &directory,
        &directory.join("model.safetensors.index.json"),
    )
    .expect("the checkpoint opens");
    let engine = InferenceEngine::new(
        config,
        Box::new(source),
        sparkflash_weights::linalg::REFERENCE_TENSOR_LIMIT,
    )
    .expect("the engine opens");
    (scratch, engine)
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|a, b| row[*a].total_cmp(&row[*b]))
        .expect("a row")
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
fn glm_runs_from_its_description_whole_chunked_and_by_token_alike() {
    let (_scratch, engine) = engine();
    let config = engine.config().clone();
    let positions: Vec<usize> = (0..TOKENS.len()).collect();

    let mut state = SequenceState::new(&config);
    let whole = engine
        .prefill_with_position_logits(&mut state, &TOKENS, &positions)
        .expect("the whole prompt runs")
        .rows;
    assert!(whole.iter().flatten().all(|value| value.is_finite()));
    let ids: Vec<usize> = whole.iter().map(|row| argmax(row)).collect();
    assert_eq!(ids, PINNED);

    let mut by_token = SequenceState::new(&config);
    let tokens: Vec<Vec<f32>> = TOKENS
        .iter()
        .map(|token| {
            engine
                .prefill(&mut by_token, &[*token])
                .expect("a token runs")
                .logits
        })
        .collect();
    assert_eq!(bits(&tokens), bits(&whole), "token by token");

    let mut chunked = SequenceState::new(&config);
    // The split falls inside a pool, so the second chunk completes a pool the first began.
    let mut rows = engine
        .prefill_with_position_logits(&mut chunked, &TOKENS[..7], &positions[..7])
        .expect("the first chunk runs")
        .rows;
    rows.extend(
        engine
            .prefill_with_position_logits(&mut chunked, &TOKENS[7..], &positions[..13])
            .expect("the second chunk runs")
            .rows,
    );
    assert_eq!(bits(&rows), bits(&whole), "two chunks");
}

/// The description's wiring reaches the value table: the own-mix collapse, the mean fold, the
/// BF16 stream, the activation tiles, the pool and its tail, the absorbed latent and the key
/// norm's epsilon.
/// D125: the absorbed MLA is the producer's expansion evaluated in another order, so the two
/// host forms agree on the same inputs to a rounding's distance. The bar (1e-2 median) is the one
/// the decision set before its run; the same-input layer is the first block in the absorbed form,
/// whose input the two runs share.
#[test]
fn the_expanded_mla_form_meets_the_absorbed_form_as_a_reorder() {
    // The first block in the absorbed form: its input is the same in both runs, so the two
    // outputs are a same-input comparison.
    let layer = 3;
    let mut attention = Vec::new();
    // The BF16 declaration: the producer capture that measured the absorbed form's 3.6e-3 to
    // 3.9e-3 from the producer ran on this checkpoint, so D125's 1e-2 median is held here. The
    // release's dynamic scheme is measured separately (its quantiser amplifies a reorder).
    for expanded in [false, true] {
        let (scratch, engine) = engine_in_mla_form_over(BF16_CONFIG, expanded);
        let config = engine.config().clone();
        let mut state = SequenceState::new(&config);
        let observed = engine
            .prefill_observed(&mut state, &TOKENS)
            .expect("the fixture runs");
        let row = observed
            .boundary
            .rows()
            .iter()
            .find(|row| row.layer() == layer)
            .expect("the first absorbed block's boundary row");
        let tensor = row
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == sparkflash_model_deepseek_v41::forward::boundary::ATTENTION_OUT)
            .expect("the block's attention output");
        attention.push(tensor.data().as_f32().expect("f32 rows").to_vec());
        drop(scratch);
    }
    let (absorbed, expanded) = (&attention[0], &attention[1]);
    assert_eq!(absorbed.len(), expanded.len());
    let width = engine_width(absorbed);
    let mut relative: Vec<f64> = (0..absorbed.len() / width)
        .map(|row| {
            let (a, e) = (
                &absorbed[row * width..][..width],
                &expanded[row * width..][..width],
            );
            let difference: f64 = a
                .iter()
                .zip(e)
                .map(|(a, e)| (f64::from(*a) - f64::from(*e)).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm: f64 = e
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt();
            difference / norm
        })
        .collect();
    relative.sort_by(f64::total_cmp);
    let median = relative[relative.len() / 2];
    assert!(
        median <= 1e-2,
        "the absorbed form is {median:.3e} from the producer's order on the same inputs; D125's bar is 1e-2 median"
    );
}

/// The fixture's own document with a BF16 declaration: the stored format the producer capture's
/// same-input comparison was made on. Its weights are BF16, so neither side quantises an
/// activation and the only difference between the two forms is the expansion's two BF16
/// roundings.
const BF16_CONFIG: &str = r#"{
  "architectures": [
    "Glm5NextForConditionalGeneration"
  ],
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "eos_token_id": [
      1
    ],
    "first_k_dense_replace": 3,
    "hc_eps": 1e-06,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 0,
    "hidden_act": "silu",
    "hidden_size": 256,
    "index_head_dim": 64,
    "index_kpool": 4,
    "index_kpool_always_select_tail": true,
    "index_kpool_compress": true,
    "index_n_heads": 2,
    "index_topk": 8,
    "index_share_for_mtp_iteration": true,
    "indexer_rope_interleave": true,
    "indexer_types": [
      "full",
      "full",
      "full",
      "full",
      "full",
      "full",
      "full",
      "shared"
    ],
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "kv_lora_rank": 128,
    "layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "deepseek_sparse_attention",
      "linear_attention",
      "linear_attention",
      "deepseek_sparse_attention",
      "deepseek_sparse_attention"
    ],
    "linear_attn_config": {
      "num_heads": 4,
      "gate_lower_bound": -5.0,
      "head_dim": 64,
      "short_conv_kernel_size": 4,
      "kda_layers": [
        0,
        1,
        2,
        4,
        5
      ],
      "full_attn_layers": [
        3,
        6,
        7
      ]
    },
    "max_position_embeddings": 4096,
    "mhc": true,
    "mla_use_nope": true,
    "mlp_layer_types": [
      "dense",
      "dense",
      "dense",
      "sparse",
      "sparse",
      "sparse",
      "sparse",
      "sparse"
    ],
    "model_type": "glm5_next_text",
    "moe_intermediate_size": 128,
    "moe_router_dtype": "float32",
    "n_group": 1,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "norm_topk_prob": true,
    "num_attention_heads": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 8,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 1,
    "output_router_logits": false,
    "pad_token_id": 0,
    "q_lora_rank": 128,
    "qk_head_dim": 64,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 0,
    "rms_norm_eps": 1e-05,
    "routed_scaling_factor": 2.5,
    "router_aux_loss_coef": 0.001,
    "scoring_func": "sigmoid",
    "swiglu_limit": 10.0,
    "tie_word_embeddings": false,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "use_cache": true,
    "v_head_dim": 64,
    "vocab_size": 512
  },
  "tie_word_embeddings": false,
  "model_type": "glm5_next",
  "transformers_version": "5.16.0",
  "quantization_config": {
    "quant_method": "bf16",
    "activation_scheme": "dynamic",
    "weight_block_size": [
      32,
      32
    ]
  },
  "image_token_id": 500,
  "video_token_id": 501,
  "image_start_token_id": 502,
  "image_end_token_id": 503,
  "video_start_token_id": 504,
  "video_end_token_id": 505
}"#;

/// One row's width from a block's attention output: the description's `dim`.
fn engine_width(values: &[f32]) -> usize {
    let config = host_config(CONFIG).expect("GLM's description");
    assert_eq!(values.len() % config.dim, 0, "whole rows");
    config.dim
}

#[test]
fn glm_states_its_wiring() {
    let config = host_config(CONFIG).expect("the host runs GLM's description");
    let wiring = config.stated();
    assert!(wiring.own_mix && wiring.mean_fold && wiring.bf16_stream);
    assert_eq!((wiring.activation_tile, wiring.index_pool), (128, 4));
    assert!(wiring.index_tail);
    assert!(config.absorbed_latent.is_some());
    assert_eq!(config.index_key_norm_eps, 1e-6);
}
