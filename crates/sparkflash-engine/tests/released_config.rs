//! The released checkpoint carries two configuration documents, and they are not the same file.
//!
//! `config/deepseek_v41_flash.released.json` is the checkpoint's own `config.json` - the released HF
//! document, nested under `text_config`/`vision_config` - while
//! `config/deepseek_v41_flash.runtime.json` is the flat document the reference loads with
//! `ModelArgs(**json.load(...))`. These tests pin the released document's schema and values, prove it
//! derives the same configuration as the runtime copy, and prove that a key the loader does not know
//! is refused rather than dropped.

use std::collections::BTreeSet;

use serde_json::Value;
use sparkflash_engine::config::{ModelConfig, V41_FLASH_RUNTIME_CONFIG};

/// The released checkpoint's `config.json`, byte-for-byte (sha256 in `config/README.md`).
const RELEASED: &str =
    include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");

/// The released document's top-level keys, pinned. A key added by a later release fails the test
/// until the loader is taught it.
const TOP_LEVEL: &[&str] = &[
    "architectures",
    "bos_token_id",
    "dtype",
    "eos_token_id",
    "image_token_id",
    "model_type",
    "pad_token_id",
    "quantization_config",
    "text_config",
    "transformers_version",
    "vision_config",
];

/// The released `text_config` keys, pinned.
const TEXT_CONFIG: &[&str] = &[
    "attention_bias",
    "attention_dropout",
    "candidate_block_size",
    "candidate_source_layer_id",
    "candidate_topk_blocks",
    "compress_ratios",
    "compress_rope_theta",
    "dspark_block_size",
    "dspark_markov_rank",
    "dspark_n_routed_experts",
    "dspark_noise_token_id",
    "dspark_num_experts_per_tok",
    "dspark_target_layer_ids",
    "engram_compressed_vocab_size",
    "engram_head_dim",
    "engram_layer_ids",
    "engram_max_ngram_size",
    "engram_n_heads",
    "engram_num_embeddings",
    "engram_pad_token_id",
    "engram_vocab_size",
    "hc_eps",
    "hc_mult",
    "hc_sinkhorn_iters",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_source_layer_ids",
    "index_topk",
    "initializer_range",
    "kv_source_layer_ids",
    "max_position_embeddings",
    "model_type",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "o_groups",
    "o_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "routed_scaling_factor",
    "scoring_func",
    "sliding_window",
    "swiglu_limit",
    "tie_word_embeddings",
    "topk_method",
    "use_cache",
    "vocab_size",
];

/// The released `vision_config` keys, pinned.
const VISION_CONFIG: &[&str] = &[
    "downsample_ratio",
    "hidden_size",
    "intermediate_size",
    "max_image_tokens",
    "max_wh_ratio",
    "min_pixels",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "patch_size",
    "rope_theta",
];

fn document() -> Value {
    serde_json::from_str(RELEASED).expect("the released fixture is valid JSON")
}

fn keys(value: &Value) -> BTreeSet<String> {
    value
        .as_object()
        .expect("a JSON object")
        .keys()
        .cloned()
        .collect()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[test]
fn the_fixture_is_the_released_file() {
    use sha2::{Digest, Sha256};

    let digest: String = Sha256::digest(RELEASED.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(
        digest, "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
        "the fixture must be the released config.json, not a re-serialisation of it"
    );
}

#[test]
fn the_released_document_derives_the_runtime_configuration() {
    let derived = ModelConfig::from_released_json(RELEASED).expect("the released document maps");
    let runtime = ModelConfig::v41_flash().expect("the runtime copy parses");
    assert_eq!(derived, runtime, "the two documents describe one model");
}

#[test]
fn from_json_accepts_both_documents() {
    let runtime = ModelConfig::v41_flash().expect("the runtime copy parses");
    assert_eq!(
        ModelConfig::from_json(RELEASED).expect("the released document parses"),
        runtime
    );
    assert_eq!(
        ModelConfig::from_json(V41_FLASH_RUNTIME_CONFIG).expect("the runtime document parses"),
        runtime
    );
}

#[test]
fn the_released_schema_is_pinned() {
    let document = document();
    assert_eq!(keys(&document), set(TOP_LEVEL));
    assert_eq!(keys(&document["text_config"]), set(TEXT_CONFIG));
    assert_eq!(keys(&document["vision_config"]), set(VISION_CONFIG));

    let text = document["text_config"].as_object().expect("an object");
    // DeepSeek-V3-era configuration fields. V4.1 dropped group routing - the gate is an ungrouped
    // top-k over all 384 experts (model.py:808-823) - and every block is a MoE block (model.py:929),
    // so a released document must carry neither and the loader has no field for either.
    for absent in ["first_k_dense_replace", "n_group", "topk_group"] {
        assert!(
            !text.contains_key(absent),
            "the released config must not carry {absent}"
        );
    }
    assert_eq!(text["num_key_value_heads"], 1, "the MLA has one KV head");
    assert_eq!(
        document["quantization_config"]["weight_block_size"],
        serde_json::json!([32, 32])
    );
    assert_eq!(document["quantization_config"]["scale_fmt"], "ue8m0");
    assert_eq!(document["quantization_config"]["expert_dtype"], "fp4");
    assert_eq!(document["quantization_config"]["quant_method"], "fp8");
    assert_eq!(document["text_config"]["rope_scaling"]["rope_type"], "yarn");
}

#[test]
fn the_released_values_are_the_ones_the_engine_was_verified_against() {
    let config = ModelConfig::from_released_json(RELEASED).expect("the released document maps");
    assert_eq!(config.vocab_size, 129_280);
    assert_eq!(config.dim, 5_120);
    assert_eq!(config.moe_inter_dim, 2_304);
    assert_eq!(config.n_layers, 40);
    assert_eq!(config.n_mtp_layers, 3);
    assert_eq!(config.n_heads, 64);
    assert_eq!(config.head_dim, 512);
    assert_eq!(config.rope_head_dim, 64);
    assert_eq!(config.q_lora_rank, 1_280);
    assert_eq!(config.o_lora_rank, 1_024);
    assert_eq!(config.o_groups, 8);
    assert_eq!(config.n_routed_experts, 384);
    assert_eq!(config.n_activated_experts, 6);
    assert_eq!(config.n_shared_experts, 1);
    assert_eq!(config.route_scale, 1.5);
    assert_eq!(config.swiglu_limit, 10.0);
    assert_eq!(config.norm_eps, 1e-20);
    assert_eq!(config.rope_theta, 10_000.0);
    assert_eq!(config.rope_factor, 16.0);
    assert_eq!(config.window_size, 128);
    assert_eq!(config.kv_source_layers, [2, 8, 14, 20]);
    assert_eq!(config.index_topk, 512);
    assert_eq!(config.hc_mult, 4);
    assert_eq!(config.engram_layer_ids, [1, 14]);
    assert_eq!(config.dspark_target_layer_ids, [37, 38, 39]);
    assert_eq!(
        config.compress_ratios[0], 0,
        "layer 0 is uncompressed, not CED"
    );
    assert_eq!(config.dspark_n_activated_experts, 3);
    assert_eq!(config.engram_num_embeddings, [384_006_168, 384_016_682]);
    assert_eq!(
        config.dtype, "fp8",
        "the linear format, not the bfloat16 storage"
    );
    assert_eq!(config.expert_dtype, "fp4");
    assert_eq!(config.image_token_id, 129_264);
    assert_eq!(config.vision_n_layers, Some(32));
    assert_eq!(config.vision_dim, Some(1_024));
}

#[test]
fn a_new_released_key_is_refused_rather_than_dropped() {
    let mut document = document();
    document["text_config"]["n_group"] = serde_json::json!(8);
    let error = ModelConfig::from_released_json(&document.to_string())
        .expect_err("an unmapped key must be an error");
    assert!(
        error.to_string().contains("n_group"),
        "the error must name the key, got {error}"
    );
}

#[test]
fn a_released_document_missing_a_field_is_refused() {
    let mut document = document();
    document["text_config"]
        .as_object_mut()
        .expect("an object")
        .remove("hidden_size");
    let error = ModelConfig::from_released_json(&document.to_string())
        .expect_err("a missing field must be an error");
    assert!(
        error.to_string().contains("hidden_size"),
        "the error must name the field, got {error}"
    );
}

#[test]
fn the_runtime_document_carries_the_fields_the_struct_serialises() {
    // The shipped runtime file and the struct are two halves of one schema. A key the file carries
    // that the struct does not would be silently dropped by serde; a field the struct requires that
    // the file omits would fail to parse. Either is a drift, and this is where it fails.
    let file: std::collections::BTreeMap<String, Value> =
        serde_json::from_str(V41_FLASH_RUNTIME_CONFIG).expect("valid JSON");
    let config = ModelConfig::v41_flash().expect("the runtime copy parses");
    let mut serialised: std::collections::BTreeMap<String, Value> =
        serde_json::from_value(serde_json::to_value(&config).expect("serialisable"))
            .expect("a JSON object");
    // The reference's ModelArgs supplies these two defaults, so the shipped runtime file does not
    // spell them out.
    serialised.remove("gate_temp");
    serialised.remove("norm_topk_prob");
    assert_eq!(
        file.keys().collect::<Vec<_>>(),
        serialised.keys().collect::<Vec<_>>()
    );
}
