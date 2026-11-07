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

//! The zero-code model acceptance test for issue #17.
//!
//! A temporary `HF_HUB_CACHE` holds a model this build has never seen:
//! `models--acme--Custom-Flash-Test/snapshots/<rev>` with a hybrid GDN + GQA + MoE
//! `config.json`, a ChatML `tokenizer.json` and synthetic safetensors. No descriptor or pack
//! table is edited: `sparkflash convert --model` resolves it through the hub, dispatches from the
//! shared architecture, and writes `rank-0-of-1.bin`; `sparkflash serve --model` plans it,
//! resolves every part, advertises the checkpoint's own id on `/v1/models` and completes a
//! `/v1/chat/completions` request on the dynamic host reference backend.
//!
//! D89: the child's `HF_HUB_CACHE` is pinned to the fixture, so the test cannot read or write
//! the operator's cache.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sparkflash_weights::dtype::DType;
use sparkflash_weights::safetensors::write_checkpoint;
use sparkflash_weights::spec::TensorSpec;

const MODEL_ID: &str = "acme/Custom-Flash-Test";
const HIDDEN: usize = 2560;
const VOCAB: usize = 256;
const HEADS: usize = 1;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const Q_WIDTH: usize = HEADS * HEAD_DIM;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const KEY_HEADS: usize = 1;
const VALUE_HEADS: usize = 1;
const KEY_DIM: usize = 64;
const VALUE_DIM: usize = 64;
const GDN_WIDTH: usize = 2 * KEY_HEADS * KEY_DIM + VALUE_HEADS * VALUE_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * VALUE_DIM;
const EXPERTS: usize = 512;

fn bf16_bytes(seed: u32, elements: usize) -> Vec<u8> {
    // Small finite values, deterministic in the element index. A BF16 bit pattern is the top
    // half of an f32, little-endian.
    let mut out = Vec::with_capacity(elements * 2);
    for index in 0..elements {
        let raw = (index as f32 + seed as f32 * 0.25) % 17.0 - 8.0;
        let value = raw * 0.01;
        let bits = (value.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn tensor(name: &str, shape: &[usize], seed: u32) -> (TensorSpec, Vec<u8>) {
    tensor_dtype(name, shape, seed, DType::BF16)
}

fn tensor_dtype(name: &str, shape: &[usize], seed: u32, dtype: DType) -> (TensorSpec, Vec<u8>) {
    let elements: usize = shape.iter().product();
    let bytes = match dtype {
        DType::F4E2M1 => {
            // Packed E2M1: two nibbles a byte, the value pattern is irrelevant to the shape
            // check and the host path decodes whatever is stored.
            vec![0x21; elements.div_ceil(2)]
        }
        DType::U8 | DType::F8E8M0 => vec![0x21; elements],
        _ => bf16_bytes(seed, elements),
    };
    (
        TensorSpec {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        },
        bytes,
    )
}

fn custom_config() -> Value {
    json!({
        "model_type": "acme_custom_flash",
        "architectures": ["AcmeCustomFlashForCausalLM"],
        "hidden_size": HIDDEN,
        "intermediate_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
        "eos_token_id": [0, 1],
        "bos_token_id": 0,
        "image_token_id": 3,
        "vision_start_token_id": 4,
        "vision_end_token_id": 5,
        "layer_types": ["linear_attention", "full_attention", "full_attention"],
        "mlp_layer_types": ["sparse", "dense", "sparse"],
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": KEY_HEADS,
        "linear_num_value_heads": VALUE_HEADS,
        "linear_key_head_dim": KEY_DIM,
        "linear_value_head_dim": VALUE_DIM,
        "num_experts": EXPERTS,
        "num_experts_per_tok": 10,
        "norm_topk_prob": true,
        "moe_intermediate_size": 16,
        "n_shared_experts": 1,
        "shared_expert_intermediate_size": 8,
        "routed_scaling_factor": 1.0
    })
}

fn tokenizer() -> Value {
    let mut vocab = BTreeMap::new();
    let specials = [
        ("<|endoftext|>", 0u64),
        ("<|im_end|>", 1),
        ("<|im_start|>", 2),
        ("<|image_pad|>", 3),
        ("<|vision_start|>", 4),
        ("<|vision_end|>", 5),
        ("hello", 6),
        ("world", 7),
        ("user", 8),
        ("assistant", 9),
        ("system", 10),
        ("You", 11),
        ("are", 12),
        ("a", 13),
        ("helpful", 14),
        ("assistant.", 15),
    ];
    let mut added = Vec::new();
    for (token, id) in specials {
        vocab.insert(token.to_owned(), json!(id));
        if token.starts_with('<') {
            added.push(json!({
                "id": id,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            }));
        }
    }
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<|endoftext|>",
        },
    })
}

fn write_fixture(cache: &Path) -> PathBuf {
    write_fixture_with(cache, |_, _| {})
}

fn write_fixture_with(
    cache: &Path,
    customize: impl FnOnce(&mut Value, &mut Vec<(TensorSpec, Vec<u8>)>),
) -> PathBuf {
    let repo = cache.join("models--acme--Custom-Flash-Test");
    let snapshot = repo.join("snapshots").join("rev1");
    std::fs::create_dir_all(&snapshot).expect("snapshot");
    std::fs::create_dir_all(repo.join("refs")).expect("refs");
    std::fs::write(
        repo.join("refs").join("main"),
        "rev1
",
    )
    .expect("ref");
    let mut config = custom_config();
    let mut tensors = fixture_tensors();
    customize(&mut config, &mut tensors);
    std::fs::write(
        snapshot.join("config.json"),
        serde_json::to_vec_pretty(&config).expect("config"),
    )
    .expect("config file");
    std::fs::write(
        snapshot.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer()).expect("tokenizer"),
    )
    .expect("tokenizer file");

    write_checkpoint(&snapshot, &tensors, 1).expect("synthetic safetensors");
    snapshot
}

fn fixture_tensors() -> Vec<(TensorSpec, Vec<u8>)> {
    let mut tensors = vec![
        tensor("model.embed_tokens.weight", &[VOCAB, HIDDEN], 1),
        tensor("model.norm.weight", &[HIDDEN], 2),
        tensor("lm_head.weight", &[VOCAB, HIDDEN], 3),
    ];
    // Layer 0: GDN + MoE.
    tensors.push(tensor(
        "model.layers.0.input_layernorm.weight",
        &[HIDDEN],
        10,
    ));
    tensors.push(tensor(
        "model.layers.0.post_attention_layernorm.weight",
        &[HIDDEN],
        11,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        &[GDN_WIDTH, HIDDEN],
        12,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.in_proj_z.weight",
        &[VALUE_WIDTH, HIDDEN],
        13,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.in_proj_b.weight",
        &[VALUE_HEADS, HIDDEN],
        14,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.in_proj_a.weight",
        &[VALUE_HEADS, HIDDEN],
        15,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.out_proj.weight",
        &[HIDDEN, VALUE_WIDTH],
        16,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.conv1d.weight",
        &[GDN_WIDTH, 1, 4],
        17,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.A_log",
        &[VALUE_HEADS],
        18,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.dt_bias",
        &[VALUE_HEADS],
        19,
    ));
    tensors.push(tensor(
        "model.layers.0.linear_attn.norm.weight",
        &[VALUE_DIM],
        20,
    ));
    push_moe(&mut tensors, 0, 30);
    // Layer 1: GQA + dense.
    push_gqa(&mut tensors, 1, 50);
    push_dense(&mut tensors, 1, 60);
    // Layer 2: GQA + MoE.
    push_gqa(&mut tensors, 2, 70);
    push_moe(&mut tensors, 2, 80);
    tensors
}

fn push_gqa(tensors: &mut Vec<(TensorSpec, Vec<u8>)>, layer: usize, seed: u32) {
    let prefix = format!("model.layers.{layer}");
    tensors.push(tensor(
        &format!("{prefix}.input_layernorm.weight"),
        &[HIDDEN],
        seed,
    ));
    tensors.push(tensor(
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[Q_WIDTH, HIDDEN],
        seed + 1,
    ));
    tensors.push(tensor(
        &format!("{prefix}.self_attn.k_proj.weight"),
        &[KV_WIDTH, HIDDEN],
        seed + 2,
    ));
    tensors.push(tensor(
        &format!("{prefix}.self_attn.v_proj.weight"),
        &[KV_WIDTH, HIDDEN],
        seed + 3,
    ));
    tensors.push(tensor(
        &format!("{prefix}.self_attn.o_proj.weight"),
        &[HIDDEN, Q_WIDTH],
        seed + 4,
    ));
    tensors.push(tensor(
        &format!("{prefix}.post_attention_layernorm.weight"),
        &[HIDDEN],
        seed + 5,
    ));
}

fn push_dense(tensors: &mut Vec<(TensorSpec, Vec<u8>)>, layer: usize, seed: u32) {
    let prefix = format!("model.layers.{layer}");
    tensors.push(tensor(
        &format!("{prefix}.mlp.gate_proj.weight"),
        &[16, HIDDEN],
        seed,
    ));
    tensors.push(tensor(
        &format!("{prefix}.mlp.up_proj.weight"),
        &[16, HIDDEN],
        seed + 1,
    ));
    tensors.push(tensor(
        &format!("{prefix}.mlp.down_proj.weight"),
        &[HIDDEN, 16],
        seed + 2,
    ));
}

fn push_moe(tensors: &mut Vec<(TensorSpec, Vec<u8>)>, layer: usize, seed: u32) {
    let prefix = format!("model.layers.{layer}");
    tensors.push(tensor(
        &format!("{prefix}.mlp.gate.weight"),
        &[EXPERTS, HIDDEN],
        seed,
    ));
    for expert in 0..EXPERTS as u32 {
        let base = format!("{prefix}.mlp.experts.{expert}");
        tensors.push(tensor(
            &format!("{base}.gate_proj.weight"),
            &[16, HIDDEN],
            seed + 10 + expert * 3,
        ));
        tensors.push(tensor(
            &format!("{base}.up_proj.weight"),
            &[16, HIDDEN],
            seed + 11 + expert * 3,
        ));
        tensors.push(tensor(
            &format!("{base}.down_proj.weight"),
            &[HIDDEN, 16],
            seed + 12 + expert * 3,
        ));
    }
    tensors.push(tensor(
        &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
        &[8, HIDDEN],
        seed + 40,
    ));
    tensors.push(tensor(
        &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
        &[8, HIDDEN],
        seed + 41,
    ));
    tensors.push(tensor(
        &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
        &[HIDDEN, 8],
        seed + 42,
    ));
    tensors.push(tensor(
        &format!("{prefix}.mlp.shared_expert_gate.weight"),
        &[1, HIDDEN],
        seed + 43,
    ));
}

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_sparkflash"))
}

fn cache_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    write_fixture(dir.path());
    dir
}

#[test]
fn a_cached_unknown_hybrid_converts_and_serves_without_a_descriptor_entry() {
    let cache = cache_fixture();
    let output = tempfile::tempdir().expect("output");
    let convert = binary()
        .arg("convert")
        .arg("--model")
        .arg(MODEL_ID)
        .arg("--output-dir")
        .arg(output.path())
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("convert runs");
    assert!(
        convert.status.success(),
        "convert failed: {}",
        String::from_utf8_lossy(&convert.stderr)
    );
    let rank_file = output.path().join("rank-0-of-1.bin");
    assert!(
        rank_file.is_file(),
        "the generic converter wrote no rank file: {}",
        String::from_utf8_lossy(&convert.stdout)
    );
    let inspect = binary()
        .arg("convert")
        .arg("--inspect")
        .arg(&rank_file)
        .arg("--verify-weights")
        .output()
        .expect("inspect runs");
    assert!(
        inspect.status.success(),
        "inspect failed: {}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let inspected = String::from_utf8_lossy(&inspect.stdout);
    assert!(
        inspected.contains("dynamic rank 0/1") && inspected.contains("weights_verified=true"),
        "the dynamic rank file inspects: {inspected}"
    );

    let mut server = Server::start(cache.path());
    let models = server.get("/v1/models");
    assert_eq!(models.status, 200, "{}", models.body);
    let listed: Value = serde_json::from_str(&models.body).expect("models JSON");
    assert_eq!(listed["data"][0]["id"], MODEL_ID);

    let chat = server.post(
        "/v1/chat/completions",
        &json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 4,
            "temperature": 0.0
        })
        .to_string(),
    );
    assert_eq!(chat.status, 200, "{}", chat.body);
    let body: Value = serde_json::from_str(&chat.body).expect("chat JSON");
    assert_eq!(body["model"], MODEL_ID);
    assert!(
        body["choices"]
            .as_array()
            .is_some_and(|choices| !choices.is_empty())
    );
    server.stop();
}

#[test]
fn an_unresolvable_checkpoint_names_every_missing_part_and_refuses() {
    // An MLA layer is a supported part in the catalogue but not in the dynamic host path; the
    // converter's part check is the exact list the issue asks a refusal to carry.
    let cache = cache_fixture();
    let repo = cache.path().join("models--acme--Custom-Flash-Test");
    let snapshot = repo.join("snapshots").join("rev1");
    let mut config = custom_config();
    // MLA at a hidden 5120 is a catalogue shape (DeepSeek's released geometry); the host
    // reference still cannot execute an MLA layer, so the capability check must use the host's
    // own part list rather than the CUDA catalogue.
    config["hidden_size"] = json!(5120);
    config["kv_lora_rank"] = json!(16);
    config["layer_types"] = json!(["full_attention", "full_attention", "full_attention"]);
    std::fs::write(
        snapshot.join("config.json"),
        serde_json::to_vec_pretty(&config).expect("config"),
    )
    .expect("config file");
    let convert = binary()
        .arg("convert")
        .arg("--model")
        .arg(MODEL_ID)
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("convert runs");
    assert_eq!(convert.status.code(), Some(2), "a refusal exits 2");
    let stderr = String::from_utf8_lossy(&convert.stderr);
    assert!(
        stderr.contains("parts the dynamic converter does not support")
            && stderr.contains("mla_attention"),
        "the refusal names the missing part: {stderr}"
    );

    // The serve path plans the same document before it opens a weight, and its refusal names
    // the part, its rank-local geometry and its precision: never a silent DeepSeek fall-through.
    let serve = binary()
        .arg("serve")
        .arg("--model")
        .arg(MODEL_ID)
        .arg("--bind")
        .arg("127.0.0.1:0")
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("serve runs");
    assert_eq!(serve.status.code(), Some(2), "a refusal exits 2");
    let stderr = String::from_utf8_lossy(&serve.stderr);
    assert!(
        stderr.contains("mla_attention") && stderr.contains("host reference"),
        "the refusal names the host-unsupported part: {stderr}"
    );
}

#[test]
fn a_malformed_config_is_a_parse_refusal_not_a_deepseek_fallthrough() {
    let cache = cache_fixture();
    let snapshot = cache
        .path()
        .join("models--acme--Custom-Flash-Test/snapshots/rev1");
    std::fs::write(snapshot.join("config.json"), b"{ not JSON").expect("malformed config");
    let serve = binary()
        .arg("serve")
        .arg("--checkpoint")
        .arg(&snapshot)
        .arg("--bind")
        .arg("127.0.0.1:0")
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("serve runs");
    assert_eq!(serve.status.code(), Some(2), "a refusal exits 2");
    let stderr = String::from_utf8_lossy(&serve.stderr);
    assert!(
        stderr.contains("config.json") && stderr.contains("not a usable architecture"),
        "the refusal names the file and parse error: {stderr}"
    );

    let convert = binary()
        .arg("convert")
        .arg("--checkpoint")
        .arg(&snapshot)
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("convert runs");
    assert_eq!(convert.status.code(), Some(2), "a refusal exits 2");
    let stderr = String::from_utf8_lossy(&convert.stderr);
    assert!(
        stderr.contains("config.json"),
        "the converter names the file: {stderr}"
    );
}

#[test]
fn a_quantized_u8_checkpoint_is_refused_before_serving() {
    let cache = tempfile::tempdir().expect("cache");
    let snapshot = write_fixture_with(cache.path(), |config, tensors| {
        config["quantization_config"] = json!({
            "quant_method": "modelopt",
            "quant_algo": "mixed_precision",
            "expert_weight": {"quant_algo": "nvfp4", "group_size": 16}
        });
        let name = "model.layers.0.mlp.experts.0.gate_proj.weight";
        let entry = tensors
            .iter_mut()
            .find(|(spec, _)| spec.name == name)
            .expect("expert gate");
        *entry = tensor_dtype(name, &[16, HIDDEN / 2], 7, DType::U8);
        tensors.push((
            TensorSpec {
                name: "model.layers.0.mlp.experts.0.gate_proj.weight_scale".to_owned(),
                dtype: DType::F8E8M0,
                shape: vec![16, HIDDEN / 32],
            },
            vec![0x7f; 16 * (HIDDEN / 32)],
        ));
    });
    let serve = binary()
        .arg("serve")
        .arg("--checkpoint")
        .arg(&snapshot)
        .arg("--bind")
        .arg("127.0.0.1:0")
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("serve runs");
    assert_eq!(serve.status.code(), Some(2), "a refusal exits 2");
    let stderr = String::from_utf8_lossy(&serve.stderr);
    assert!(
        stderr.contains("quantized") && stderr.contains("U8"),
        "the refusal names the quantization and storage: {stderr}"
    );
}

#[test]
fn a_yarn_rope_checkpoint_is_refused_at_boot() {
    let cache = tempfile::tempdir().expect("cache");
    let snapshot = write_fixture_with(cache.path(), |config, _| {
        config["rope_scaling"] = json!({
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128
        });
    });
    let serve = binary()
        .arg("serve")
        .arg("--checkpoint")
        .arg(&snapshot)
        .arg("--bind")
        .arg("127.0.0.1:0")
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("serve runs");
    assert_eq!(serve.status.code(), Some(2), "a refusal exits 2");
    let stderr = String::from_utf8_lossy(&serve.stderr);
    assert!(
        stderr.contains("rope.scaling") && stderr.contains("yarn"),
        "the boot refusal names the unsupported rotary rule: {stderr}"
    );
}

#[test]
fn a_single_file_checkpoint_converts_and_serves_without_an_index() {
    let cache = cache_fixture();
    let snapshot = cache
        .path()
        .join("models--acme--Custom-Flash-Test/snapshots/rev1");
    std::fs::remove_file(snapshot.join("model.safetensors.index.json")).expect("drop index");
    let output = tempfile::tempdir().expect("output");
    let convert = binary()
        .arg("convert")
        .arg("--model")
        .arg(MODEL_ID)
        .arg("--output-dir")
        .arg(output.path())
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("convert runs");
    assert!(
        convert.status.success(),
        "single-file convert failed: {}",
        String::from_utf8_lossy(&convert.stderr)
    );
    let mut server = Server::start(cache.path());
    let chat = server.post(
        "/v1/chat/completions",
        &json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "temperature": 0.0
        })
        .to_string(),
    );
    assert_eq!(chat.status, 200, "{}", chat.body);
    server.stop();
}

#[test]
fn the_tokenizer_flag_wins_and_an_unknown_tokenizer_is_refused() {
    let cache = cache_fixture();
    let snapshot = cache
        .path()
        .join("models--acme--Custom-Flash-Test/snapshots/rev1");
    let elsewhere = tempfile::tempdir().expect("tokenizer dir");
    let tokenizer = elsewhere.path().join("tokenizer.json");
    std::fs::copy(snapshot.join("tokenizer.json"), &tokenizer).expect("copy tokenizer");
    std::fs::remove_file(snapshot.join("tokenizer.json")).expect("remove checkpoint tokenizer");
    // The checkpoint has no tokenizer.json now; only --tokenizer can make this run serve.
    let mut server = Server::start_with_tokenizer(cache.path(), &tokenizer);
    let chat = server.post(
        "/v1/chat/completions",
        &json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "temperature": 0.0
        })
        .to_string(),
    );
    assert_eq!(chat.status, 200, "{}", chat.body);
    server.stop();

    std::fs::write(&tokenizer, br#"{"model":{"type":"WordLevel"}}"#).expect("unknown tokenizer");
    let serve = binary()
        .arg("serve")
        .arg("--model")
        .arg(MODEL_ID)
        .arg("--tokenizer")
        .arg(&tokenizer)
        .arg("--bind")
        .arg("127.0.0.1:0")
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("serve runs");
    assert_eq!(serve.status.code(), Some(2), "a refusal exits 2");
    let stderr = String::from_utf8_lossy(&serve.stderr);
    assert!(
        stderr.contains("tokenizer this path can name"),
        "the refusal names the tokenizer: {stderr}"
    );
}

#[test]
fn a_default_output_directory_is_outside_the_hub_cache() {
    let cache = cache_fixture();
    let cwd = tempfile::tempdir().expect("working directory");
    let convert = binary()
        .arg("convert")
        .arg("--model")
        .arg(MODEL_ID)
        .current_dir(cwd.path())
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("convert runs");
    assert!(
        convert.status.success(),
        "default-output convert failed: {}",
        String::from_utf8_lossy(&convert.stderr)
    );
    let output = cwd
        .path()
        .join("models--acme--Custom-Flash-Test-converted")
        .join("rank-0-of-1.bin");
    assert!(
        output.is_file(),
        "the default rank file is outside the cache: {}",
        output.display()
    );
    assert!(
        !cache
            .path()
            .join("models--acme--Custom-Flash-Test/snapshots/rev1/rank-0-of-1.bin")
            .exists(),
        "the conversion did not write into the cache snapshot"
    );
}

#[test]
fn the_generic_dynamic_writer_refuses_a_world_above_one() {
    let cache = cache_fixture();
    let output = tempfile::tempdir().expect("output");
    let convert = binary()
        .arg("convert")
        .arg("--model")
        .arg(MODEL_ID)
        .arg("--output-dir")
        .arg(output.path())
        .arg("--tp-world-size")
        .arg("4")
        .env("HF_HUB_CACHE", cache.path())
        .output()
        .expect("convert runs");
    assert_eq!(convert.status.code(), Some(2), "a refusal exits 2");
    let stderr = String::from_utf8_lossy(&convert.stderr);
    assert!(
        stderr.contains("TP1") && stderr.contains("whole copies"),
        "the refusal explains the writer's TP1 shape: {stderr}"
    );
}

struct Response {
    status: u16,
    body: String,
}

struct Server {
    child: Child,
    base: String,
    log: Arc<Mutex<Vec<String>>>,
}

impl Server {
    fn start(cache: &Path) -> Self {
        let mut child = binary();
        child
            .arg("serve")
            .arg("--model")
            .arg(MODEL_ID)
            .arg("--bind")
            .arg("127.0.0.1:0")
            .env("HF_HUB_CACHE", cache);
        Self::spawn(child)
    }

    fn start_with_tokenizer(cache: &Path, tokenizer: &Path) -> Self {
        let mut child = binary();
        child
            .arg("serve")
            .arg("--model")
            .arg(MODEL_ID)
            .arg("--tokenizer")
            .arg(tokenizer)
            .arg("--bind")
            .arg("127.0.0.1:0")
            .env("HF_HUB_CACHE", cache);
        Self::spawn(child)
    }

    fn spawn(mut command: Command) -> Self {
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("serve starts");
        let log = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = mpsc::channel();
        for pipe in [
            child
                .stdout
                .take()
                .map(|p| Box::new(p) as Box<dyn Read + Send>),
            child
                .stderr
                .take()
                .map(|p| Box::new(p) as Box<dyn Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let log = Arc::clone(&log);
            let tx = tx.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(pipe);
                for line in reader.lines().map_while(Result::ok) {
                    let _ = tx.send(line.clone());
                    log.lock().expect("log").push(line);
                }
            });
        }
        drop(tx);
        let deadline = Instant::now() + Duration::from_secs(120);
        let mut base = None;
        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(line) => {
                    if let Some(rest) = line.split("listening on ").nth(1) {
                        base = Some(rest.trim().to_owned());
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if child.try_wait().ok().flatten().is_some() {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        let Some(base) = base else {
            let _ = child.kill();
            panic!(
                "the dynamic server did not report its address; log:
{}",
                log.lock().expect("log").join(
                    "
"
                )
            );
        };
        Self { child, base, log }
    }

    fn get(&self, path: &str) -> Response {
        exchange(&self.base, "GET", path, None)
    }

    fn post(&self, path: &str, body: &str) -> Response {
        exchange(&self.base, "POST", path, Some(body))
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = &self.log;
    }
}

fn exchange(base: &str, method: &str, path: &str, body: Option<&str>) -> Response {
    let address = base.trim_start_matches("http://");
    let mut stream = TcpStream::connect(address).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .expect("timeout");
    let body = body.unwrap_or("");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("write");
    stream.flush().expect("flush");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").expect("HTTP head");
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .expect("status");
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        dechunk(body)
    } else {
        body.to_owned()
    };
    Response { status, body }
}

fn dechunk(body: &str) -> String {
    let mut rest = body;
    let mut out = String::new();
    while let Some((size, tail)) = rest.split_once("\r\n") {
        let Ok(size) = usize::from_str_radix(size.trim(), 16) else {
            break;
        };
        if size == 0 {
            break;
        }
        if tail.len() < size {
            break;
        }
        out.push_str(&tail[..size]);
        rest = tail[size..].strip_prefix("\r\n").unwrap_or("");
    }
    out
}
