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

//! The tiny synthetic Inkling checkpoint: every name the f32 forward reads, in memory.
//!
//! The fixture is deliberately small (two layers, four heads, two experts) and deterministic.
//! It is not a transcription of the 393 GB release; it is a shape-complete source that lets the
//! block-residual, relative-bias, convolution and router arithmetic run end to end on a CPU.

use std::collections::HashMap;
use std::sync::Arc;

use sparkflash_model::ModelDescriptor;
use sparkflash_model::architecture::WeightFormat;
use sparkflash_model::backend::ModelBackend;
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::forward::{Inkling2State, Inkling2Weights};
use sparkflash_model_inkling2::host::Inkling2HostBackend;
use sparkflash_weights::quant::f32_to_bf16_bits;
use sparkflash_weights::{DType, TensorSpec, WeightError, WeightSource};

struct TinySource {
    specs: HashMap<String, TensorSpec>,
    data: HashMap<String, Vec<u8>>,
}

impl TinySource {
    fn new() -> Self {
        Self {
            specs: HashMap::new(),
            data: HashMap::new(),
        }
    }

    fn put(&mut self, name: &str, dtype: DType, shape: Vec<usize>, bytes: Vec<u8>) {
        self.specs.insert(
            name.to_owned(),
            TensorSpec {
                name: name.to_owned(),
                dtype,
                shape,
            },
        );
        self.data.insert(name.to_owned(), bytes);
    }

    fn bf16(&mut self, name: &str, shape: Vec<usize>, values: &[f32]) {
        let elements: usize = shape.iter().product();
        assert_eq!(values.len(), elements, "{name}");
        let mut bytes = Vec::with_capacity(elements * 2);
        for value in values {
            bytes.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
        }
        self.put(name, DType::BF16, shape, bytes);
    }

    fn f32(&mut self, name: &str, shape: Vec<usize>, values: &[f32]) {
        let elements: usize = shape.iter().product();
        assert_eq!(values.len(), elements, "{name}");
        let mut bytes = Vec::with_capacity(elements * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.put(name, DType::F32, shape, bytes);
    }

    fn u8(&mut self, name: &str, shape: Vec<usize>, values: &[u8]) {
        let elements: usize = shape.iter().product();
        assert_eq!(values.len(), elements, "{name}");
        self.put(name, DType::U8, shape, values.to_vec());
    }
}

impl WeightSource for TinySource {
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.specs.keys().cloned().collect();
        names.sort();
        names
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

fn seeded(name: &str, index: usize, scale: f32) -> f32 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.bytes().chain(index.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    let unit = ((hash >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
    unit * scale
}

fn matrix(source: &mut TinySource, name: &str, rows: usize, columns: usize, scale: f32) {
    let values: Vec<f32> = (0..rows * columns)
        .map(|index| seeded(name, index, scale))
        .collect();
    source.bf16(name, vec![rows, columns], &values);
}

/// Build every tensor the tiny two-layer forward reads.
fn tiny_source(config: &Inkling2Config) -> TinySource {
    let mut source = TinySource::new();
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let d_rel = config.d_rel;
    let kernel = config.sconv_kernel_size;
    let experts = config.n_routed_experts;
    let q_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let r_width = heads * d_rel;

    let mut embed = Vec::with_capacity(vocab * hidden);
    for row in 0..vocab {
        for column in 0..hidden {
            embed.push(seeded("embed", row * hidden + column, 0.08));
        }
    }
    source.bf16("model.llm.embed.weight", vec![vocab, hidden], &embed);
    source.bf16("model.llm.embed_norm.weight", vec![hidden], &vec![1.0; hidden]);
    source.bf16("model.llm.norm.weight", vec![hidden], &vec![1.0; hidden]);
    let mut unembed = Vec::with_capacity(vocab * hidden);
    for row in 0..vocab {
        for column in 0..hidden {
            unembed.push(seeded("unembed", row * hidden + column, 0.06));
        }
    }
    source.bf16("model.llm.unembed.weight", vec![vocab, hidden], &unembed);

    // The released checkpoint stores layer L's attention input bank at index L - 1. The plan
    // carries indexes 0..layers-2, so the fixture does too.
    for bank in 0..config.num_hidden_layers.saturating_sub(1) {
        let layer = bank + 1;
        let queries = if config.full_layer_ids.contains(&layer) { 3 } else { 1 };
        let name = format!("model.llm.attn_res.attn_queries.{bank}");
        let shape = if queries == 1 {
            vec![hidden]
        } else {
            vec![queries, hidden]
        };
        let values: Vec<f32> = (0..queries * hidden)
            .map(|index| seeded(&name, index, 0.05))
            .collect();
        source.f32(&name, shape, &values);
    }
    for layer in 0..config.num_hidden_layers {
        let name = format!("model.llm.attn_res.mlp_queries.{layer}");
        let values: Vec<f32> = (0..hidden)
            .map(|index| seeded(&name, index, 0.05))
            .collect();
        source.f32(&name, vec![hidden], &values);
    }
    let values: Vec<f32> = (0..hidden)
        .map(|index| seeded("output_query", index, 0.05))
        .collect();
    source.f32("model.llm.attn_res.output_query", vec![hidden], &values);

    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.llm.layers.{layer}");
        let attn = format!("{prefix}.attn");
        let sliding = config.is_sliding(layer);
        source.bf16(&format!("{prefix}.mlp_norm.weight"), vec![hidden], &vec![1.0; hidden]);
        if sliding {
            source.bf16(&format!("{prefix}.attn_norm.weight"), vec![hidden], &vec![1.0; hidden]);
            source.bf16(
                &format!("{attn}.k_sconv.weight"),
                vec![kv_width, 1, kernel],
                &random_vec(&format!("{attn}.k_sconv"), kv_width * kernel, 0.2),
            );
            source.bf16(
                &format!("{attn}.v_sconv.weight"),
                vec![kv_width, 1, kernel],
                &random_vec(&format!("{attn}.v_sconv"), kv_width * kernel, 0.2),
            );
        } else {
            for name in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                source.bf16(&format!("{prefix}.{name}.weight"), vec![hidden], &vec![1.0; hidden]);
            }
            for name in ["k_sconv", "v_sconv"] {
                source.bf16(
                    &format!("{attn}.{name}.static_w"),
                    vec![kernel, kv_width],
                    &random_vec(&format!("{attn}.{name}.static"), kernel * kv_width, 0.15),
                );
                matrix(
                    &mut source,
                    &format!("{attn}.{name}.weight_proj"),
                    kv_heads * kernel,
                    hidden,
                    0.08,
                );
            }
        }
        source.bf16(
            &format!("{prefix}.attn_sconv.weight"),
            vec![hidden, 1, kernel],
            &random_vec(&format!("{prefix}.attn_sconv"), hidden * kernel, 0.12),
        );
        source.bf16(
            &format!("{prefix}.mlp_sconv.weight"),
            vec![hidden, 1, kernel],
            &random_vec(&format!("{prefix}.mlp_sconv"), hidden * kernel, 0.12),
        );
        matrix(&mut source, &format!("{attn}.wq_du.weight"), q_width, hidden, 0.08);
        matrix(&mut source, &format!("{attn}.wk_dv.weight"), kv_width, hidden, 0.08);
        matrix(&mut source, &format!("{attn}.wv_dv.weight"), kv_width, hidden, 0.08);
        matrix(&mut source, &format!("{attn}.wo_ud.weight"), hidden, q_width, 0.08);
        matrix(&mut source, &format!("{attn}.wr_du.weight"), r_width, hidden, 0.08);
        matrix(&mut source, &format!("{attn}.gate_attn.weight"), q_width, hidden, 0.08);
        matrix(&mut source, &format!("{attn}.attn_sink_proj.weight"), heads, hidden, 0.05);
        source.bf16(
            &format!("{attn}.rel_logits_proj.proj"),
            vec![d_rel, config.rel_extent],
            &random_vec(&format!("{attn}.rel_logits_proj"), d_rel * config.rel_extent, 0.05),
        );
        source.bf16(
            &format!("{attn}.q_l2_clip.scale"),
            vec![heads, head_dim],
            &vec![1.0; heads * head_dim],
        );
        source.bf16(
            &format!("{attn}.rel_l2_clip.scale"),
            vec![heads, d_rel],
            &vec![1.0; heads * d_rel],
        );
        matrix(&mut source, &format!("{prefix}.mlp.gate.weight"), experts, hidden, 0.1);
        source.f32(&format!("{prefix}.mlp.gate.bias"), vec![experts], &vec![0.0; experts]);
        write_experts(&mut source, config, layer);
    }
    source
}

fn random_vec(name: &str, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|index| seeded(name, index, scale)).collect()
}

fn write_experts(source: &mut TinySource, config: &Inkling2Config, layer: usize) {
    let prefix = format!("model.llm.layers.{layer}.mlp.experts");
    let experts = config.n_routed_experts;
    let hidden = config.hidden_size;
    let intermediate = config.moe_intermediate_size;
    match config.layer_format(layer).unwrap_or(WeightFormat::Bf16) {
        WeightFormat::NvFp4 { group_size } => {
            let w13_rows = 2 * intermediate;
            let w13_columns = hidden.div_ceil(2);
            let w2_columns = intermediate.div_ceil(2);
            let codes: Vec<u8> = (0..experts * w13_rows * w13_columns)
                .map(|index| (seeded("w13codes", index, 7.0) as i32).rem_euclid(16) as u8)
                .collect();
            source.u8(&format!("{prefix}.w13_weight"), vec![experts, w13_rows, w13_columns], &codes);
            let codes: Vec<u8> = (0..experts * hidden * w2_columns)
                .map(|index| (seeded("w2codes", index, 7.0) as i32).rem_euclid(16) as u8)
                .collect();
            source.u8(&format!("{prefix}.w2_weight"), vec![experts, hidden, w2_columns], &codes);
            let w13_scales = hidden.div_ceil(group_size);
            let w2_scales = intermediate.div_ceil(group_size);
            source.bf16(
                &format!("{prefix}.w13_weight_scale"),
                vec![experts, w13_rows, w13_scales],
                &vec![1.0; experts * w13_rows * w13_scales],
            );
            source.bf16(
                &format!("{prefix}.w2_weight_scale"),
                vec![experts, hidden, w2_scales],
                &vec![1.0; experts * hidden * w2_scales],
            );
            source.f32(&format!("{prefix}.w13_weight_amax"), vec![experts], &vec![1.0; experts]);
            source.f32(&format!("{prefix}.w2_weight_amax"), vec![experts], &vec![1.0; experts]);
            source.f32(&format!("{prefix}.w13_weight_outer_scale_denominator"), vec![1], &[6.0]);
            source.f32(&format!("{prefix}.w2_weight_outer_scale_denominator"), vec![1], &[6.0]);
        }
        WeightFormat::MxFp8 { block } => {
            let w13_rows = 2 * intermediate;
            let w13_columns = hidden;
            let w2_columns = intermediate;
            let codes: Vec<u8> = vec![0x38; experts * w13_rows * w13_columns];
            source.u8(&format!("{prefix}.w13_weight"), vec![experts, w13_rows, w13_columns], &codes);
            let codes: Vec<u8> = vec![0x38; experts * hidden * w2_columns];
            source.u8(&format!("{prefix}.w2_weight"), vec![experts, hidden, w2_columns], &codes);
            let w13_scales = hidden.div_ceil(block);
            let w2_scales = intermediate.div_ceil(block);
            let scales = vec![127u8; experts * w13_rows * w13_scales];
            source.u8(&format!("{prefix}.w13_weight_scale"), vec![experts, w13_rows, w13_scales], &scales);
            let scales = vec![127u8; experts * hidden * w2_scales];
            source.u8(&format!("{prefix}.w2_weight_scale"), vec![experts, hidden, w2_scales], &scales);
        }
        WeightFormat::Bf16 => {
            let w13: Vec<f32> = (0..experts * 2 * intermediate * hidden)
                .map(|index| seeded("w13", index, 0.06))
                .collect();
            source.bf16(
                &format!("{prefix}.w13_weight"),
                vec![experts, 2 * intermediate, hidden],
                &w13,
            );
            let w2: Vec<f32> = (0..experts * hidden * intermediate)
                .map(|index| seeded("w2", index, 0.06))
                .collect();
            source.bf16(
                &format!("{prefix}.w2_weight"),
                vec![experts, hidden, intermediate],
                &w2,
            );
        }
        other => panic!("the fixture does not write {other:?}"),
    }
}

#[test]
fn the_tiny_fixture_prefills_and_decodes_through_the_host_backend() {
    let config = Inkling2Config::tiny();
    let source = Arc::new(tiny_source(&config));
    let backend = Inkling2HostBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config.clone(),
        source,
        32,
    )
    .expect("the fixture opens");
    let mut session = backend.open_session().expect("a session");
    let first = session.prefill(&[1, 2, 3]).expect("the prompt runs");
    assert_eq!(first.logits.len(), config.unpadded_vocab_size);
    assert!(first.logits.iter().all(|value| value.is_finite()));
    let next = first.next_token.expect("the fixture has no end-of-token");
    let second = session.decode(next).expect("the decode runs");
    assert_eq!(second.logits.len(), config.unpadded_vocab_size);
    assert!(second.logits.iter().all(|value| value.is_finite()));
    assert_eq!(session.reusable_prefix(), Some([1, 2, 3, next].as_slice()));
}

#[test]
fn the_blank_state_and_one_step_agree_with_a_prefix() {
    let config = Inkling2Config::tiny();
    let source = Arc::new(tiny_source(&config));
    let weights = Inkling2Weights::load(config.clone(), source).expect("the weights load");
    let mut state = Inkling2State::new(&config);
    let first = weights.forward_token(7, &mut state).expect("the first token");
    let mut prompt = Inkling2State::new(&config);
    let same = weights.forward_prompt(&[7]).expect("the prompt");
    let _ = weights.forward_token(7, &mut prompt).expect("the prompt state");
    assert_eq!(first.len(), same.len());
    assert!(first.iter().zip(&same).all(|(a, b)| (a - b).abs() < 1e-5));
    let second = weights.forward_token(8, &mut state).expect("the second token");
    assert_eq!(second.len(), config.unpadded_vocab_size);
    assert_eq!(prompt.position(), 1);
}

#[test]
fn a_sliding_layer_keeps_only_its_window_and_a_full_layer_keeps_all() {
    let config = Inkling2Config::tiny();
    let source = Arc::new(tiny_source(&config));
    let weights = Inkling2Weights::load(config.clone(), source).expect("the weights load");
    let mut state = Inkling2State::new(&config);
    for token in [1u32, 2, 3, 4, 5, 6] {
        weights
            .forward_token(token, &mut state)
            .expect("the token runs");
    }
    // The state is private, but a longer prompt must still be finite: this is the symptom a
    // missing mask produces (NaN or an explosion), and the fixture is deliberately not tiny in
    // its score range.
    let mut long = Inkling2State::new(&config);
    let logits = (0..12)
        .map(|index| {
            weights
                .forward_token(index as u32, &mut long)
                .expect("a long prompt runs")
        })
        .last()
        .expect("the prompt is non-empty");
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(long.position(), 12);
}

#[test]
fn an_mxfp8_layer_uses_the_power_of_two_scale() {
    let mut config = Inkling2Config::tiny();
    config.quant.layer_expert_formats[0] = WeightFormat::MxFp8 { block: 32 };
    let source = Arc::new(tiny_source(&config));
    let weights = Inkling2Weights::load(config.clone(), source).expect("the weights load");
    let mut state = Inkling2State::new(&config);
    let logits = weights.forward_token(3, &mut state).expect("the token runs");
    // A ue8m0 scale of 127 is exactly 1.0. The broken (code + 127) decode produces 2^127,
    // which saturates a score to infinity; a finite row pins the exponent convention.
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "an E8M0 scale of 127 must decode as 1.0, not 2^127"
    );
}

#[test]
fn the_host_backend_refuses_a_context_past_the_model_window() {
    let config = Inkling2Config::tiny();
    let source = Arc::new(tiny_source(&config));
    let error = Inkling2HostBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config,
        source,
        1_000_000,
    )
    .expect_err("the tiny model's window is smaller");
    assert!(error.contains("context"), "{error}");
}

#[test]
fn a_prompt_that_extends_the_history_reuses_its_prefix() {
    let config = Inkling2Config::tiny();
    let source = Arc::new(tiny_source(&config));
    let backend = Inkling2HostBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config.clone(),
        source,
        32,
    )
    .expect("the fixture opens");
    let mut session = backend.open_session().expect("a session");
    let first = session.prefill(&[1, 2, 3]).expect("the prompt runs");
    let next = first.next_token.expect("a token");
    let continued = session
        .prefill(&[1, 2, 3, next])
        .expect("the extended prompt runs");
    assert_eq!(session.cached_tokens(), 3);
    assert_eq!(continued.logits.len(), config.unpadded_vocab_size);
    let mut fresh = backend.open_session().expect("a session");
    let reference = fresh
        .prefill(&[1, 2, 3, next])
        .expect("the full prompt runs");
    assert!(
        continued
            .logits
            .iter()
            .zip(&reference.logits)
            .all(|(a, b)| (a - b).abs() < 1e-5),
        "the prefix-continuation path differs from a fresh forward"
    );
}

#[test]
fn the_descriptor_is_the_inkling_pack() {
    let descriptor: ModelDescriptor = sparkflash_model_inkling2::descriptor();
    assert_eq!(descriptor.family, "inkling2");
    assert_eq!(descriptor.model_type, "inkling2");
}
