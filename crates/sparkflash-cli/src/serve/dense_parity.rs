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

//! The shared dense device arm's gate: a per-layer parity report against the host reference.
//!
//! Two synthetic BF16 checkpoints of families no pack names run through both backends over the
//! same prompt and decode steps, and every layer's residual (every row) and the logits are
//! compared. The first is a plain Llama-shaped decoder whose prompt spans two chunks on the wide
//! tile; the second gates its attention output, norms its queries and keys, rotates half of each
//! head, ties its head to the embedding and has an MLP width the wide tile does not cover (the
//! row GEMV a group at a time). The arm keeps its key/value cache in BF16 where the reference
//! keeps f32, so the bound is a tolerance, and the next token must agree at every step.
//!
//! Three routed checkpoints follow: a softmax router with a sigmoid-gated shared expert, a
//! sigmoid router after a dense first layer (picks kept as scored and scaled, an ungated shared
//! expert), and a sqrt-softplus router with no shared expert. Two block-scaled FP8 checkpoints
//! close it, as the Qwen3 FP8 releases store their projections and experts (E4M3 codes under one
//! BF16 or F32 inverse scale a 128 x 128 tile): a dense gated one and the first routed one.
//!
//! The arm's serving contract is checked here too, on the same fixtures: fused lanes answer every
//! step's logits to the bit as each conversation alone, and a kept lane continues its
//! conversation or goes back to the end of its prompt.
//!
//! The arm's tensor-parallel ranks are gated the same way, as an in-process group of two on this
//! one device: rank 0's layers and logits against the host reference, every other rank's to the
//! bit as rank 0's (each rank sums the same partials in rank order, so the residuals agree), and
//! the device legs' sums (one-shot and two-shot, met here through the host) to the bit as the
//! host leg's.
//!
//! Without the kernels or a CUDA device the test says so and passes: the gate runs where the arm
//! can.

use std::collections::BTreeMap;
use std::sync::{Arc, mpsc};

use serde_json::{Value, json};
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model::{Architecture, Capabilities, ModelBackend, ModelDescriptor, ModelSession};
use sparkflash_model_dense::DenseDeviceBackend;
use sparkflash_ops::gemv::CudaHandle;
use sparkflash_ops::parity::{ParityReport, Tolerance};
use sparkflash_ops::plan::OpKind;
use sparkflash_weights::dtype::DType;
use sparkflash_weights::linalg::EngineError;
use sparkflash_weights::spec::TensorSpec;
use sparkflash_weights::tp::{LocalBus, LocalRank, TpCollective, TpLayout, TpRuntime, Wire};
use sparkflash_weights::weights::{WeightError, WeightSource};

use super::dynamic_host::DynamicHostBackend;

/// The bound one residual or logit element meets: the BF16 cache's rounding of every key and
/// value, carried through two layers.
const TOLERANCE: Tolerance = Tolerance::new(2e-2, 2e-2);

struct Memory(BTreeMap<String, (TensorSpec, Vec<u8>)>);

/// `elements` deterministic words salted by `name`.
fn mixed(name: &str, elements: usize) -> impl Iterator<Item = u32> {
    let salt = name.bytes().fold(0x811c_9dc5u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    });
    (0..elements as u32).map(move |index| {
        (index ^ salt)
            .wrapping_mul(0x9e37_79b9)
            .rotate_left(13)
            .wrapping_mul(0x85eb_ca6b)
    })
}

impl Memory {
    fn insert(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: Vec<u8>) {
        let spec = TensorSpec {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        };
        self.0.insert(name.to_owned(), (spec, bytes));
    }

    /// A BF16 tensor of deterministic values in `[-scale, scale]` around `centre`.
    fn push(&mut self, name: &str, shape: &[usize], centre: f32, scale: f32) {
        let elements: usize = shape.iter().product();
        let bytes = mixed(name, elements)
            .flat_map(|mixed| {
                let unit = (mixed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
                let value = centre + unit * scale;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        self.insert(name, DType::BF16, shape, bytes);
    }

    /// A block-scaled E4M3 `[rows, columns]` weight as the FP8 releases store one: codes of
    /// magnitude under two and a `weight_scale_inv` grid a 128 x 128 tile around `scale`, BF16 or
    /// (`wide`) F32 words that hold BF16 values.
    fn push_fp8(&mut self, name: &str, [rows, columns]: [usize; 2], scale: f32, wide: bool) {
        // Exponent fields 0..=7 keep every code finite and under 1.875 in magnitude.
        let codes = mixed(name, rows * columns)
            .map(|mixed| ((mixed >> 24) & 0x80 | (mixed >> 8) & 0x3f) as u8)
            .collect();
        self.insert(name, DType::F8E4M3, &[rows, columns], codes);
        let grid = [rows.div_ceil(128), columns.div_ceil(128)];
        let scale_name = name.replace(".weight", ".weight_scale_inv");
        let scales = mixed(&scale_name, grid[0] * grid[1]).map(|mixed| {
            let unit = (mixed >> 8) as f32 / (1u32 << 24) as f32;
            ((scale * (0.5 + unit)).to_bits() >> 16) as u16
        });
        if wide {
            let words = scales.flat_map(|bits| (u32::from(bits) << 16).to_le_bytes());
            self.insert(&scale_name, DType::F32, &grid, words.collect());
        } else {
            let words = scales.flat_map(u16::to_le_bytes);
            self.insert(&scale_name, DType::BF16, &grid, words.collect());
        }
    }
}

impl WeightSource for Memory {
    fn names(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.0
            .get(name)
            .map(|(spec, _)| spec.clone())
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        self.0
            .get(name)
            .map(|(_, bytes)| bytes.clone())
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }
}

/// One synthetic checkpoint: its configuration and the shapes its tensors take.
struct Fixture {
    config: Value,
    gated: bool,
    tied: bool,
    /// Its projections and experts are block-scaled E4M3 (`weight_block_size` 128 x 128), each
    /// odd layer's scales F32; the embedding, head, norms and routers stay BF16.
    fp8: bool,
}

impl Fixture {
    fn value(&self, key: &str) -> usize {
        self.config[key].as_u64().expect("an integer key") as usize
    }

    fn source(&self) -> Memory {
        let hidden = self.value("hidden_size");
        let vocab = self.value("vocab_size");
        let heads = self.value("num_attention_heads");
        let kv_heads = self.value("num_key_value_heads");
        let head_dim = self.value("head_dim");
        let intermediate = self.value("intermediate_size");
        let weight = (hidden as f32).sqrt().recip();
        let mut source = Memory(BTreeMap::new());
        source.push("model.embed_tokens.weight", &[vocab, hidden], 0.0, 1.0);
        source.push("model.norm.weight", &[hidden], 1.0, 0.2);
        if !self.tied {
            source.push("lm_head.weight", &[vocab, hidden], 0.0, weight);
        }
        let query_rows = heads * head_dim * if self.gated { 2 } else { 1 };
        for layer in 0..self.value("num_hidden_layers") {
            let at = |name: &str| format!("model.layers.{layer}.{name}.weight");
            let form = Form {
                fp8: self.fp8,
                wide: layer % 2 == 1,
            };
            source.push(&at("input_layernorm"), &[hidden], 1.0, 0.2);
            source.push(&at("post_attention_layernorm"), &[hidden], 1.0, 0.2);
            if self.config["layer_types"][layer] == "linear_attention" {
                self.linear(&mut source, layer, form);
            } else {
                let kv = kv_heads * head_dim;
                for (name, shape) in [
                    ("self_attn.q_proj", [query_rows, hidden]),
                    ("self_attn.k_proj", [kv, hidden]),
                    ("self_attn.v_proj", [kv, hidden]),
                    ("self_attn.o_proj", [hidden, heads * head_dim]),
                ] {
                    form.push(&mut source, &at(name), shape);
                }
                if self.gated {
                    source.push(&at("self_attn.q_norm"), &[head_dim], 1.0, 0.2);
                    source.push(&at("self_attn.k_norm"), &[head_dim], 1.0, 0.2);
                }
            }
            let first_dense = self.config["first_k_dense_replace"].as_u64().unwrap_or(0) as usize;
            if self.config.get("num_experts").is_none() || layer < first_dense {
                form.mlp(
                    &mut source,
                    &format!("model.layers.{layer}.mlp"),
                    intermediate,
                    hidden,
                );
                continue;
            }
            let experts = self.value("num_experts");
            let width = self.value("moe_intermediate_size");
            source.push(&at("mlp.gate"), &[experts, hidden], 0.0, weight);
            for expert in 0..experts {
                let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
                form.mlp(&mut source, &prefix, width, hidden);
            }
            let counted = self.config.get("n_shared_experts").is_some();
            let shared = match self.config["shared_expert_intermediate_size"].as_u64() {
                Some(shared) => Some(shared as usize),
                None => counted.then_some(width),
            };
            if let Some(shared) = shared {
                let prefix = format!("model.layers.{layer}.mlp.shared_expert");
                form.mlp(&mut source, &prefix, shared, hidden);
                if !counted {
                    source.push(&at("mlp.shared_expert_gate"), &[1, hidden], 0.0, weight);
                }
            }
        }
        source
    }

    /// A Gated DeltaNet layer's tensors: the fused q/k/v, output gate and out projections in the
    /// layer's form, the beta and decay projections BF16, the conv as `[width][1][taps]`, decay
    /// rates that keep a head's state within a few steps' memory, and the gated norm.
    fn linear(&self, source: &mut Memory, layer: usize, form: Form) {
        let hidden = self.value("hidden_size");
        let (key_heads, heads) = (
            self.value("linear_num_key_heads"),
            self.value("linear_num_value_heads"),
        );
        let (key_dim, value_dim) = (
            self.value("linear_key_head_dim"),
            self.value("linear_value_head_dim"),
        );
        let taps = self.value("linear_conv_kernel_dim");
        let (key_width, value_width) = (key_heads * key_dim, heads * value_dim);
        let conv_width = 2 * key_width + value_width;
        let at = |name: &str| format!("model.layers.{layer}.linear_attn.{name}");
        for (name, shape) in [
            ("in_proj_qkv.weight", [conv_width, hidden]),
            ("in_proj_z.weight", [value_width, hidden]),
            ("out_proj.weight", [hidden, value_width]),
        ] {
            form.push(source, &at(name), shape);
        }
        let weight = (hidden as f32).sqrt().recip();
        source.push(&at("in_proj_b.weight"), &[heads, hidden], 0.0, weight);
        source.push(&at("in_proj_a.weight"), &[heads, hidden], 0.0, weight);
        source.push(&at("conv1d.weight"), &[conv_width, 1, taps], 0.0, 0.5);
        source.push(&at("A_log"), &[heads], -0.5, 0.5);
        source.push(&at("dt_bias"), &[heads], 0.0, 0.5);
        source.push(&at("norm.weight"), &[value_dim], 1.0, 0.2);
    }
}

/// How a layer stores its projections.
#[derive(Clone, Copy)]
struct Form {
    /// Block-scaled E4M3 rather than BF16.
    fp8: bool,
    /// F32 scale grids rather than BF16.
    wide: bool,
}

impl Form {
    /// A projection whose values stay within about `1 / sqrt(columns)`.
    fn push(self, source: &mut Memory, name: &str, [rows, columns]: [usize; 2]) {
        let weight = (columns as f32).sqrt().recip();
        if self.fp8 {
            // The codes' root mean square is about a uniform unit value's, and the tile scales
            // average one in `weight`.
            source.push_fp8(name, [rows, columns], weight, self.wide);
        } else {
            source.push(name, &[rows, columns], 0.0, weight);
        }
    }

    /// One SwiGLU MLP of `width` under `prefix`.
    fn mlp(self, source: &mut Memory, prefix: &str, width: usize, hidden: usize) {
        self.push(
            source,
            &format!("{prefix}.gate_proj.weight"),
            [width, hidden],
        );
        self.push(source, &format!("{prefix}.up_proj.weight"), [width, hidden]);
        self.push(
            source,
            &format!("{prefix}.down_proj.weight"),
            [hidden, width],
        );
    }
}

fn plain() -> Fixture {
    Fixture {
        config: json!({
            "model_type": "acme_dense",
            "architectures": ["AcmeDenseForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": false
        }),
        gated: false,
        tied: false,
        fp8: false,
    }
}

fn gated() -> Fixture {
    Fixture {
        config: json!({
            "model_type": "acme_dense_gated",
            "architectures": ["AcmeDenseGatedForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 520,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.5,
            "hidden_act": "silu",
            "attn_output_gate": true,
            "tie_word_embeddings": true
        }),
        gated: true,
        tied: true,
        fp8: false,
    }
}

/// Qwen's routed form on a family no pack names: a softmax over eight experts, two renormalised
/// picks, and a sigmoid-gated shared expert twice an expert's width.
fn routed() -> Fixture {
    Fixture {
        config: json!({
            "model_type": "acme_moe",
            "architectures": ["AcmeMoeForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 256,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": false
        }),
        gated: false,
        tied: false,
        fp8: false,
    }
}

/// The independent scores: a dense first layer, then a sigmoid over sixteen experts, four picks
/// kept as scored and scaled, an expert width the staging pads (136), and one ungated shared
/// expert of an expert's width; gated attention and a tied head.
fn sigmoid_routed() -> Fixture {
    Fixture {
        config: json!({
            "model_type": "acme_moe_sigmoid",
            "architectures": ["AcmeMoeSigmoidForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 384,
            "moe_intermediate_size": 136,
            "n_shared_experts": 1,
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "first_k_dense_replace": 1,
            "scoring_func": "sigmoid",
            "norm_topk_prob": false,
            "routed_scaling_factor": 2.5,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.5,
            "hidden_act": "silu",
            "attn_output_gate": true,
            "tie_word_embeddings": true
        }),
        gated: true,
        tied: true,
        fp8: false,
    }
}

/// A sqrt-softplus score over twelve experts, three renormalised picks, and no shared expert.
fn softplus_routed() -> Fixture {
    Fixture {
        config: json!({
            "model_type": "acme_moe_softplus",
            "architectures": ["AcmeMoeSoftplusForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 64,
            "num_experts": 12,
            "num_experts_per_tok": 3,
            "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": true,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": false
        }),
        gated: false,
        tied: false,
        fp8: false,
    }
}

/// The Qwen3 FP8 releases' form on a dense family: every projection E4M3 under a BF16 (odd
/// layers: F32) inverse scale a 128 x 128 tile, gated normed attention whose 64-row key and value
/// projections fill half a tile, and an MLP of three tiles.
fn fp8_dense() -> Fixture {
    Fixture {
        config: json!({
            "model_type": "acme_dense_fp8",
            "architectures": ["AcmeDenseFp8ForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 384,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_act": "silu",
            "attn_output_gate": true,
            "tie_word_embeddings": false,
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128]
            }
        }),
        gated: true,
        tied: false,
        fp8: true,
    }
}

/// [routed]'s checkpoint with block-scaled FP8 experts, shared expert and attention; the router
/// and the shared expert's gate stay BF16, as the releases keep them.
fn fp8_routed() -> Fixture {
    let mut fixture = routed();
    fixture.config["model_type"] = json!("acme_moe_fp8");
    fixture.config["architectures"] = json!(["AcmeMoeFp8ForCausalLM"]);
    fixture.config["quantization_config"] = json!({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128]
    });
    fixture.fp8 = true;
    fixture
}

/// A hybrid family no pack names: Gated DeltaNet layers (two key heads, four value heads of 128,
/// a SiLU gate, four taps) between gated normed attention layers, the first two chunks apart.
fn hybrid() -> Fixture {
    Fixture {
        config: json!({
            "model_type": "acme_hybrid",
            "architectures": ["AcmeHybridForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention", "full_attention", "linear_attention", "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.5,
            "hidden_act": "silu",
            "attn_output_gate": true,
            "tie_word_embeddings": false
        }),
        gated: true,
        tied: false,
        fp8: false,
    }
}

/// [hybrid]'s family at the other end of the scan's pairs: 64-wide heads (four key heads, eight
/// value heads), a sigmoid gate, three taps, three of eight routed experts with a gated shared
/// expert, and block-scaled FP8 projections.
fn hybrid_fp8_routed() -> Fixture {
    let mut fixture = hybrid();
    let config = &mut fixture.config;
    config["model_type"] = json!("acme_hybrid_moe_fp8");
    config["architectures"] = json!(["AcmeHybridMoeFp8ForCausalLM"]);
    for (key, value) in [
        ("linear_num_key_heads", 4),
        ("linear_num_value_heads", 8),
        ("linear_key_head_dim", 64),
        ("linear_value_head_dim", 64),
        ("linear_conv_kernel_dim", 3),
        ("moe_intermediate_size", 128),
        ("shared_expert_intermediate_size", 256),
        ("num_experts", 8),
        // Two picks put a prompt row's last-layer router at a near tie (the whole row takes
        // another expert there, every layer before it within the tolerance); three do not.
        ("num_experts_per_tok", 3),
    ] {
        config[key] = json!(value);
    }
    config["output_gate_type"] = json!("sigmoid");
    config["norm_topk_prob"] = json!(true);
    config["quantization_config"] = json!({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128]
    });
    fixture.fp8 = true;
    fixture
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}

/// The fixture on the dense arm, with what opened it, or `None` where the arm cannot run.
fn open(
    fixture: &Fixture,
    what: &str,
) -> Option<(
    Architecture,
    ModelDescriptor,
    Arc<Memory>,
    DenseDeviceBackend,
)> {
    let (architecture, descriptor, source) = prepare(fixture, what)?;
    let context = descriptor.context_tokens;
    let device = DenseDeviceBackend::open(descriptor.clone(), &architecture, &*source, context)
        .expect("the dense arm opens");
    Some((architecture, descriptor, source, device))
}

/// The fixture's architecture, descriptor and tensors, or `None` where the arm cannot run.
fn prepare(fixture: &Fixture, what: &str) -> Option<(Architecture, ModelDescriptor, Arc<Memory>)> {
    let architecture =
        Architecture::from_json(&fixture.config.to_string()).expect("the fixture parses");
    let family = architecture.family.clone();
    if let Some(refusal) = sparkflash_ops::parts::dense_device_refusal(&architecture) {
        panic!("the dense arm refuses the {family} fixture: {refusal}");
    }
    if !DenseDeviceBackend::available() {
        eprintln!("skipping the {family} {what}: this build has no dense-arm kernels");
        return None;
    }
    if let Err(error) = sparkflash_kernels::cuda_driver::Cuda::load() {
        eprintln!("skipping the {family} {what}: no CUDA device ({error})");
        return None;
    }
    let context = architecture.max_position_embeddings;
    let descriptor = ModelDescriptor::from_architecture(
        family.clone(),
        &architecture,
        context,
        Capabilities::ALL,
    );
    Some((architecture, descriptor, Arc::new(fixture.source())))
}

/// An in-process group's rank whose collective has the device legs: the slots each leg fills
/// ([TpCollective::exchange_on_device], [TpCollective::scatter_on_device]) are read back, gathered
/// over the host bus and written into the peers' slots, so the arm sums them on the device as it
/// does over NCCL. The f32 wire leaves every bit as it was sent.
struct DeviceLegs {
    local: LocalRank,
    cuda: CudaHandle,
}

impl DeviceLegs {
    /// Every rank's `bytes` at `send`, in rank order.
    fn gather(&mut self, send: u64, bytes: usize) -> Result<Vec<u8>, EngineError> {
        let device = |detail: String| EngineError::Shape {
            what: "in-process device legs".to_owned(),
            detail,
        };
        let mut mine = vec![0u8; bytes];
        self.cuda
            .copy_dtoh(&mut mine, &Deviceptr(send))
            .map_err(device)?;
        let mine: Vec<f32> = mine
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        let mut everyone = vec![0.0f32; mine.len() * self.local.world()];
        self.local.all_gather_assign(&mine, &mut everyone)?;
        Ok(everyone
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect())
    }

    fn write(&self, at: u64, bytes: &[u8]) -> Result<(), EngineError> {
        self.cuda
            .copy_htod(&Deviceptr(at), bytes)
            .map_err(|detail| EngineError::Shape {
                what: "in-process device legs".to_owned(),
                detail,
            })
    }
}

impl TpCollective for DeviceLegs {
    fn world(&self) -> usize {
        self.local.world()
    }
    fn rank(&self) -> usize {
        self.local.rank()
    }
    fn label(&self) -> &'static str {
        "in-process device legs"
    }
    fn all_reduce_assign(&mut self, values: &mut [f32]) -> Result<(), EngineError> {
        self.local.all_reduce_assign(values)
    }
    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        self.local.all_gather_assign(local, out)
    }
    fn device_wire(&self) -> Option<Wire> {
        Some(Wire::F32)
    }
    fn exchange_on_device(
        &mut self,
        _stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), EngineError> {
        let everyone = self.gather(send, bytes)?;
        for peer in (0..self.world()).filter(|peer| *peer != self.rank()) {
            let at = slots + (peer * bytes) as u64;
            self.write(at, &everyone[peer * bytes..][..bytes])?;
        }
        Ok(())
    }
    fn scatter_on_device(
        &mut self,
        _stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), EngineError> {
        let chunk = bytes / self.world();
        let everyone = self.gather(send, bytes)?;
        for peer in (0..self.world()).filter(|peer| *peer != self.rank()) {
            let at = slots + (peer * chunk) as u64;
            self.write(at, &everyone[peer * bytes + self.rank() * chunk..][..chunk])?;
        }
        Ok(())
    }
}

/// A group of `world` ranks on this one device, the arm's sums on the host leg or (with
/// `device_legs`) the one-shot and two-shot device legs.
#[derive(Clone, Copy)]
struct Group {
    world: usize,
    device_legs: bool,
}

impl Group {
    /// Rank `rank`'s collective over `bus`, opened on the calling thread.
    fn runtime(self, bus: &Arc<LocalBus>, rank: usize) -> TpRuntime {
        let local = LocalRank {
            rank,
            bus: Arc::clone(bus),
        };
        let collective: Box<dyn TpCollective> = if self.device_legs {
            let cuda = CudaHandle::new(Cuda::load().expect("a CUDA device"));
            Box::new(DeviceLegs { local, cuda })
        } else {
            Box::new(local)
        };
        let layout = TpLayout {
            rank,
            world_size: self.world,
        };
        TpRuntime::new(layout, collective).expect("the group's collective")
    }
}

/// One step's every layer trace and logits, as a rank answered them.
type Answer = (Vec<Vec<f32>>, Vec<f32>);

/// A worker rank on its own thread: it steps each feed it is sent and answers what it computed.
struct Peer {
    feeds: mpsc::Sender<Vec<u32>>,
    answers: mpsc::Receiver<Answer>,
    thread: std::thread::JoinHandle<()>,
}

impl Peer {
    fn start(
        group: Group,
        bus: &Arc<LocalBus>,
        rank: usize,
        (architecture, descriptor, source): (&Architecture, &ModelDescriptor, &Arc<Memory>),
    ) -> Self {
        let (feeds, fed) = mpsc::channel::<Vec<u32>>();
        let (answer, answers) = mpsc::channel();
        let (bus, architecture, descriptor, source) = (
            Arc::clone(bus),
            architecture.clone(),
            descriptor.clone(),
            Arc::clone(source),
        );
        let thread = std::thread::spawn(move || {
            let context = descriptor.context_tokens;
            let backend = DenseDeviceBackend::open_rank(
                descriptor,
                &architecture,
                &*source,
                context,
                group.runtime(&bus, rank),
            )
            .unwrap_or_else(|error| panic!("rank {rank} opens: {error}"));
            let mut session = backend.session().expect("a worker session");
            session.trace(true);
            // An empty answer says the rank opened, before rank 0 waits on it in a sum.
            if answer.send((Vec::new(), Vec::new())).is_err() {
                return;
            }
            for tokens in fed {
                let logits = session.forward(&tokens).expect("a worker forward");
                if answer.send((session.take_trace(), logits)).is_err() {
                    return;
                }
            }
        });
        let peer = Self {
            feeds,
            answers,
            thread,
        };
        peer.answers.recv().expect("the worker rank opens");
        peer
    }
}

/// `values` as their bits, for comparing to the bit.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Runs `fixture` through both backends and returns the report, or `None` where the arm cannot
/// run.
fn parity(fixture: &Fixture, prompt: usize, steps: usize) -> Option<ParityReport> {
    parity_over(fixture, prompt, steps, None).map(|(report, _)| report)
}

/// [parity] with the arm as rank 0 of `group`, every worker rank's layers and logits checked to
/// the bit against rank 0's; answers the report and rank 0's every answer.
fn group_parity(
    fixture: &Fixture,
    prompt: usize,
    steps: usize,
    group: Group,
) -> Option<(ParityReport, Vec<Answer>)> {
    parity_over(fixture, prompt, steps, Some(group))
}

fn parity_over(
    fixture: &Fixture,
    prompt: usize,
    steps: usize,
    group: Option<Group>,
) -> Option<(ParityReport, Vec<Answer>)> {
    let (architecture, descriptor, source) = prepare(fixture, "parity report")?;
    let family = architecture.family.clone();
    let context = descriptor.context_tokens;
    let (device, peers) = match group {
        None => (
            DenseDeviceBackend::open(descriptor.clone(), &architecture, &*source, context)
                .expect("the dense arm opens"),
            Vec::new(),
        ),
        Some(group) => {
            let bus = Arc::new(LocalBus::new(group.world).with_wire(Wire::F32));
            let peers: Vec<Peer> = (1..group.world)
                .map(|rank| Peer::start(group, &bus, rank, (&architecture, &descriptor, &source)))
                .collect();
            let device = DenseDeviceBackend::open_rank(
                descriptor.clone(),
                &architecture,
                &*source,
                context,
                group.runtime(&bus, 0),
            )
            .expect("rank 0 opens");
            assert_eq!(device.rank(), (0, group.world));
            (device, peers)
        }
    };
    let host =
        DynamicHostBackend::open(descriptor, Arc::new(architecture.clone()), source, context)
            .expect("the host reference opens");
    let mut device = device.session().expect("a device session");
    let mut host = host.host_session().expect("a host session");
    device.trace(true);
    host.trace(true);
    let tokens: Vec<u32> = (0..prompt).map(|index| (index * 37 % 251) as u32).collect();
    let mut report = ParityReport::new(format!("{family} dense arm against the host reference"));
    let record =
        |report: &mut ParityReport, what: &str, kind: OpKind, reference: &[f32], actual: &[f32]| {
            let _ = report.record(what, kind, reference, actual, TOLERANCE);
            let worst = reference
                .iter()
                .zip(actual)
                .map(|(reference, actual)| (reference - actual).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{family} {what}: {} values, largest difference {worst:e}",
                reference.len()
            );
        };
    let mut answered = Vec::new();
    let mut step = |report: &mut ParityReport, label: &str, tokens: &[u32]| -> u32 {
        let mut expected = Vec::new();
        for token in tokens {
            expected = host.forward(*token).expect("the host forward");
        }
        for peer in &peers {
            peer.feeds
                .send(tokens.to_vec())
                .expect("a worker rank takes the feed");
        }
        let actual = device.forward(tokens).expect("the device forward");
        let trace = device.take_trace();
        for (index, peer) in peers.iter().enumerate() {
            let (layers, logits) = peer.answers.recv().expect("a worker rank answers");
            assert_eq!(layers.len(), trace.len());
            for (layer, (mine, theirs)) in trace.iter().zip(&layers).enumerate() {
                assert!(
                    bits(mine) == bits(theirs),
                    "{family} {label}: rank {} layer {layer} differs from rank 0's",
                    index + 1
                );
            }
            assert!(
                bits(&actual) == bits(&logits),
                "{family} {label}: rank {}'s logits differ from rank 0's",
                index + 1
            );
        }
        for (layer, (reference, actual)) in host.take_trace().iter().zip(&trace).enumerate() {
            record(
                report,
                &format!("{label} layer {layer}"),
                OpKind::GqaAttention,
                reference,
                actual,
            );
        }
        record(
            report,
            &format!("{label} logits"),
            OpKind::DenseGemm,
            &expected,
            &actual,
        );
        let (want, got) = (argmax(&expected), argmax(&actual));
        // The next token must agree wherever the reference's choice is clear of the tolerance;
        // a near tie is reported, and both backends go on from the reference's token.
        let runner_up = expected
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != want as usize)
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        let margin = expected[want as usize] - runner_up;
        if margin > 4.0 * TOLERANCE.absolute {
            assert_eq!(got, want, "{family} {label}: the next token differs");
        } else {
            eprintln!("{family} {label}: a near tie ({margin:e}), reference {want}, arm {got}");
        }
        answered.push((trace, actual));
        want
    };
    let mut next = step(&mut report, "prompt", &tokens);
    for index in 0..steps {
        next = step(&mut report, &format!("decode {index}"), &[next]);
    }
    for peer in peers {
        drop(peer.feeds);
        peer.thread.join().expect("a worker rank ends");
    }
    Some((report, answered))
}

/// `fixture` at TP=2 on the host leg and on the device legs: each passes [group_parity], and the
/// two legs' answers agree to the bit.
fn two_ranks(fixture: &Fixture, prompt: usize, steps: usize) {
    let host = group_parity(
        fixture,
        prompt,
        steps,
        Group {
            world: 2,
            device_legs: false,
        },
    );
    let Some((report, host)) = host else {
        return;
    };
    assert_passed(Some(report));
    let (report, device) = group_parity(
        fixture,
        prompt,
        steps,
        Group {
            world: 2,
            device_legs: true,
        },
    )
    .expect("the device legs run where the host leg did");
    assert_passed(Some(report));
    assert_eq!(host.len(), device.len());
    for (step, ((host_layers, host_logits), (device_layers, device_logits))) in
        host.iter().zip(&device).enumerate()
    {
        for (layer, (host, device)) in host_layers.iter().zip(device_layers).enumerate() {
            assert!(
                bits(host) == bits(device),
                "step {step} layer {layer}: the device legs' residual differs from the host leg's"
            );
        }
        assert!(
            bits(host_logits) == bits(device_logits),
            "step {step}: the device legs' logits differ from the host leg's"
        );
    }
}

#[test]
fn two_ranks_of_a_two_chunk_llama_shaped_prompt_match_the_host_reference() {
    two_ranks(&plain(), 300, 3);
}

#[test]
fn two_ranks_sharing_one_key_value_head_match_the_host_reference() {
    // One key/value head for two ranks: both read it, each its own two query heads; the MLP
    // width (520) splits into two 260-wide shares the row GEMV takes.
    two_ranks(&gated(), 300, 3);
}

#[test]
fn two_ranks_of_sigmoid_routed_experts_match_the_host_reference() {
    // A 136-wide expert splits into 72 and 64 columns, whole runs of the eight the BF16 expert
    // passes read at a time.
    two_ranks(&sigmoid_routed(), 40, 3);
}

#[test]
fn two_ranks_of_gated_deltanet_layers_match_the_host_reference() {
    two_ranks(&hybrid(), 300, 3);
}

#[test]
fn two_ranks_of_fp8_gated_deltanet_layers_and_experts_match_the_host_reference() {
    // Each rank holds whole 128 x 128 tiles of every expert (two tiles split one apiece) and
    // a 64-row key/value share inside one tile.
    let mut fixture = hybrid_fp8_routed();
    fixture.config["moe_intermediate_size"] = json!(256);
    two_ranks(&fixture, 300, 3);
}

fn assert_passed(report: Option<ParityReport>) {
    if let Some(report) = report {
        assert!(report.passed(), "{}: {:?}", report.label, report.failures());
    }
}

#[test]
fn a_two_chunk_llama_shaped_prompt_matches_the_host_reference_layer_by_layer() {
    assert_passed(parity(&plain(), 300, 3));
}

#[test]
fn a_gated_normed_tied_checkpoint_matches_the_host_reference_layer_by_layer() {
    assert_passed(parity(&gated(), 40, 3));
}

#[test]
fn a_prompt_of_decode_rows_matches_the_host_reference_layer_by_layer() {
    // Six rows stay on the row GEMV's batch, the decode path's kernel.
    assert_passed(parity(&gated(), 6, 2));
}

#[test]
fn a_routed_checkpoint_with_a_gated_shared_expert_matches_the_host_reference_layer_by_layer() {
    assert_passed(parity(&routed(), 300, 3));
}

#[test]
fn sigmoid_routed_experts_after_a_dense_layer_match_the_host_reference_layer_by_layer() {
    assert_passed(parity(&sigmoid_routed(), 40, 3));
}

#[test]
fn sqrt_softplus_routed_experts_without_a_shared_expert_match_the_host_reference() {
    assert_passed(parity(&softplus_routed(), 40, 3));
}

#[test]
fn routed_decode_rows_match_the_host_reference_layer_by_layer() {
    assert_passed(parity(&routed(), 6, 2));
}

#[test]
fn block_scaled_fp8_projections_match_the_host_reference_layer_by_layer() {
    // Two chunks: 256 rows on the FP8 tensor-core tile, then 44, and decode rows on the tiled
    // E4M3 GEMV.
    assert_passed(parity(&fp8_dense(), 300, 3));
}

#[test]
fn block_scaled_fp8_decode_rows_match_the_host_reference_layer_by_layer() {
    assert_passed(parity(&fp8_dense(), 6, 2));
}

#[test]
fn block_scaled_fp8_experts_match_the_host_reference_layer_by_layer() {
    assert_passed(parity(&fp8_routed(), 300, 3));
}

#[test]
fn gated_deltanet_layers_match_the_host_reference_layer_by_layer() {
    // Two chunks: the second's conv reads the first's history, its scan the first's state.
    assert_passed(parity(&hybrid(), 300, 3));
}

#[test]
fn gated_deltanet_decode_rows_match_the_host_reference_layer_by_layer() {
    assert_passed(parity(&hybrid(), 6, 2));
}

#[test]
fn narrow_sigmoid_gated_deltanet_layers_with_fp8_experts_match_the_host_reference() {
    assert_passed(parity(&hybrid_fp8_routed(), 300, 3));
}

#[test]
fn fused_gated_deltanet_lanes_answer_every_step_as_each_conversation_alone() {
    // A row's conv and scan read only its own sequence's state, and the prep and the gated norm
    // are row-wise, so a lane's recurrence is its own at any batch.
    lanes_answer_as_alone(&hybrid());
}

#[test]
fn fused_fp8_lanes_answer_every_step_as_each_conversation_alone() {
    // A decode row's E4M3 sums are its own at any batch: the tiled GEMV splits by the matrix.
    lanes_answer_as_alone(&fp8_routed());
}

/// One conversation turn: each step's token and its logits' bits, the prompt's first.
type Turn = Vec<(u32, Vec<u32>)>;

/// `prompt` and `steps` decode steps on `session`, feeding each step's token.
fn converse(session: &mut dyn ModelSession, prompt: &[u32], steps: usize) -> Turn {
    let mut turn = Turn::new();
    let mut step = session.prefill(prompt).expect("the prompt");
    for index in 0..=steps {
        let token = step.next_token.expect("the arm picks every token");
        turn.push((
            token,
            step.logits.iter().map(|value| value.to_bits()).collect(),
        ));
        if index < steps {
            step = session.decode(token).expect("a decode step");
        }
    }
    turn
}

fn prompt_of(length: usize, salt: usize) -> Vec<u32> {
    (0..length)
        .map(|index| ((index * 31 + salt * 7 + 3) % 251) as u32)
        .collect()
}

#[test]
fn fused_lanes_answer_every_step_as_each_conversation_alone() {
    lanes_answer_as_alone(&gated());
}

#[test]
fn fused_routed_lanes_answer_every_step_as_each_conversation_alone() {
    // The expert passes group a run's rows two at a time, and a row's dot is its own at any
    // grouping, so a lane's experts sum as they would alone.
    lanes_answer_as_alone(&sigmoid_routed());
}

fn lanes_answer_as_alone(fixture: &Fixture) {
    let Some((_, _, _, mut backend)) = open(fixture, "fused lanes") else {
        return;
    };
    backend.enable_fusion(4).expect("the fused engine starts");
    // A decode-sized prompt, a wide-tile one, one in two parts and one of a single id.
    let prompts: Vec<Vec<u32>> = [(5, 1), (40, 2), (300, 3), (1, 4)]
        .into_iter()
        .map(|(length, salt)| prompt_of(length, salt))
        .collect();
    let steps = 6;
    let alone: Vec<Turn> = prompts
        .iter()
        .map(|prompt| {
            let mut session = backend.open_session().expect("a lane");
            converse(&mut *session, prompt, steps)
        })
        .collect();
    let backend = &backend;
    let together: Vec<Turn> = std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .map(|prompt| {
                scope.spawn(move || {
                    let mut session = backend.open_session().expect("a lane");
                    converse(&mut *session, prompt, steps)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a lane's thread"))
            .collect()
    });
    for (index, (alone, together)) in alone.iter().zip(&together).enumerate() {
        assert_eq!(
            alone.iter().map(|(token, _)| *token).collect::<Vec<_>>(),
            together.iter().map(|(token, _)| *token).collect::<Vec<_>>(),
            "conversation {index}: the tokens differ together"
        );
        assert!(
            alone == together,
            "conversation {index}: a step's logits differ together"
        );
    }
}

#[test]
fn a_kept_lane_continues_its_conversation_and_goes_back_to_its_prompt() {
    kept_lane(&plain());
}

#[test]
fn a_kept_gated_deltanet_lane_continues_and_goes_back_to_its_prompts_state() {
    // The recurrent state past the prompt is not the prompt's: the rewind copies it back.
    kept_lane(&hybrid());
}

/// A kept lane continues its conversation as a cold prompt would, and a prompt that parts from
/// the reply goes back to the end of the last prompt, again as a cold prompt would.
fn kept_lane(fixture: &Fixture) {
    let Some((_, _, _, mut backend)) = open(fixture, "kept lane") else {
        return;
    };
    backend.enable_fusion(1).expect("the fused engine starts");
    let first = prompt_of(20, 5);
    let mut session = backend.open_session().expect("a lane");
    let reply = converse(&mut *session, &first, 4);
    // The lane holds the prompt and every token fed since: the reply's last pick was not fed.
    let mut held = first.clone();
    held.extend(reply[..reply.len() - 1].iter().map(|(token, _)| *token));
    let mut next = held.clone();
    next.extend([9, 10]);
    let continued = session.prefill(&next).expect("the continued turn");
    assert_eq!(
        session.cached_tokens(),
        held.len(),
        "the turn continues the lane"
    );
    let against_cold = |tokens: &[u32], logits: &[f32], what: &str| {
        let mut cold = backend.session().expect("a cold session");
        let expected = cold.forward(tokens).expect("the cold prompt");
        let worst = expected
            .iter()
            .zip(logits)
            .map(|(cold, warm)| (cold - warm).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{what}'s logits against cold: largest difference {worst:e}");
        assert!(
            worst <= 4.0 * TOLERANCE.absolute,
            "{what}'s logits are {worst} from cold"
        );
    };
    against_cold(&next, &continued.logits, "a continued turn");
    // Two steps past that turn, a prompt that parts from their reply goes back to the end of
    // the last prompt.
    let token = continued.next_token.expect("the arm picks every token");
    let after = session.decode(token).expect("a decode step");
    let _ = session
        .decode(after.next_token.expect("the arm picks every token"))
        .expect("a decode step");
    let mut parted = next.clone();
    parted.push(if token == 77 { 78 } else { 77 });
    let rewound = session
        .prefill(&parted)
        .expect("a turn from the prompt's end");
    assert_eq!(
        session.cached_tokens(),
        next.len(),
        "the turn goes back to the end of the last prompt"
    );
    against_cold(&parted, &rewound.logits, "a turn from the rewind point");
}
