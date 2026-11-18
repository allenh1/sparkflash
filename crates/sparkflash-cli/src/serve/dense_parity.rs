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
//! The arm's serving contract is checked here too, on the same fixtures: fused lanes answer every
//! step's logits to the bit as each conversation alone, and a kept lane continues its
//! conversation or goes back to the end of its prompt.
//!
//! Without the kernels or a CUDA device the test says so and passes: the gate runs where the arm
//! can.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};
use sparkflash_model::{Architecture, Capabilities, ModelBackend, ModelDescriptor, ModelSession};
use sparkflash_model_dense::DenseDeviceBackend;
use sparkflash_ops::parity::{ParityReport, Tolerance};
use sparkflash_ops::plan::OpKind;
use sparkflash_weights::dtype::DType;
use sparkflash_weights::spec::TensorSpec;
use sparkflash_weights::weights::{WeightError, WeightSource};

use super::dynamic_host::DynamicHostBackend;

/// The bound one residual or logit element meets: the BF16 cache's rounding of every key and
/// value, carried through two layers.
const TOLERANCE: Tolerance = Tolerance::new(2e-2, 2e-2);

struct Memory(BTreeMap<String, (TensorSpec, Vec<u8>)>);

impl Memory {
    /// A BF16 tensor of deterministic values in `[-scale, scale]` around `centre`.
    fn push(&mut self, name: &str, shape: &[usize], centre: f32, scale: f32) {
        let elements: usize = shape.iter().product();
        let salt = name.bytes().fold(0x811c_9dc5u32, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
        });
        let bytes = (0..elements as u32)
            .flat_map(|index| {
                let mixed = (index ^ salt)
                    .wrapping_mul(0x9e37_79b9)
                    .rotate_left(13)
                    .wrapping_mul(0x85eb_ca6b);
                let unit = (mixed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
                let value = centre + unit * scale;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        self.0.insert(
            name.to_owned(),
            (
                TensorSpec {
                    name: name.to_owned(),
                    dtype: DType::BF16,
                    shape: shape.to_vec(),
                },
                bytes,
            ),
        );
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
            source.push(&at("input_layernorm"), &[hidden], 1.0, 0.2);
            source.push(&at("post_attention_layernorm"), &[hidden], 1.0, 0.2);
            source.push(&at("self_attn.q_proj"), &[query_rows, hidden], 0.0, weight);
            source.push(
                &at("self_attn.k_proj"),
                &[kv_heads * head_dim, hidden],
                0.0,
                weight,
            );
            source.push(
                &at("self_attn.v_proj"),
                &[kv_heads * head_dim, hidden],
                0.0,
                weight,
            );
            source.push(
                &at("self_attn.o_proj"),
                &[hidden, heads * head_dim],
                0.0,
                weight,
            );
            if self.gated {
                source.push(&at("self_attn.q_norm"), &[head_dim], 1.0, 0.2);
                source.push(&at("self_attn.k_norm"), &[head_dim], 1.0, 0.2);
            }
            source.push(&at("mlp.gate_proj"), &[intermediate, hidden], 0.0, weight);
            source.push(&at("mlp.up_proj"), &[intermediate, hidden], 0.0, weight);
            source.push(
                &at("mlp.down_proj"),
                &[hidden, intermediate],
                0.0,
                (intermediate as f32).sqrt().recip(),
            );
        }
        source
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
    }
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
    let source = Arc::new(fixture.source());
    let device = DenseDeviceBackend::open(descriptor.clone(), &architecture, &*source, context)
        .expect("the dense arm opens");
    Some((architecture, descriptor, source, device))
}

/// Runs `fixture` through both backends and returns the report, or `None` where the arm cannot
/// run.
fn parity(fixture: &Fixture, prompt: usize, steps: usize) -> Option<ParityReport> {
    let (architecture, descriptor, source, device) = open(fixture, "parity report")?;
    let family = architecture.family.clone();
    let context = descriptor.context_tokens;
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
    let mut step = |report: &mut ParityReport, label: &str, tokens: &[u32]| -> u32 {
        let mut expected = Vec::new();
        for token in tokens {
            expected = host.forward(*token).expect("the host forward");
        }
        let actual = device.forward(tokens).expect("the device forward");
        for (layer, (reference, actual)) in host
            .take_trace()
            .iter()
            .zip(device.take_trace())
            .enumerate()
        {
            record(
                report,
                &format!("{label} layer {layer}"),
                OpKind::GqaAttention,
                reference,
                &actual,
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
        want
    };
    let mut next = step(&mut report, "prompt", &tokens);
    for index in 0..steps {
        next = step(&mut report, &format!("decode {index}"), &[next]);
    }
    Some(report)
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
    let Some((_, _, _, mut backend)) = open(&gated(), "fused lanes") else {
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
    let Some((_, _, _, mut backend)) = open(&plain(), "kept lane") else {
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
    let mut cold = backend.session().expect("a cold session");
    let expected = cold.forward(&next).expect("the cold prompt");
    let worst = expected
        .iter()
        .zip(&continued.logits)
        .map(|(cold, continued)| (cold - continued).abs())
        .fold(0.0f32, f32::max);
    eprintln!("a continued turn's logits against cold: largest difference {worst:e}");
    assert!(
        worst <= 4.0 * TOLERANCE.absolute,
        "a continued turn's logits are {worst} from cold"
    );
    // Two steps past that turn, a prompt that parts from their reply goes back to the end of
    // the last prompt.
    let token = continued.next_token.expect("the arm picks every token");
    let after = session.decode(token).expect("a decode step");
    let _ = session
        .decode(after.next_token.expect("the arm picks every token"))
        .expect("a decode step");
    let mut parted = next.clone();
    parted.push(if token == 77 { 78 } else { 77 });
    session
        .prefill(&parted)
        .expect("a turn from the prompt's end");
    assert_eq!(
        session.cached_tokens(),
        next.len(),
        "the turn goes back to the end of the last prompt"
    );
}
