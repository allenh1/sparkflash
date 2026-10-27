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

//! A deliberately non-V4.1 model definition, exercised on the generic weight path.
//!
//! The toy has a dense SwiGLU block, its own tiny vocabulary and an ASCII prompt codec. It is not
//! a real checkpoint and it does not implement the device path; its job is to keep the contracts
//! from silently assuming MLA, mHC, Engram, DSpark, a 129,280-token vocabulary or a four-rank
//! topology.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};
use sparkflash_model::{Capabilities, ModelDescriptor, PromptCodec, PromptError};
use sparkflash_weights::{
    DType, TensorSpec, TensorView, WeightError, WeightSource, check_source_against_specs, linalg,
};

struct ToyTokens;

impl PromptCodec for ToyTokens {
    fn encode_chat(&self, request: &Value) -> Result<Vec<u32>, PromptError> {
        let messages = request
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                PromptError::invalid("messages must be an array", Some("messages"), None)
            })?;
        let mut text = String::new();
        for message in messages {
            let content = message
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    PromptError::invalid("content must be a string", Some("messages"), None)
                })?;
            text.push_str(content);
        }
        self.encode_text(&text, true)
    }

    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError> {
        let mut ids = Vec::with_capacity(text.len() + usize::from(add_special_tokens));
        if add_special_tokens {
            ids.push(1);
        }
        for byte in text.bytes() {
            if byte >= 32 {
                ids.push(u32::from(byte) + 2);
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Option<String> {
        let mut text = String::new();
        for &id in ids {
            if id == 1 {
                continue;
            }
            text.push(char::from(u8::try_from(id.checked_sub(2)?).ok()?));
        }
        Some(text)
    }

    fn token_str(&self, id: u32) -> Option<String> {
        self.decode(&[id])
    }
}

struct ToyWeights {
    spec: TensorSpec,
    bytes: Vec<u8>,
}

impl WeightSource for ToyWeights {
    fn names(&self) -> Vec<String> {
        vec![self.spec.name.clone()]
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        if name == self.spec.name {
            Ok(self.spec.clone())
        } else {
            Err(WeightError::Missing {
                name: name.to_owned(),
            })
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        if name == self.spec.name {
            Ok(self.bytes.clone())
        } else {
            Err(WeightError::Missing {
                name: name.to_owned(),
            })
        }
    }
}

fn toy_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "toy/tiny-dense".to_owned(),
        family: "toy-dense".to_owned(),
        weights_id: "toy-dense".to_owned(),
        context_tokens: 128,
        max_output_tokens: 16,
        vocab_size: 258,
        capabilities: Capabilities::default(),
    }
}

fn bf16(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    ((rounded >> 16) as u16).to_le_bytes()
}

#[test]
fn the_toy_descriptor_is_not_the_v41_shape() {
    let descriptor = toy_descriptor();
    assert!(!descriptor.is_v41_flash());
    assert_eq!(descriptor.context_tokens, 128);
    assert_eq!(descriptor.vocab_size, 258);
    assert_eq!(descriptor.capabilities, Capabilities::default());
    assert_ne!(descriptor.family, "deepseek-v4.1");
}

#[test]
fn the_toy_prompt_codec_round_trips_through_the_contract() {
    let codec: Arc<dyn PromptCodec> = Arc::new(ToyTokens);
    let ids = codec
        .encode_chat(&json!({
            "messages": [
                {"role": "system", "content": "hi"},
                {"role": "user", "content": "ok"}
            ]
        }))
        .expect("the toy chat request encodes");
    assert_eq!(ids.first(), Some(&1));
    assert_eq!(codec.decode(&ids).as_deref(), Some("hiok"));
    assert_eq!(codec.token_str(ids[1]).as_deref(), Some("h"));
}

#[test]
fn the_toy_dense_block_runs_on_the_generic_weight_and_arithmetic_path() {
    let spec = TensorSpec {
        name: "blk.0.ffn.down.weight".to_owned(),
        dtype: DType::BF16,
        shape: vec![2, 3],
    };
    let mut bytes = Vec::new();
    for value in [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0] {
        bytes.extend_from_slice(&bf16(value));
    }
    let source = ToyWeights {
        spec: spec.clone(),
        bytes,
    };
    let report = check_source_against_specs(&source, std::slice::from_ref(&spec));
    assert!(report.is_conformant(), "{report:?}");

    let data = source.read(&spec.name).expect("the toy weight reads");
    let view = TensorView {
        name: &spec.name,
        dtype: spec.dtype,
        shape: &spec.shape,
        data: &data,
    };
    let mut out = [0.0f32; 2];
    linalg::matmul_bf16(&mut out, &[2.0, 3.0, 4.0], view.data, 2, 3, 1)
        .expect("the generic bf16 matmul accepts the toy geometry");
    assert_eq!(out, [2.0, 3.0]);
}

#[test]
fn tensor_specs_use_the_models_own_names() {
    let specs = BTreeMap::from([
        (
            "blk.0.attn.q.weight".to_owned(),
            TensorSpec {
                name: "blk.0.attn.q.weight".to_owned(),
                dtype: DType::BF16,
                shape: vec![4, 8],
            },
        ),
        (
            "blk.0.ffn.gate.weight".to_owned(),
            TensorSpec {
                name: "blk.0.ffn.gate.weight".to_owned(),
                dtype: DType::BF16,
                shape: vec![16, 8],
            },
        ),
    ]);
    assert!(specs.keys().all(|name| name.starts_with("blk.0.")));
    assert_eq!(specs["blk.0.attn.q.weight"].storage_bytes(), 4 * 8 * 2);
}
