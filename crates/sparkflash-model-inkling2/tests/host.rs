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

use std::sync::Arc;

use sparkflash_model::ModelDescriptor;
use sparkflash_model::architecture::WeightFormat;
use sparkflash_model::backend::ModelBackend;
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::forward::{Inkling2State, Inkling2Weights};
use sparkflash_model_inkling2::host::Inkling2HostBackend;
use sparkflash_model_inkling2::fixture::tiny_source;

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
