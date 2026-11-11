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

//! The tiny fixture through the real safetensors loader and host backend.

use sparkflash_model::backend::{ModelBackend, Sampling};
use sparkflash_model_dory::config::DoryConfig;
use sparkflash_model_dory::fixture;
use sparkflash_model_dory::forward::DoryState;
use sparkflash_model_dory::{DoryHostBackend, DoryWeights};
use sparkflash_weights::safetensors::open_checkpoint;

#[test]
fn the_tiny_fixture_loads_and_generates_through_the_host_backend() {
    let config = DoryConfig::tiny();
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    fixture::write_tensors(directory.path(), &config).expect("the fixture writes");
    let source = open_checkpoint(directory.path()).expect("the fixture opens");
    let weights = DoryWeights::load(&config, &source).expect("the weights load");
    assert_eq!(weights.blocks.len(), 4);
    let logits = weights.forward_prompt(&[1, 2, 3]).expect("the prompt runs");
    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
    let mut state = DoryState::new(&config);
    assert_eq!(state.slots(0), 1);
    assert_eq!(state.slots(1), 3);
    assert_eq!(state.slots(2), 3);
    assert_eq!(state.slots(3), 1);
    weights
        .forward_token(1, 0, &mut state)
        .expect("the first token runs");
    assert_eq!(state.position, 1);
}

#[test]
fn the_host_backend_prefills_and_decodes() {
    let config = DoryConfig::tiny();
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    fixture::write_tensors(directory.path(), &config).expect("the fixture writes");
    let source = open_checkpoint(directory.path()).expect("the fixture opens");
    let backend = DoryHostBackend::open(
        sparkflash_model_dory::descriptor(),
        config.clone(),
        &source,
        32,
    )
    .expect("the backend opens");
    assert_eq!(backend.descriptor().context_tokens, 32);
    let mut session = backend.open_session().expect("a session");
    let first = session.prefill(&[1, 2, 3]).expect("the prompt runs");
    assert_eq!(first.logits.len(), config.vocab_size);
    assert!(first.next_token.is_some());
    let second = session.decode(4).expect("the decode runs");
    assert_eq!(second.logits.len(), config.vocab_size);
}

#[test]
fn the_host_backend_honors_sampling() {
    let config = DoryConfig::tiny();
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    fixture::write_tensors(directory.path(), &config).expect("the fixture writes");
    let source = open_checkpoint(directory.path()).expect("the fixture opens");
    let backend = DoryHostBackend::open(
        sparkflash_model_dory::descriptor(),
        config.clone(),
        &source,
        32,
    )
    .expect("the backend opens");
    let mut greedy = backend.open_session().expect("a session");
    let chosen = greedy
        .prefill(&[1, 2, 3])
        .expect("the prompt runs")
        .next_token
        .expect("the fixture has no end-of-turn id");
    // A repetition penalty that rewrites the row before the greedy comparison must change the
    // choice when the prompt already holds the token that would otherwise win.
    let mut penalized = backend.open_session().expect("a session");
    penalized.set_sampling(Sampling {
        repetition_penalty: 100.0,
        ..Sampling::default()
    });
    let scored = penalized
        .prefill(&[chosen, 4])
        .expect("the penalized prompt runs")
        .next_token
        .expect("the fixture has no end-of-turn id");
    assert_ne!(
        scored, chosen,
        "the host backend ignored the request's sampling policy"
    );
}

#[test]
fn an_end_of_turn_id_ends_the_host_turn() {
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    let mut config = DoryConfig::tiny();
    fixture::write_tensors(directory.path(), &config).expect("the fixture writes");
    let source = open_checkpoint(directory.path()).expect("the fixture opens");
    let turn = |config: &DoryConfig| {
        let backend = DoryHostBackend::open(
            sparkflash_model_dory::descriptor(),
            config.clone(),
            &source,
            32,
        )
        .expect("the backend opens");
        let mut session = backend.open_session().expect("a session");
        let mut steps = vec![session.prefill(&[1, 2, 3]).expect("the prompt runs")];
        while steps.len() < 8 {
            let Some(token) = steps.last().and_then(|step| step.next_token) else {
                break;
            };
            steps.push(session.decode(token).expect("the decode runs"));
        }
        steps
    };
    assert!(config.eos_token_ids.is_empty());
    let reference = turn(&config);
    let tokens: Vec<u32> = reference
        .iter()
        .map(|step| step.next_token.expect("no end-of-turn id is set"))
        .collect();
    // The prompt's own token, then a later one the turn has not chosen before.
    let later = (1..tokens.len())
        .find(|index| !tokens[..*index].contains(&tokens[*index]))
        .expect("a decode step chooses a new token");
    for end in [0, later] {
        config.eos_token_ids = vec![tokens[end]];
        let ended = turn(&config);
        assert_eq!(ended.len(), end + 1, "the turn ends at step {end}");
        assert_eq!(ended[end].next_token, None);
        for (index, (one, other)) in reference.iter().zip(&ended).enumerate() {
            assert!(
                one.logits
                    .iter()
                    .zip(&other.logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "step {index}'s logits are unchanged"
            );
            if index < end {
                assert_eq!(one.next_token, other.next_token);
            }
        }
    }
}

#[test]
fn last_loop_mode_shares_one_cache_per_recurrent_block() {
    let mut config = DoryConfig::tiny();
    config.recurrent_kv_cache_mode = sparkflash_model_dory::KvCacheMode::LastLoop;
    for block in &mut config.blocks {
        if block.partition == sparkflash_model_dory::Partition::Recurrent {
            block.cache_slots = 1;
        }
    }
    let state = DoryState::new(&config);
    assert_eq!(state.slots(1), 1);
    assert_eq!(state.slots(2), 1);
}
