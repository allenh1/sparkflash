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

//! A SenseNova-Loop turn ends at the checkpoint's own stop id.
//!
//! The test lives here rather than in `sparkflash-model-sensenova` because it renders the prompt
//! with the same ChatML codec the dynamic serve path picks, and a model crate may not depend on
//! another model crate (`sparkflash-ops/tests/one_stack.rs`). The CLI already carries both.

use std::path::PathBuf;

use sparkflash_model::backend::ModelBackend;
use sparkflash_model::{ModelDescriptor, PromptCodec};

#[test]
#[ignore = "loads the 11 GB release; run with SENSENOVA_FP8_DIR=<dir> --ignored --test-threads=1"]
fn the_sequence_ends_at_the_checkpoints_stop_id() {
    let directory = PathBuf::from(std::env::var("SENSENOVA_FP8_DIR").expect("SENSENOVA_FP8_DIR"));
    let descriptor = ModelDescriptor::sensenova_loop_9b();
    let backend =
        sparkflash_model_sensenova::SenseNovaDeviceBackend::open(&directory, descriptor, 4096)
            .expect("device backend");
    let eos = sparkflash_model_sensenova::SenseNovaConfig::from_checkpoint_dir(&directory)
        .expect("config")
        .eos_token_ids;
    assert!(!eos.is_empty(), "the checkpoint names no stop id");
    // The prompt is rendered by the same ChatML codec the server uses, so this measures the turn
    // the arm actually serves rather than a hand-assembled token list.
    let architecture = sparkflash_model::Architecture::from_checkpoint_dir(&directory)
        .expect("the checkpoint's architecture");
    let codec = sparkflash_model_qwen4_exp::QwenCodec::from_path_with_architecture(
        &directory.join("tokenizer.json"),
        &architecture,
    )
    .expect("the ChatML codec");
    let prompt = codec
        .encode_chat(&serde_json::json!({
            "messages": [{"role": "user", "content": "Say hello in one word."}],
        }))
        .expect("the prompt");
    let mut session = backend.open_session().expect("session");
    let step = session.prefill(&prompt).expect("prefill");
    let mut next = step.next_token;
    let mut steps = 0usize;
    while let Some(token) = next {
        // A stop id returned as a token is the defect this pins: the serving loop reads `None`
        // as "the model finished", so the turn would run to max_tokens instead.
        assert!(
            !eos.contains(&token),
            "a stop id was emitted as a token: {token}"
        );
        if steps >= 256 {
            panic!("the sequence did not end within 256 tokens");
        }
        next = session.decode(token).expect("decode").next_token;
        steps += 1;
    }
    assert!(steps > 0, "the model ended before answering");
    eprintln!("the sequence ended after {steps} tokens, at one of {eos:?}");
}
