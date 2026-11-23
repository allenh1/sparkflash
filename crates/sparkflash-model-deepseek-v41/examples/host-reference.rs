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

//! The host reference forward over a checkpoint, as greedy token ids.
//!
//! `host-reference CHECKPOINT PROMPTS.json MAX_TOKENS` reads a JSON array of token-id arrays and
//! prints a JSON array of the ids the f32 reference forward selects from each prompt, one per
//! step, until the cap or the end-of-sequence token. It is the arm the four-rank device group is
//! compared against; it shares the loader, the model config and the token list, so a difference
//! is arithmetic, not a prompt.

use std::path::Path;

use sparkflash_model_deepseek_v41::config::ModelConfig;
use sparkflash_model_deepseek_v41::forward::{InferenceEngine, SequenceState};
use sparkflash_weights::safetensors::SafeTensorsSource;

/// The largest single tensor the reference will materialise.
const TENSOR_LIMIT: usize = 512 * 1024 * 1024;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let checkpoint = args
        .next()
        .ok_or("usage: host-reference CHECKPOINT PROMPTS.json MAX_TOKENS")?;
    let prompts = args
        .next()
        .ok_or("usage: host-reference CHECKPOINT PROMPTS.json MAX_TOKENS")?;
    let max_tokens: usize = args
        .next()
        .ok_or("usage: host-reference CHECKPOINT PROMPTS.json MAX_TOKENS")?
        .parse()
        .map_err(|error| format!("max tokens: {error}"))?;
    let dir = Path::new(&checkpoint);
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|error| format!("{}: {error}", dir.join("config.json").display()))?;
    let config = ModelConfig::from_json(&text).map_err(|error| format!("config: {error}"))?;
    let source = SafeTensorsSource::open(dir, &dir.join("model.safetensors.index.json"))
        .map_err(|error| format!("checkpoint: {error}"))?;
    let engine = InferenceEngine::new(config.clone(), Box::new(source), TENSOR_LIMIT)
        .map_err(|error| format!("engine: {error}"))?;
    let prompts: Vec<Vec<u32>> = serde_json::from_str(
        &std::fs::read_to_string(&prompts).map_err(|error| format!("{prompts}: {error}"))?,
    )
    .map_err(|error| format!("{prompts}: {error}"))?;
    let eos = 1u32;
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        if prompt.is_empty() {
            return Err("a prompt has no tokens".to_string());
        }
        let mut state = SequenceState::new(&config);
        let mut step = engine
            .prefill(&mut state, prompt)
            .map_err(|error| format!("prefill: {error}"))?;
        let mut tokens = Vec::with_capacity(max_tokens);
        while tokens.len() < max_tokens {
            let Some(token) = step.next_token else { break };
            tokens.push(token);
            if token == eos {
                break;
            }
            step = engine
                .decode(&mut state, token)
                .map_err(|error| format!("decode: {error}"))?;
        }
        out.push(tokens);
    }
    println!(
        "{}",
        serde_json::to_string(&out).map_err(|error| format!("serialise: {error}"))?
    );
    Ok(())
}
