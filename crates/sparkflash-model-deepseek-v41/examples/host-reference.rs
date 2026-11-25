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
//!
//! `--tokenizer FILE` hashes the Engram n-grams over that tokenizer's compressed map, as the
//! reference and the device arm do; a configuration with Engram layers is refused without one.
//! `--ids-compressed` is the opt-out: the prompts' ids are hashed as already compressed, which is
//! only right for synthetic ids.
//!
//! `--tops FILE` also writes, beside the ids, each step's `TOPS` most likely ids as
//! `[id, logprob]` pairs in descending order (a JSON list a prompt of lists a position), the
//! sidecar a near-tie comparison reads the reference's own alternatives from.
//!
//! A fourth argument, `DUMP_DIR`, also writes the first prompt's prefill boundary table there:
//! one little-endian f32 file a layer and tensor (`L<layer>-<name>.f32`) and `shapes.json`
//! naming each file's shape, for a layer-by-layer comparison with a device arm's dump.

use std::path::Path;

use sparkflash_model_deepseek_v41::config::ModelConfig;
use sparkflash_model_deepseek_v41::forward::boundary::BoundaryData;
use sparkflash_model_deepseek_v41::forward::{InferenceEngine, SequenceState};

/// The largest single tensor the reference will materialise.
const TENSOR_LIMIT: usize = 512 * 1024 * 1024;

const USAGE: &str = "usage: host-reference CHECKPOINT PROMPTS.json MAX_TOKENS [DUMP_DIR] \
                     [--tokenizer FILE | --ids-compressed] [--tops FILE]";

/// How many alternatives `--tops` records a position, as many as the served comparison asks for.
const TOPS: usize = 8;

fn main() -> Result<(), String> {
    let mut positional = Vec::new();
    let mut tokenizer = None;
    let mut ids_compressed = false;
    let mut tops_file = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokenizer" => tokenizer = Some(args.next().ok_or(USAGE)?),
            "--ids-compressed" => ids_compressed = true,
            "--tops" => tops_file = Some(args.next().ok_or(USAGE)?),
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}; {USAGE}")),
            _ => positional.push(arg),
        }
    }
    let mut positional = positional.into_iter();
    let checkpoint = positional.next().ok_or(USAGE)?;
    let prompts = positional.next().ok_or(USAGE)?;
    let max_tokens: usize = positional
        .next()
        .ok_or(USAGE)?
        .parse()
        .map_err(|error| format!("max tokens: {error}"))?;
    let dump = positional.next();
    if positional.next().is_some() {
        return Err(USAGE.to_owned());
    }
    let dir = Path::new(&checkpoint);
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|error| format!("{}: {error}", dir.join("config.json").display()))?;
    // The value table from the checkpoint's description, the route the device forward reads
    // through its plan, so the reference and the device open one table.
    let config: ModelConfig = sparkflash_model_deepseek_v41::described::host_config(&text)
        .map_err(|error| format!("config: {error}"))?;
    let source =
        sparkflash_model::layout::open_checkpoint(dir, &dir.join("model.safetensors.index.json"))
            .map_err(|error| format!("checkpoint: {error}"))?;
    let mut engine = InferenceEngine::new(config.clone(), Box::new(source), TENSOR_LIMIT)
        .map_err(|error| format!("engine: {error}"))?;
    match (tokenizer, ids_compressed) {
        (Some(_), true) => return Err("--tokenizer and --ids-compressed disagree".to_owned()),
        (Some(path), false) => {
            let map = sparkflash_engram::compress::compressed_token_map_json(&path)
                .map_err(|error| format!("{path}: {error}"))?;
            engine
                .set_engram_token_map(map)
                .map_err(|error| format!("{path}: {error}"))?;
        }
        (None, false) if !config.engram_layer_ids.is_empty() => {
            return Err(format!(
                "the configuration has Engram layers {:?}: pass --tokenizer FILE to hash over its \
                 compressed map (--ids-compressed hashes the ids as they are)",
                config.engram_layer_ids
            ));
        }
        (None, _) => {}
    }
    let prompts: Vec<Vec<u32>> = serde_json::from_str(
        &std::fs::read_to_string(&prompts).map_err(|error| format!("{prompts}: {error}"))?,
    )
    .map_err(|error| format!("{prompts}: {error}"))?;
    if let (Some(dump), Some(first)) = (dump.as_deref(), prompts.first()) {
        write_boundary(&engine, &config, first, Path::new(dump))?;
    }
    let eos = 1u32;
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
    let mut tops: Vec<Vec<Vec<(u32, f64)>>> = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        if prompt.is_empty() {
            return Err("a prompt has no tokens".to_string());
        }
        let mut state = SequenceState::new(&config);
        let mut step = engine
            .prefill(&mut state, prompt)
            .map_err(|error| format!("prefill: {error}"))?;
        let mut tokens = Vec::with_capacity(max_tokens);
        let mut alternatives = Vec::with_capacity(max_tokens);
        while tokens.len() < max_tokens {
            let Some(token) = step.next_token else { break };
            tokens.push(token);
            if tops_file.is_some() {
                alternatives.push(top_logprobs(&step.logits, token)?);
            }
            if token == eos {
                break;
            }
            step = engine
                .decode(&mut state, token)
                .map_err(|error| format!("decode: {error}"))?;
        }
        out.push(tokens);
        tops.push(alternatives);
    }
    if let Some(path) = tops_file {
        let text = serde_json::to_string(&tops).map_err(|error| format!("tops: {error}"))?;
        std::fs::write(&path, text + "\n").map_err(|error| format!("{path}: {error}"))?;
    }
    println!(
        "{}",
        serde_json::to_string(&out).map_err(|error| format!("serialise: {error}"))?
    );
    Ok(())
}

/// The `TOPS` largest log-probabilities of `logits`, descending, as `(id, logprob)`; the chosen
/// `token` must lead them, so a reference whose choice is not its own argmax is refused.
fn top_logprobs(logits: &[f32], token: u32) -> Result<Vec<(u32, f64)>, String> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let norm = logits
        .iter()
        .map(|value| (*value as f64 - max).exp())
        .sum::<f64>()
        .ln();
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]).then(a.cmp(b)));
    let top: Vec<(u32, f64)> = order
        .into_iter()
        .take(TOPS)
        .map(|id| (id, logits[id as usize] as f64 - max - norm))
        .collect();
    if top.first().map(|entry| entry.0) != Some(token) {
        return Err(format!(
            "the reference chose {token}, which does not lead its own logits ({:?})",
            top.first()
        ));
    }
    Ok(top)
}

/// The prefill boundary table of `prompt` as one f32 file a layer and tensor, and their shapes.
fn write_boundary(
    engine: &InferenceEngine,
    config: &ModelConfig,
    prompt: &[u32],
    dir: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    let mut state = SequenceState::new(config);
    let observed = engine
        .prefill_observed(&mut state, prompt)
        .map_err(|error| format!("prefill observed: {error}"))?;
    let mut shapes = serde_json::Map::new();
    for row in observed.boundary.rows() {
        for tensor in row.tensors() {
            let BoundaryData::F32(values) = tensor.data() else {
                continue;
            };
            let name = format!("L{:02}-{}.f32", row.layer(), tensor.name());
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join(&name), bytes).map_err(|error| format!("{name}: {error}"))?;
            shapes.insert(name, serde_json::json!(tensor.shape()));
        }
    }
    let text = serde_json::to_string_pretty(&shapes).map_err(|error| format!("shapes: {error}"))?;
    std::fs::write(dir.join("shapes.json"), text).map_err(|error| format!("shapes.json: {error}"))
}
