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

//! The shared host reference's logits for a checkpoint, as the producer capture reads them.
//!
//! `dump-host-logits [--boundaries DIR] CHECKPOINT_DIR OUT.json [TOKEN ...]` opens the
//! checkpoint through its description (the same open the forward uses), runs the default prompt
//! when no tokens are given, and writes every position's f32 logit row and its argmax. With
//! `--boundaries DIR` it also writes every layer's boundary tensors (`L<n>-<name>.f32` and
//! `shapes.json`) so a miss against the producer can be localised to a layer and sublayer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sparkflash_model_deepseek_v41::described::host_config;
use sparkflash_model_deepseek_v41::forward::{InferenceEngine, SequenceState};

/// Twenty tokens: five complete pools of four, past the tiny budget of two pools, so the ranking
/// and the tail both decide what a query reads (the described-GLM host test's prompt).
const TOKENS: [u32; 20] = [
    3, 17, 42, 5, 63, 0, 28, 9, 11, 200, 7, 33, 21, 60, 14, 90, 2, 150, 77, 8,
];

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1).peekable();
    let mut boundaries_dir: Option<PathBuf> = None;
    while let Some(flag) = args.peek() {
        if flag == "--boundaries" {
            args.next();
            let dir = args
                .next()
                .ok_or("--boundaries needs a directory path")?;
            boundaries_dir = Some(PathBuf::from(dir));
        } else {
            break;
        }
    }
    let checkpoint = args
        .next()
        .ok_or("usage: dump-host-logits [--boundaries DIR] CHECKPOINT_DIR OUT.json [TOKEN ...]")?;
    let out = args
        .next()
        .ok_or("usage: dump-host-logits [--boundaries DIR] CHECKPOINT_DIR OUT.json [TOKEN ...]")?;
    let tokens: Vec<u32> = args
        .map(|token| token.parse().map_err(|error| format!("token {token}: {error}")))
        .collect::<Result<_, _>>()?;
    let tokens = if tokens.is_empty() {
        TOKENS.to_vec()
    } else {
        tokens
    };
    let directory = Path::new(&checkpoint);
    let text = std::fs::read_to_string(directory.join("config.json"))
        .map_err(|error| format!("{checkpoint}/config.json: {error}"))?;
    let config = host_config(&text).map_err(|error| format!("the description: {error}"))?;
    let source = sparkflash_model::layout::open_checkpoint(
        directory,
        &directory.join("model.safetensors.index.json"),
    )
    .map_err(|error| format!("the checkpoint: {error}"))?;
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(source),
        sparkflash_weights::linalg::REFERENCE_TENSOR_LIMIT,
    )
    .map_err(|error| format!("the engine: {error}"))?;
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let mut state = SequenceState::new(&config);
    let rows = engine
        .prefill_with_position_logits(&mut state, &tokens, &positions)
        .map_err(|error| format!("the prompt: {error}"))?
        .rows;
    if let Some(dir) = boundaries_dir {
        std::fs::create_dir_all(&dir)
            .map_err(|error| format!("create {}: {error}", dir.display()))?;
        let mut observed_state = SequenceState::new(&config);
        let observed = engine
            .prefill_observed(&mut observed_state, &tokens)
            .map_err(|error| format!("observed prompt: {error}"))?;
        let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for row in observed.boundary.rows() {
            for tensor in row.tensors() {
                if let Some(values) = tensor.data().as_f32() {
                    let file = format!("L{}-{}.f32", row.layer(), tensor.name());
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(dir.join(&file), &bytes)
                        .map_err(|error| format!("write {file}: {error}"))?;
                    shapes.insert(file, tensor.shape().to_vec());
                }
            }
        }
        std::fs::write(
            dir.join("shapes.json"),
            serde_json::to_vec_pretty(&shapes).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write shapes.json: {error}"))?;
    }
    let ids: Vec<usize> = rows
        .iter()
        .map(|row| {
            (0..row.len())
                .max_by(|a, b| row[*a].total_cmp(&row[*b]))
                .expect("a logit row")
        })
        .collect();
    let body = serde_json::json!({
        "schema": "sparkflash-host-logits/1",
        "tokens": tokens,
        "vocab": rows.first().map_or(0, Vec::len),
        "ids": ids,
        "logits": rows,
    });
    std::fs::write(&out, serde_json::to_vec(&body).map_err(|error| error.to_string())?)
        .map_err(|error| format!("{out}: {error}"))?;
    println!("{out}: {} positions, {} values a row", rows.len(), body["vocab"]);
    Ok(())
}
