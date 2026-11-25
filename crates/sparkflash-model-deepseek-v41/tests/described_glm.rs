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

//! GLM-5.3-Flash's description on the base's host forward, at tiny widths.
//!
//! The checkpoint is written from the layout (`write_described_checkpoint`) under GLM's own
//! stored names, and the base opens it through the description alone (`described::host_config`).
//! The forward runs GLM's stated wiring: KDA blocks, absorbed MLA with the pooled indexer and a
//! shared selection, the dense prefix and FP8-tile experts, each sublayer's own-mix collapse and
//! the mean head fold, all on the producer's BF16 type map.
//!
//! The pinned ids are the greedy tokens of every position on the type map. Before the GLM crate
//! was deleted its host reference (f32 streams) ran the same checkpoint: with the BF16 rounds and
//! the dynamic FP8 activation quantiser switched off in both, every logit of every position
//! agreed within 3e-6. With the quantiser on, an E4M3 code can flip on a 1e-7 input difference,
//! and on the type map 18 of the 20 positions keep the crate's argmax.

use std::path::PathBuf;

use sparkflash_model_deepseek_v41::checkpoint::write_described_checkpoint;
use sparkflash_model_deepseek_v41::described::host_config;
use sparkflash_model_deepseek_v41::forward::{InferenceEngine, SequenceState};

const CONFIG: &str = include_str!("fixtures/glm-tiny/config.json");

/// Twenty tokens: five complete pools of four, past the tiny budget of two pools, so the
/// ranking and the tail both decide what a query reads.
const TOKENS: [u32; 20] = [
    3, 17, 42, 5, 63, 0, 28, 9, 11, 200, 7, 33, 21, 60, 14, 90, 2, 150, 77, 8,
];

/// Every position's greedy token.
const PINNED: [usize; 20] = [
    505, 499, 288, 219, 201, 311, 461, 326, 90, 461, 66, 340, 110, 346, 169, 398, 332, 56, 284, 326,
];

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn engine() -> (Scratch, InferenceEngine) {
    let directory = std::env::temp_dir().join(format!("glm-tiny-{}", std::process::id()));
    let scratch = Scratch(directory.clone());
    std::fs::create_dir_all(&directory).expect("a scratch directory");
    std::fs::write(directory.join("config.json"), CONFIG).expect("the config writes");
    let architecture = sparkflash_model::Architecture::from_json(CONFIG).expect("GLM parses");
    write_described_checkpoint(&directory, &architecture, 7, 2).expect("the checkpoint writes");
    let config = host_config(CONFIG).expect("the host runs GLM's description");
    let source = sparkflash_model::layout::open_checkpoint(
        &directory,
        &directory.join("model.safetensors.index.json"),
    )
    .expect("the checkpoint opens");
    let engine = InferenceEngine::new(
        config,
        Box::new(source),
        sparkflash_weights::linalg::REFERENCE_TENSOR_LIMIT,
    )
    .expect("the engine opens");
    (scratch, engine)
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|a, b| row[*a].total_cmp(&row[*b]))
        .expect("a row")
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
fn glm_runs_from_its_description_whole_chunked_and_by_token_alike() {
    let (_scratch, engine) = engine();
    let config = engine.config().clone();
    let positions: Vec<usize> = (0..TOKENS.len()).collect();

    let mut state = SequenceState::new(&config);
    let whole = engine
        .prefill_with_position_logits(&mut state, &TOKENS, &positions)
        .expect("the whole prompt runs")
        .rows;
    assert!(whole.iter().flatten().all(|value| value.is_finite()));
    let ids: Vec<usize> = whole.iter().map(|row| argmax(row)).collect();
    assert_eq!(ids, PINNED);

    let mut by_token = SequenceState::new(&config);
    let tokens: Vec<Vec<f32>> = TOKENS
        .iter()
        .map(|token| {
            engine
                .prefill(&mut by_token, &[*token])
                .expect("a token runs")
                .logits
        })
        .collect();
    assert_eq!(bits(&tokens), bits(&whole), "token by token");

    let mut chunked = SequenceState::new(&config);
    // The split falls inside a pool, so the second chunk completes a pool the first began.
    let mut rows = engine
        .prefill_with_position_logits(&mut chunked, &TOKENS[..7], &positions[..7])
        .expect("the first chunk runs")
        .rows;
    rows.extend(
        engine
            .prefill_with_position_logits(&mut chunked, &TOKENS[7..], &positions[..13])
            .expect("the second chunk runs")
            .rows,
    );
    assert_eq!(bits(&rows), bits(&whole), "two chunks");
}

/// The description's wiring reaches the value table: the own-mix collapse, the mean fold, the
/// BF16 stream, the activation tiles, the pool and its tail, the absorbed latent and the key
/// norm's epsilon.
#[test]
fn glm_states_its_wiring() {
    let config = host_config(CONFIG).expect("the host runs GLM's description");
    let wiring = config.stated();
    assert!(wiring.own_mix && wiring.mean_fold && wiring.bf16_stream);
    assert_eq!((wiring.activation_tile, wiring.index_pool), (128, 4));
    assert!(wiring.index_tail);
    assert!(config.absorbed_latent.is_some());
    assert_eq!(config.index_key_norm_eps, 1e-6);
}
