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

//! The tiny seeded checkpoints the pack's tests run: the crate's own and its integration tests'.

use std::path::Path;

use sparkflash_weights::quant::f32_to_bf16_bits;
use sparkflash_weights::rng::TensorRng;
use sparkflash_weights::safetensors::write_checkpoint;

use crate::config::DoryConfig;
use crate::tensor::DoryTensorPlan;

/// Writes every tensor `config` serves into `directory`, BF16, each seeded by its name.
///
/// A per-vocabulary logit scale of 0.02 keeps a tiny model's logits finite.
///
/// # Errors
/// The checkpoint writer's.
pub fn write_tensors(directory: &Path, config: &DoryConfig) -> Result<(), String> {
    let plan = DoryTensorPlan::served(config);
    let mut tensors = Vec::with_capacity(plan.len());
    for spec in plan.specs() {
        let mut rng = TensorRng::new(7, &spec.name);
        let elements = spec.elements() as usize;
        let values: Vec<f32> = if spec.name == "lm_head.logit_scale" {
            vec![0.02; elements]
        } else {
            (0..elements).map(|_| rng.signed() * 0.05).collect()
        };
        let bytes = values
            .iter()
            .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
            .collect();
        tensors.push((spec.clone(), bytes));
    }
    write_checkpoint(directory, &tensors, 1).map_err(|error| error.to_string())
}

/// Writes the checkpoint for the config document `json` into `directory`: its tensors
/// ([write_tensors]) and `config.json`.
///
/// # Errors
/// An invalid document, or the writers'.
pub fn write(directory: &Path, json: &str) -> Result<(), String> {
    let config = DoryConfig::from_json(json).map_err(|error| error.to_string())?;
    write_tensors(directory, &config)?;
    std::fs::write(directory.join("config.json"), json).map_err(|error| error.to_string())
}
