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

//! Writes a synthetic checkpoint from a configuration document.
//!
//! `write-synthetic-checkpoint CONFIG.json OUT_DIR [SEED] [SHARDS]` writes every tensor the
//! DeepSeek text plan names for that configuration, in the released storage formats.

use std::path::Path;

use sparkflash_model_deepseek_v41::checkpoint::write_synthetic_checkpoint;
use sparkflash_model_deepseek_v41::config::ModelConfig;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let config = args
        .next()
        .ok_or("usage: write-synthetic-checkpoint CONFIG.json OUT_DIR [SEED] [SHARDS]")?;
    let out = args
        .next()
        .ok_or("usage: write-synthetic-checkpoint CONFIG.json OUT_DIR [SEED] [SHARDS]")?;
    let seed: u64 = args
        .next()
        .map_or(Ok(7), |v| v.parse())
        .map_err(|error| format!("seed: {error}"))?;
    let shards: usize = args
        .next()
        .map_or(Ok(4), |v| v.parse())
        .map_err(|error| format!("shards: {error}"))?;
    let config_path = Path::new(&config).to_path_buf();
    let text =
        std::fs::read_to_string(&config_path).map_err(|error| format!("{config}: {error}"))?;
    let config = ModelConfig::from_json(&text).map_err(|error| format!("config: {error}"))?;
    std::fs::create_dir_all(&out).map_err(|error| format!("{out}: {error}"))?;
    // The writer does not write config.json (the checkpoint's own document); copy it beside the
    // weights so the loader reads the same document the tensors were planned from.
    std::fs::copy(&config_path, Path::new(&out).join("config.json"))
        .map_err(|error| format!("config.json: {error}"))?;
    write_synthetic_checkpoint(Path::new(&out), &config, seed, shards)?;
    println!("{out}: synthetic checkpoint written");
    Ok(())
}
