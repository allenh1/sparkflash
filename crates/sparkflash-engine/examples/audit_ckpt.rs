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

//! task-29: read-only audit of the released checkpoint through the E2 readers.
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::loader::LoadedWeights;
use sparkflash_engine::safetensors::CheckpointIndex;
use sparkflash_engine::tensor::TensorPlan;
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: audit_ckpt <snapshot dir>"),
    );
    let index_path = dir.join("model.safetensors.index.json");
    let index = CheckpointIndex::load(&index_path).expect("index parses");
    println!(
        "INDEX names={} total_size={:?} shards={}",
        index.len(),
        index.total_size(),
        index.shards().len()
    );

    let config = ModelConfig::v41_flash().expect("released config");
    let plan = TensorPlan::language_model(&config);
    match LoadedWeights::audit(&dir, &index_path, &plan) {
        Ok(v) => {
            println!(
                "AUDIT {} conformant={} exact={}",
                v.summary(),
                v.is_conformant(),
                v.is_exact()
            );
            for m in v.mismatched.iter().take(10) {
                println!("  MISMATCH {m}");
            }
            println!(
                "  unknown={} first={:?}",
                v.unknown.len(),
                &v.unknown[..v.unknown.len().min(6)]
            );
            println!(
                "  missing={} first={:?}",
                v.missing.len(),
                &v.missing[..v.missing.len().min(6)]
            );
        }
        Err(error) => println!("AUDIT ERROR {error}"),
    }

    let weights = LoadedWeights::open(&dir, &index_path, &plan).expect("shards open");
    println!(
        "OPEN ok limit={} refused={}",
        weights.limit(),
        weights.refused_names().len()
    );
    println!("  refused list: {:?}", weights.refused_names());
    for name in [
        "layers.0.attn_norm.weight",
        "layers.0.attn.wo_a.weight",
        "layers.0.attn.wq_a.weight",
        "layers.1.engram.q_weight",
        "layers.1.engram.k_weight",
        "layers.1.engram.wkv.weight",
        "embed.weight",
        "head.weight",
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
    ] {
        match weights.get(name) {
            Ok(t) => println!(
                "GET {name} ok dtype={} shape={:?} elements={} bytes={}",
                t.dtype.name(),
                t.shape,
                t.elements(),
                t.data.len()
            ),
            Err(error) => println!("GET {name} ERR {error}"),
        }
    }
}
