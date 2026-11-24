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

//! The DeepSeek plan against the milestone-2 part A baseline manifest.
//!
//! The baseline was recorded from the real checkpoint on four ranks. This test builds the plan
//! from that checkpoint's `config.json` and the layout, at the same world and rank, and requires
//! the plan to reproduce every model value, every device-bound tensor (name, shape, format, rank
//! slice and bytes) and the exact kernel symbol set. The fixture is the gzipped rank-0 manifest
//! from the baseline run; see `docs/engine/m2a-baseline-20261122.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::PathBuf;

use flate2::read::GzDecoder;
use sparkflash_model::{FactValue, Part, Target};
use sparkflash_ops::engine_plan::EnginePlan;

const BASELINE: &[u8] = include_bytes!("data/deepseek-v41-manifest-r0.jsonl.gz");
const SNAPSHOT_REPO: &str = "models--deepseek-ai--DeepSeek-V4.1-Flash-1017";
const SNAPSHOT_REVISION: &str = "1bae0e1088993b9b82de08f59de5be5a34e42170";

fn checkpoint() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SPARKFLASH_CHECKPOINT") {
        let path = PathBuf::from(path);
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home)
        .join(".cache/sparkflash/acceptance/hub")
        .join(SNAPSHOT_REPO)
        .join("snapshots")
        .join(SNAPSHOT_REVISION);
    path.join("config.json").is_file().then_some(path)
}

fn baseline() -> Vec<serde_json::Value> {
    let mut text = String::new();
    GzDecoder::new(BASELINE)
        .read_to_string(&mut text)
        .expect("the baseline manifest decompresses");
    text.lines()
        .map(|line| serde_json::from_str(line).expect("a baseline manifest line"))
        .collect()
}

fn fact_json(value: &FactValue) -> serde_json::Value {
    match value {
        FactValue::Bool(value) => serde_json::json!(value),
        FactValue::Integer(value) => serde_json::json!(value),
        FactValue::Float(value) => serde_json::json!(value),
        FactValue::Text(value) => serde_json::json!(value),
        FactValue::Integers(value) => serde_json::json!(value),
        FactValue::Null => serde_json::Value::Null,
    }
}

fn plan() -> EnginePlan {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: the released DeepSeek checkpoint is not on this host");
        return EnginePlan::from_json(
            include_str!(
                "../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
            ),
            Target::Sm121,
            4,
            0,
        )
        .expect("the released configuration plans");
    };
    let config = std::fs::read_to_string(dir.join("config.json")).expect("the checkpoint config");
    EnginePlan::from_json(&config, Target::Sm121, 4, 0).expect("the checkpoint plans at TP4")
}

#[test]
fn the_plan_reproduces_the_deepseek_baseline_manifest() {
    let records = baseline();
    let plan = plan();
    assert_eq!(plan.tp().world(), 4);
    assert_eq!(plan.tp().rank(), 0);

    // Every model value the baseline recorded is a fact on the plan, at the same value.
    let mut baseline_model = BTreeMap::new();
    for record in &records {
        if record["kind"] == "model" {
            baseline_model.insert(
                record["name"].as_str().expect("a model name").to_owned(),
                record["value"].clone(),
            );
        }
    }
    let plan_model: BTreeMap<String, serde_json::Value> = plan
        .model_facts()
        .iter()
        .map(|(name, value)| (name.clone(), fact_json(value)))
        .collect();
    assert_eq!(
        plan_model, baseline_model,
        "the plan's model facts differ from the baseline manifest"
    );

    // Every device-bound tensor is claimed once, with the baseline's stored shape, format,
    // rank slice and byte count.
    let mut baseline_tensors = BTreeMap::new();
    for record in &records {
        if record["kind"] == "tensor" {
            baseline_tensors.insert(
                record["name"].as_str().expect("a tensor name").to_owned(),
                (
                    record["shape"].clone(),
                    record["format"].as_str().expect("a format").to_owned(),
                    record["rank_slice"].clone(),
                    record["shard_bytes"].as_u64().expect("bytes"),
                ),
            );
        }
    }
    let plan_tensors: BTreeMap<String, (serde_json::Value, String, serde_json::Value, u64)> = plan
        .tensors()
        .iter()
        .filter(|tensor| tensor.placement == "resident")
        .map(|tensor| {
            (
                tensor.name.clone(),
                (
                    serde_json::json!(tensor.shape),
                    tensor.format.clone(),
                    serde_json::json!(tensor.rank_shape),
                    tensor.shard_bytes as u64,
                ),
            )
        })
        .collect();
    assert_eq!(
        plan_tensors.len(),
        baseline_tensors.len(),
        "the plan's resident tensor count differs from the baseline"
    );
    for (name, expected) in &baseline_tensors {
        let actual = plan_tensors
            .get(name)
            .unwrap_or_else(|| panic!("the plan does not claim {name}"));
        assert_eq!(actual, expected, "tensor {name}");
    }

    // The catalogue's DeepSeek entries are exactly the symbols the baseline loaded, and every
    // symbol is grouped under at least one part.
    let mut baseline_symbols = BTreeSet::new();
    for record in &records {
        if record["kind"] == "kernel" {
            baseline_symbols.insert(record["symbol"].as_str().expect("a symbol").to_owned());
        }
    }
    let mut plan_symbols = BTreeSet::new();
    for part in plan.symbols() {
        assert!(
            !part.symbols.is_empty(),
            "part {:?} has an empty DeepSeek catalogue entry",
            part.part
        );
        let mut in_part = BTreeSet::new();
        for symbol in &part.symbols {
            assert!(
                in_part.insert(*symbol),
                "symbol {symbol} is listed twice in one part's entry"
            );
            plan_symbols.insert((*symbol).to_owned());
        }
    }
    // Symbols the catalogue claims beyond the baseline's loaded set. The baseline recorded what
    // the four-rank run launched; these are the fallback artifacts the forward can reach when a
    // preferred body is absent, so the catalogue must carry them even though that run did not
    // load them. Every addition is named here with the part that claims it.
    let additions: BTreeSet<&str> = [
        ("b12x_mxfp4_moe_sm121", Part::Moe),
        ("b12x_mxfp8_dense_sm121", Part::DenseGemm),
        ("b12x_mxfp8_wo_a_sm121", Part::DenseGemm),
        ("flash_mla_csa2_sm121", Part::MlaAttention),
        ("lm_head_reduce_sm121", Part::LmHead),
        // The row-scaled MXFP8 expert body the plan chooses for a checkpoint that stores its
        // routed experts one ue8m0 scale a row every 32 columns (F4).
        ("opt_mma_fp8_rowscale_f32o", Part::Moe),
    ]
    .into_iter()
    .map(|(symbol, _part)| symbol)
    .collect();
    let missing: Vec<&String> = baseline_symbols.difference(&plan_symbols).collect();
    let extra: Vec<&String> = plan_symbols
        .difference(&baseline_symbols)
        .filter(|symbol| !additions.contains(symbol.as_str()))
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "the catalogue's DeepSeek symbols differ from the baseline manifest: missing {missing:?}, extra {extra:?}"
    );
    for (symbol, part) in [
        ("b12x_mxfp4_moe_sm121", Part::Moe),
        ("b12x_mxfp8_dense_sm121", Part::DenseGemm),
        ("b12x_mxfp8_wo_a_sm121", Part::DenseGemm),
        ("flash_mla_csa2_sm121", Part::MlaAttention),
        ("lm_head_reduce_sm121", Part::LmHead),
        ("opt_mma_fp8_rowscale_f32o", Part::Moe),
    ] {
        assert!(
            baseline_symbols.contains(symbol) || plan_symbols.contains(symbol),
            "the addition {symbol} is not claimed"
        );
        assert!(
            plan.symbols()
                .iter()
                .any(|entry| entry.part == part && entry.symbols.contains(&symbol)),
            "{symbol} is not claimed under {part:?}"
        );
    }
}

#[test]
fn the_layer_facts_sit_on_that_layers_parts() {
    let plan = plan();
    let facts = |layer: usize| {
        plan.stacks()
            .iter()
            .find(|stack| stack.kind == sparkflash_model::StackKind::Language)
            .and_then(|stack| stack.blocks.iter().find(|block| block.index == layer))
            .and_then(|block| block.parts.first())
            .map(|part| &part.facts)
            .unwrap_or_else(|| panic!("layer {layer} has a block"))
    };
    assert_eq!(facts(0).get("compress_ratio"), Some(&FactValue::Integer(0)));
    assert_eq!(
        facts(0).get("encoder_side"),
        Some(&FactValue::Text("encoder".to_owned()))
    );
    assert_eq!(facts(0).get("kv_source"), Some(&FactValue::Bool(false)));
    assert_eq!(
        facts(0).get("tap_layers"),
        Some(&FactValue::Integers(vec![37, 38, 39]))
    );
    assert_eq!(
        facts(20).get("compress_ratio"),
        Some(&FactValue::Integer(1))
    );
    assert_eq!(
        facts(20).get("candidate_source"),
        Some(&FactValue::Bool(true))
    );
    assert_eq!(facts(20).get("kv_source"), Some(&FactValue::Bool(true)));
    assert_eq!(
        facts(20).get("encoder_side"),
        Some(&FactValue::Text("encoder".to_owned()))
    );
    assert_eq!(
        facts(21).get("encoder_side"),
        Some(&FactValue::Text("decoder".to_owned()))
    );
    assert_eq!(facts(14).get("engram_member"), Some(&FactValue::Bool(true)));
    assert_eq!(
        facts(14).get("engram_rows"),
        Some(&FactValue::Integer(384016682))
    );
    assert_eq!(
        facts(1).get("engram_rows"),
        Some(&FactValue::Integer(384006168))
    );
    assert_eq!(
        plan.symbols()
            .iter()
            .filter(|entry| entry.part == Part::Moe)
            .count(),
        1
    );
}
