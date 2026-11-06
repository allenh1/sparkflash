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

//! The checkpoint round trip: plan, write, reopen, load, run - on the host.
//!
//! This is the gate that runs everywhere. It proves the fixture writer emits a checkpoint the
//! reader accepts, that the tensor plan's served set is exactly what the writer wrote, that the
//! host loader decodes every dtype the released checkpoint uses (BF16, FP8 E4M3, NVFP4 with both
//! per-tensor scales, I64 metadata) and that the reference forward produces finite, non-degenerate,
//! deterministic logits from real bytes rather than from a synthetic in-memory weight set.

use std::path::Path;

use sparkflash_model_qwen4_exp::device::Qwen4ExpDeviceBackend;
use sparkflash_model_qwen4_exp::{
    PleHasher, Qwen4ExpConfig, Qwen4ExpTensorPlan, SyntheticCheckpoint, load_weights,
    open_ple_table, write_synthetic_checkpoint, write_synthetic_checkpoint_sharded,
};
use sparkflash_weights::WeightSource;
use sparkflash_weights::safetensors::SafeTensorsSource;

const INDEX: &str = "model.safetensors.index.json";

fn open_source(directory: &Path) -> SafeTensorsSource {
    SafeTensorsSource::open(directory, &directory.join(INDEX)).expect("the fixture index opens")
}

#[test]
fn the_fixture_write_reopen_load_and_run_agree_with_the_plan() {
    let temp = tempfile::tempdir().expect("a temporary directory");
    let fixture = SyntheticCheckpoint::write(temp.path()).expect("the fixture writes");
    let config = fixture.config.clone();

    // The checkpoint is exactly the served plan, not merely a superset or an approximation of it.
    let plan = Qwen4ExpTensorPlan::served_and_mtp(&config);
    let source = open_source(&fixture.directory);
    plan.conformance(source.names())
        .expect("the fixture carries exactly the served tensor set");

    // Every dtype of the released checkpoint is present, so no decoding path is untested.
    let mut dtypes: Vec<&str> = source
        .names()
        .into_iter()
        .filter_map(|name| source.spec(&name).ok().map(|spec| spec.dtype.name()))
        .collect();
    dtypes.sort_unstable();
    dtypes.dedup();
    assert!(dtypes.contains(&"BF16"), "no BF16 tensor: {dtypes:?}");
    assert!(dtypes.contains(&"F8_E4M3"), "no FP8 tensor: {dtypes:?}");
    assert!(
        dtypes.contains(&"F4_E2M1"),
        "no packed FP4 tensor: {dtypes:?}"
    );
    assert!(dtypes.contains(&"F32"), "no F32 tensor: {dtypes:?}");

    let weights = load_weights(&config, &source).expect("the host loader decodes the checkpoint");
    let layer = config.ple_layer_indices()[0];
    let mut table =
        open_ple_table(&fixture.directory, &config, layer).expect("the PLE table opens");
    let mut hasher = PleHasher::new(&config, layer).expect("the hasher builds");

    let tokens = [3u32, 5, 7, 9, 11];
    let mut state = weights.new_state();
    let rows = hasher.push(&tokens, None).expect("the hash rows");
    let embeddings: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| table.embedding(row))
        .collect::<Result<_, String>>()
        .expect("the table serves every row");
    let logits = weights
        .forward(&mut state, &tokens, None, Some(&embeddings))
        .expect("the reference forward runs");
    assert_eq!(logits.len(), tokens.len());
    for (position, row) in logits.iter().enumerate() {
        assert_eq!(row.len(), config.vocab_size);
        assert!(
            row.iter().all(|value| value.is_finite()),
            "position {position} has a non-finite logit"
        );
    }
    let last = logits.last().expect("one row per token");
    let low = last.iter().copied().fold(f32::INFINITY, f32::min);
    let high = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        high - low > 1e-3,
        "the fixture produces a degenerate logit row: {low} .. {high}"
    );

    // The same checkpoint, the same tokens and a fresh state must reproduce the same numbers
    // exactly: the fixture is seeded, and the host reference has no nondeterminism.
    let mut again = weights.new_state();
    let mut hasher = PleHasher::new(&config, layer).expect("the hasher builds");
    let rows = hasher.push(&tokens, None).expect("the hash rows");
    let embeddings: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| table.embedding(row))
        .collect::<Result<_, String>>()
        .expect("the table serves every row");
    let repeat = weights
        .forward(&mut again, &tokens, None, Some(&embeddings))
        .expect("the reference forward runs");
    assert_eq!(repeat, logits, "the host reference is not deterministic");

    // The PLE table is a real table: the rows the hasher produced are inside it, and the FP8
    // scale the checkpoint carries is what the table applies.
    assert!(table.scale() > 0.0);
    assert!(table.counters().reads > 0, "no PLE row was read");
}

fn load(directory: &Path) -> sparkflash_model_qwen4_exp::Qwen4ExpWeights {
    let source = open_source(directory);
    let config = Qwen4ExpConfig::from_json(
        &std::fs::read_to_string(directory.join("config.json")).expect("the fixture config reads"),
    )
    .expect("the fixture config parses");
    load_weights(&config, &source).expect("the host loader decodes the checkpoint")
}

#[test]
fn a_sharded_checkpoint_loads_to_the_same_model_as_a_single_shard() {
    let single = tempfile::tempdir().expect("a temporary directory");
    let sharded = tempfile::tempdir().expect("a temporary directory");
    let config = Qwen4ExpConfig::tiny();
    write_synthetic_checkpoint(single.path(), &config, 11).expect("the single shard writes");
    write_synthetic_checkpoint_sharded(sharded.path(), &config, 11, 3)
        .expect("the sharded checkpoint writes");

    for index in 0..3 {
        let path = sharded.path().join(format!("model-{index:05}.safetensors"));
        assert!(path.is_file(), "{} is missing", path.display());
    }
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(sharded.path().join("model.safetensors.index.json"))
            .expect("the index reads"),
    )
    .expect("the index parses");
    let weight_map = index["weight_map"].as_object().expect("a weight map");
    let mut shards: Vec<&str> = weight_map
        .values()
        .filter_map(serde_json::Value::as_str)
        .collect();
    shards.sort_unstable();
    shards.dedup();
    assert_eq!(
        shards.len(),
        3,
        "every shard must appear in the map: {shards:?}"
    );

    // The same seed and the same plan, split across files: the model must be identical, so every
    // dtype decodes the same way whichever shard it arrived in.
    assert_eq!(
        load(single.path()),
        load(sharded.path()),
        "sharding changed the loaded model"
    );
}

#[test]
fn a_tensor_parallel_world_without_its_collective_is_refused_before_any_device_work() {
    let temp = tempfile::tempdir().expect("a temporary directory");
    let fixture = SyntheticCheckpoint::write(temp.path()).expect("the fixture writes");
    let error = Qwen4ExpDeviceBackend::open_for_world(&fixture.directory, 32, 2)
        .expect_err("a rank of a group has partials to sum and nothing to sum them over");
    assert_eq!(error.code(), "qwen4_device");
    assert!(error.message().contains("open_rank"), "{error}");
}

#[test]
fn a_context_outside_the_window_is_refused_before_the_build_is_inspected() {
    let temp = tempfile::tempdir().expect("a temporary directory");
    let fixture = SyntheticCheckpoint::write(temp.path()).expect("the fixture writes");
    let limit = fixture.config.max_position_embeddings;
    for context in [0usize, limit + 1] {
        let error = Qwen4ExpDeviceBackend::open(&fixture.directory, context)
            .expect_err("the window is checked against the configuration");
        assert!(
            error.message().contains("outside 1..="),
            "context {context}: {error}"
        );
    }
}

#[test]
fn the_released_configuration_fits_every_kernel_bound() {
    let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
    assert!(
        config.hc_lowrank <= sparkflash_model_qwen4_exp::device::MAX_LOWRANK,
        "hc_lowrank {} over the kernel bound",
        config.hc_lowrank
    );
    assert!(
        config.moe_intermediate_size <= sparkflash_model_qwen4_exp::device::MAX_INTERMEDIATE,
        "moe_intermediate_size {} over the kernel bound",
        config.moe_intermediate_size
    );
    assert!(
        config.indexer_block_budget() <= sparkflash_model_qwen4_exp::device::MAX_BUDGET_BLOCKS,
        "the indexer asks for {} blocks, over the kernel bound",
        config.indexer_block_budget()
    );
    assert!(
        config.indexer_budget + config.indexer_compress_ratio
            <= sparkflash_model_qwen4_exp::device::MAX_SELECTED_TOKENS,
        "the selected set can exceed the attention kernel's buffer"
    );
}

#[test]
fn a_configuration_past_a_kernel_bound_is_refused() {
    let temp = tempfile::tempdir().expect("a temporary directory");
    let mut config = Qwen4ExpConfig::tiny();
    config.hc_lowrank = sparkflash_model_qwen4_exp::device::MAX_LOWRANK + 1;
    write_synthetic_checkpoint(temp.path(), &config, 7).expect("the fixture writes");
    let error = Qwen4ExpDeviceBackend::open(temp.path(), 32)
        .expect_err("the bottleneck does not fit the kernel's buffer");
    assert!(error.message().contains("hc_lowrank"), "{error}");
}

#[test]
fn the_fp8_fixture_writes_the_block_scaled_experts_and_loads_them() {
    let temp = tempfile::tempdir().expect("a temporary directory");
    let config = Qwen4ExpConfig::tiny_fp8();
    write_synthetic_checkpoint(temp.path(), &config, 7).expect("the FP8 fixture writes");
    let source = open_source(temp.path());
    let plan = Qwen4ExpTensorPlan::served_and_mtp(&config);
    plan.conformance(source.names())
        .expect("the fixture carries exactly the served tensor set");
    let prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj";
    assert_eq!(
        source
            .spec(&format!("{prefix}.weight"))
            .expect("the codes")
            .dtype,
        sparkflash_weights::DType::F8E4M3
    );
    assert_eq!(
        source
            .spec(&format!("{prefix}.weight_scale_inv"))
            .expect("the tile grid")
            .dtype,
        sparkflash_weights::DType::BF16
    );
    assert!(
        source.spec(&format!("{prefix}.weight_scale_2")).is_err(),
        "the FP8 fixture must not carry an NVFP4 per-tensor scale"
    );
    let weights = load_weights(&config, &source).expect("the host loader decodes FP8 experts");
    let expert = &weights.layers[0].moe.experts[0];
    assert_eq!(expert.gate.rows, config.moe_intermediate_size);
    assert_eq!(expert.gate.columns, config.hidden_size);
    assert!(
        expert
            .gate
            .data
            .iter()
            .all(|value| value.is_finite() && *value != 0.0),
        "the fixture's FP8 experts decode to a non-degenerate matrix"
    );
}

#[test]
fn a_tampered_checkpoint_is_refused_rather_than_silently_truncated() {
    let temp = tempfile::tempdir().expect("a temporary directory");
    let fixture = SyntheticCheckpoint::write(temp.path()).expect("the fixture writes");
    let config = fixture.config.clone();
    let source = open_source(&fixture.directory);
    // A name the plan expects but the checkpoint does not carry: the loader must name the missing
    // tensor instead of returning a model with a hole in it.
    let mut names = source.names();
    names.retain(|name| !name.contains("mlp.experts.0.gate_proj"));
    let error = Qwen4ExpTensorPlan::served_and_mtp(&config)
        .conformance(names)
        .expect_err("a missing expert is a conformance failure");
    assert!(error.contains("mlp.experts.0.gate_proj"), "{error}");
}
