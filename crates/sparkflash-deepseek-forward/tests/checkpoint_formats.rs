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

//! A checkpoint header's stored format is a property of the checkpoint's own plan, not of
//! whether the caller happened to install that plan first.
//!
//! The chain loads the rank's headers through `setup_for_chain` and installs the open only after
//! the plan is built. This test runs exactly that order in a fresh process (this file is its own
//! test binary) and requires the header's format to be stated anyway.

use std::path::Path;

use sparkflash_deepseek_forward::tp4_resident::setup_for_chain;
use sparkflash_model::format::StoredFormat;
use sparkflash_weights::safetensors::write_checkpoint;
use sparkflash_weights::{DType, TensorSpec};

fn write_second_shape_checkpoint(directory: &Path) {
    std::fs::write(
        directory.join("config.json"),
        include_str!("fixtures/second-shape/config.json"),
    )
    .expect("the second-shape config");
    let spec = TensorSpec {
        name: "layers.0.attn_norm.weight".to_owned(),
        dtype: DType::BF16,
        shape: vec![4096],
    };
    let bytes = vec![0u8; spec.storage_bytes() as usize];
    write_checkpoint(directory, &[(spec, bytes)], 1).expect("a one-tensor checkpoint");
}

#[test]
fn the_format_is_stated_before_the_chain_installs_its_open() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_second_shape_checkpoint(dir.path());
    let evidence = dir.path().join("evidence");
    let checkpoint = dir.path().to_str().expect("utf-8");
    let evidence_text = evidence.to_str().expect("utf-8");
    // The chain's order: `setup_for_chain` builds the rank plan first, and `plan::install` runs
    // after it. Nothing is installed when the headers are read.
    let (setup, _source) = setup_for_chain(&[
        "--checkpoint",
        checkpoint,
        "--tp-rank",
        "0",
        "--tp-world-size",
        "1",
        "--evidence-dir",
        evidence_text,
    ])
    .expect("the checkpoint plans without an installed description");
    assert!(
        sparkflash_deepseek_forward::plan::installed().is_none(),
        "the test must observe the pre-install order"
    );
    let shard = setup
        .plan
        .shards()
        .iter()
        .find(|shard| shard.name == "layers.0.attn_norm.weight")
        .expect("the planned attn_norm");
    assert_eq!(
        shard.format,
        Some(StoredFormat::BF16),
        "the checkpoint header states BF16 without an installed plan"
    );
}
