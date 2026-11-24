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

//! The wo_a operand is one rank's rows at every world, for both shapes.
//!
//! The device chain binds a projection by the plan's own operand, so the plan must state the
//! per-rank slice and the bind must read that statement rather than the whole tensor's stored
//! length. The defect these tests pin: the stored-layout byte check compared the arena slice
//! (the rank's rows) against the unsharded length, which coincides at world 1 and refuses worlds
//! 2 and 4 by exactly the world. The second failure is the plan's arrangement: a 32 x 32
//! power-of-two block-grid class is read in B12x micro-tiles, so a plan that states the file's
//! row-major arrangement sends a stored operand to a body that reads tiles.
//
//! Every value here is host-side: the plan, the resident plan's slice, and the operand layout.

use std::path::{Path, PathBuf};

use sparkflash_device::resident::{HeaderEntry, OperandLayout, ResidentPlan};
use sparkflash_model::Target;
use sparkflash_model_deepseek_v41::config::ModelConfig;
use sparkflash_model_deepseek_v41::tensor::TensorPlan;
use sparkflash_ops::engine_plan::EnginePlan;

fn second_shape() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/second-shape/config.json")
}

fn released() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json")
}

fn model_config(path: &Path) -> ModelConfig {
    let text = std::fs::read_to_string(path).expect("the configuration");
    ModelConfig::from_json(&text).expect("the configuration parses")
}

/// The kernel source one rank opens, read through the same header function the chain uses.
///
/// Only the named tensors are planned: the packed expert container has no element width in a
/// safetensors header, and it says nothing about the attention output or an embedding.
fn resident(path: &Path, world: usize, written: &TensorPlan, names: &[&str]) -> ResidentPlan {
    let plan = EnginePlan::from_path(path, Target::Sm121, world, 0)
        .unwrap_or_else(|error| panic!("world {world}: {error}"));
    let entries = written
        .specs()
        .iter()
        .filter(|spec| names.contains(&spec.name.as_str()))
        .map(|spec| {
            let dtype = spec.dtype.name();
            let format = sparkflash_deepseek_forward::tp4_resident::tensor_format(&plan, dtype);
            HeaderEntry::new(spec.name.clone(), dtype, spec.shape.clone()).with_format(format)
        });
    ResidentPlan::plan(0, world, entries)
        .unwrap_or_else(|error| panic!("world {world}: the resident plan: {error}"))
}

#[test]
fn the_wo_a_operand_is_one_ranks_rows_at_every_world() {
    // (label, config path, whole wo_a rows, whole group width, per-world bytes)
    let cases = [
        (
            "second shape",
            second_shape(),
            1024usize,
            4096usize,
            [(1usize, 4_194_304u64), (2, 2_097_152), (4, 1_048_576)],
        ),
        (
            "released",
            released(),
            8192,
            4096,
            [(1, 33_554_432), (2, 16_777_216), (4, 8_388_608)],
        ),
    ];
    for (label, path, rows, inner, worlds) in cases {
        let config = model_config(&path);
        let written = TensorPlan::language_model(&config);
        for (world, want_bytes) in worlds {
            let resident = resident(&path, world, &written, &["layers.0.attn.wo_a.weight"]);
            let shard = resident
                .shards()
                .iter()
                .find(|shard| shard.name == "layers.0.attn.wo_a.weight")
                .unwrap_or_else(|| panic!("{label} world {world}: wo_a is not planned"));
            let want_rows = rows / world;
            assert_eq!(
                shard.stored_shape,
                vec![rows, inner],
                "{label} world {world}: the stored wo_a shape"
            );
            assert_eq!(
                shard.stored_bytes,
                (rows * inner) as u64,
                "{label} world {world}: the stored wo_a bytes"
            );
            assert_eq!(
                shard.shard_shape,
                vec![want_rows, inner],
                "{label} world {world}: this rank's wo_a rows"
            );
            assert_eq!(
                shard.shard_bytes, want_bytes,
                "{label} world {world}: this rank's wo_a bytes"
            );
            match shard.operand_layout() {
                OperandLayout::B12x {
                    cols,
                    inner: got_inner,
                    weight_bytes,
                    scale_bytes,
                    block,
                } => {
                    assert_eq!(
                        (cols, got_inner),
                        (want_rows, inner),
                        "{label} world {world}: the operand geometry"
                    );
                    assert_eq!(
                        weight_bytes, want_bytes,
                        "{label} world {world}: the B12x payload is this rank's rows, not the whole tensor"
                    );
                    assert_eq!(block, 32, "{label} world {world}: the micro-tile");
                    assert_eq!(
                        scale_bytes,
                        (want_rows.div_ceil(32) * inner.div_ceil(32)) as u64,
                        "{label} world {world}: the group's scale grid"
                    );
                }
                other => panic!(
                    "{label} world {world}: wo_a is planned {other:?}; the body reads B12x and a stored plan would not tile it"
                ),
            }
            // The failing comparison in one line: the whole tensor's length is world-blind, and at
            // world 1 it is indistinguishable from the slice.
            assert_eq!(
                shard.shard_bytes * world as u64,
                shard.stored_bytes,
                "{label} world {world}: the slice is the whole tensor over the world"
            );
            if world > 1 {
                assert_ne!(
                    shard.shard_bytes, shard.stored_bytes,
                    "{label} world {world}: a row split must not leave the slice at the whole tensor's length"
                );
            }
        }
    }
}

#[test]
fn a_split_stored_slice_is_checked_by_its_own_bytes() {
    // The stored arm of the bind reads the plan's operand, and a stored operand states this
    // rank's slice. At world 1 the two are equal; at 2 and 4 a stored split tensor is the case
    // the old comparison refused by the world.
    for (label, path) in [("second shape", second_shape()), ("released", released())] {
        let config = model_config(&path);
        let written = TensorPlan::language_model(&config);
        for world in [2usize, 4] {
            let resident = resident(&path, world, &written, &["embed.weight"]);
            let shard = resident
                .shards()
                .iter()
                .find(|shard| {
                    matches!(
                        shard.operand_layout(),
                        OperandLayout::Stored { bytes, .. } if bytes < shard.stored_bytes
                    )
                })
                .unwrap_or_else(|| panic!("{label} world {world}: no stored split tensor to pin"));
            match shard.operand_layout() {
                OperandLayout::Stored { bytes, .. } => {
                    assert_eq!(
                        bytes, shard.shard_bytes,
                        "{label} world {world}: {} states this rank's slice",
                        shard.name
                    );
                    assert_eq!(
                        bytes * world as u64,
                        shard.stored_bytes,
                        "{label} world {world}: {} is the whole tensor over the world",
                        shard.name
                    );
                }
                other => panic!(
                    "{label} world {world}: {} is not stored: {other:?}",
                    shard.name
                ),
            }
        }
    }
}
