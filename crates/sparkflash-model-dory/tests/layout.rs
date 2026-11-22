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

//! The layout's checkpoint plan equals the arm's own plan, name for name and shape for shape.
//!
//! The arm does not build an `ArchitectureTensorPlan`, so this test is the test-only oracle the
//! layout-equality instruction allows: [DoryTensorPlan::served] is the plan the arm loads and
//! executes today, and [ArchitectureTensorPlan::checkpoint] is the plan the shared description
//! builds from `models/layouts/dory1-4b.toml`. The comparison uses the same row a layout
//! author reads from `crates/sparkflash-ops/tests/layout_equality.rs`: stored dtype, logical
//! shape, rank shape, rank run and role.
//!
//! Dory's "rank" is a latent-recurrence stage, not a tensor-parallel slice: every rank holds
//! whole tensors (`recurrence.rs:17-23`). The layout therefore states `split = "whole"` for
//! every tensor, and this test checks that no world introduces a shard.

use std::collections::BTreeMap;

use sparkflash_model::{Architecture, TpPlan};
use sparkflash_model_dory::config::DoryConfig;
use sparkflash_model_dory::{DoryTensorPlan, Role as ArmRole};
use sparkflash_ops::tensor_plan::{ArchitectureTensorPlan, RankRun, Role as PlanRole};
use sparkflash_weights::TensorSpec;

const RELEASED: &str = include_str!("../config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json");

/// The shared plan's role for an arm role.
fn plan_role(role: ArmRole) -> PlanRole {
    match role {
        ArmRole::Embedding => PlanRole::Embedding,
        ArmRole::LmHead => PlanRole::LmHead,
        ArmRole::Attention => PlanRole::Attention,
        ArmRole::Mlp => PlanRole::DenseMlp,
    }
}

/// One tensor in the comparison, in the layout-equality harness's tuple order.
type Row = (String, Vec<usize>, Vec<usize>, Option<RankRun>, String);

fn arm_rows(plan: &DoryTensorPlan) -> BTreeMap<String, Row> {
    plan.specs()
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            (
                spec.name.clone(),
                (
                    format!("{:?}", spec.dtype),
                    spec.shape.clone(),
                    // The arm has one plan for every recurrence rank: a whole tensor is the
                    // rank's tensor unchanged.
                    spec.shape.clone(),
                    None,
                    format!("{:?}", plan_role(plan.role(index))),
                ),
            )
        })
        .collect()
}

fn layout_rows(plan: &ArchitectureTensorPlan) -> BTreeMap<String, Row> {
    let mut out = BTreeMap::new();
    for index in 0..plan.len() {
        let spec = plan.spec_at(index);
        out.insert(
            spec.name.clone(),
            (
                format!("{:?}", spec.dtype),
                spec.shape.clone(),
                plan.rank_spec_at(index).shape.clone(),
                plan.rank_run_at(index),
                format!("{:?}", plan.plan().role(index)),
            ),
        );
    }
    out
}

/// The first difference between the arm and the layout plan, in the harness's vocabulary.
fn first_difference(
    reference: &BTreeMap<String, Row>,
    layout: &BTreeMap<String, Row>,
) -> Option<String> {
    for name in reference.keys() {
        if !layout.contains_key(name) {
            return Some(format!("tensor {name}: the layout does not claim it"));
        }
    }
    for name in layout.keys() {
        if !reference.contains_key(name) {
            return Some(format!(
                "tensor {name}: the layout claims it, the arm does not"
            ));
        }
    }
    for (name, expected) in reference {
        let actual = &layout[name];
        if expected.0 != actual.0 {
            return Some(format!(
                "tensor {name}: expected stored dtype {}, layout built {}",
                expected.0, actual.0
            ));
        }
        if expected.1 != actual.1 {
            return Some(format!(
                "tensor {name}: expected shape {:?}, layout built {:?}",
                expected.1, actual.1
            ));
        }
        if expected.2 != actual.2 {
            return Some(format!(
                "tensor {name}: expected rank shape {:?}, layout built {:?}",
                expected.2, actual.2
            ));
        }
        if expected.3 != actual.3 {
            return Some(format!(
                "tensor {name}: expected rank run {:?}, layout built {:?}",
                expected.3, actual.3
            ));
        }
        if expected.4 != actual.4 {
            return Some(format!(
                "tensor {name}: expected role {}, layout built {}",
                expected.4, actual.4
            ));
        }
    }
    None
}

fn architecture() -> Architecture {
    sparkflash_model_dory::architecture_from_carried().expect("the carried config parses")
}

#[test]
fn the_layout_reproduces_the_arm_plan_at_every_rank() {
    let config = DoryConfig::from_json(RELEASED).expect("the released configuration parses");
    let reference = DoryTensorPlan::served(&config);
    let want = arm_rows(&reference);
    let architecture = architecture();
    for world in [1usize, 2, 4] {
        for rank in 0..world {
            let tp = TpPlan::new(world, rank).expect("a rank");
            let layout = ArchitectureTensorPlan::checkpoint(&architecture, tp)
                .expect("the layout checkpoint plan builds");
            assert_eq!(
                layout.len(),
                reference.len(),
                "TP{world} rank {rank}: tensor count"
            );
            assert_eq!(
                layout.storage_bytes(),
                reference.storage_bytes(),
                "TP{world} rank {rank}: byte total"
            );
            let have = layout_rows(&layout);
            if let Some(difference) = first_difference(&want, &have) {
                panic!("TP{world} rank {rank}: {difference}");
            }
        }
    }
}

#[test]
fn every_layout_tensor_is_a_whole_arm_tensor() {
    let config = DoryConfig::from_json(RELEASED).expect("the released configuration parses");
    let reference = DoryTensorPlan::served(&config);
    let architecture = architecture();
    let tp = TpPlan::new(1, 0).expect("one rank");
    let layout = ArchitectureTensorPlan::checkpoint(&architecture, tp).expect("the layout plan");
    for spec in reference.specs() {
        let found: &TensorSpec = layout.spec(&spec.name).expect("the arm tensor is claimed");
        assert_eq!(found.dtype, spec.dtype, "{} stored dtype", spec.name);
        assert_eq!(found.shape, spec.shape, "{} logical shape", spec.name);
        assert_eq!(
            layout.rank_spec(&spec.name).expect("the rank tensor").shape,
            spec.shape,
            "{} is not whole at TP1",
            spec.name
        );
        assert!(
            layout.rank_run(&spec.name).is_none(),
            "{} is sharded, but a recurrence rank holds whole tensors",
            spec.name
        );
    }
}
