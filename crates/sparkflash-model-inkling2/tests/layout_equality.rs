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

//! Proof that models/layouts/inkling2.toml reproduces the arm's plan.
//!
//! The arm's plan is its own type (`InklingTensorPlan`), so this test is the oracle that
//! captures it and compares it with the plan the shared reader builds from the layout, in the
//! shape of `crates/sparkflash-ops/tests/layout_equality.rs`: by tensor name, logical shape and
//! rank-local shape/run. The arm's MTP names are for its converter (a separate file and a
//! `mtp.` prefix) and its plan is not the full stage, so the comparison is the language and
//! vision stacks, where the arm is the released plan.

use std::collections::BTreeMap;

use sparkflash_model::TpPlan;
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::tensor::InklingTensorPlan;
use sparkflash_ops::ArchitectureTensorPlan;

const CONFIG: &str = include_str!("../config/inkling2_small.hf.json");
const QUANT: &str = include_str!("../config/inkling2_small.hf_quant.json");

type Row = (
    sparkflash_weights::DType,
    Vec<usize>,
    Vec<usize>,
    Option<sparkflash_ops::RankRun>,
);

fn arm_rows(config: &Inkling2Config, tp: TpPlan) -> BTreeMap<String, Row> {
    let plan = InklingTensorPlan::checkpoint(config, tp).expect("the arm's plan");
    let mut rows = BTreeMap::new();
    for index in 0..plan.plan().len() {
        let spec = &plan.plan().specs()[index];
        if !(spec.name.starts_with("model.llm.")
            || spec.name.starts_with("model.visual.")
            || spec.name.starts_with("mtp."))
        {
            continue;
        }
        let run = plan
            .rank_run_at(index)
            .map(|slice| sparkflash_ops::RankRun {
                offset: slice.split.offset,
                count: slice.split.per_rank,
                total: slice.split.total,
            });
        rows.insert(
            spec.name.clone(),
            (
                spec.dtype,
                spec.shape.clone(),
                plan.rank_spec_at(index).shape.clone(),
                run,
            ),
        );
    }
    rows
}

fn layout_rows(config: &Inkling2Config, tp: TpPlan) -> BTreeMap<String, Row> {
    let plan = ArchitectureTensorPlan::checkpoint(&config.architecture, tp)
        .expect("the layout's checkpoint plan");
    let mut rows = BTreeMap::new();
    for index in 0..plan.plan().len() {
        let spec = plan.spec_at(index);
        // The checkpoint's mtp.safetensors stores bare `layers.N` names; the arm prefixes
        // them with its runtime `mtp.` root. Compare under the arm's spelling.
        let name = if spec.name.starts_with("model.") {
            spec.name.clone()
        } else {
            format!("mtp.{}", spec.name)
        };
        rows.insert(
            name,
            (
                spec.dtype,
                spec.shape.clone(),
                plan.rank_spec_at(index).shape.clone(),
                plan.rank_run_at(index),
            ),
        );
    }
    rows
}

#[test]
fn the_layout_reproduces_the_arms_language_and_vision_plan() {
    let config = Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the released config");
    for world in [1usize, 2, 4] {
        let tp = TpPlan::new(world, 0).expect("a TP plan");
        let arm = arm_rows(&config, tp);
        let layout = layout_rows(&config, tp);

        let missing: Vec<&String> = arm
            .keys()
            .filter(|name| !layout.contains_key(*name))
            .collect();
        assert!(
            missing.is_empty(),
            "TP{world}: the layout misses {missing:?}"
        );

        // The arm's converter plans only the MTP tensors its runtime reads; the checkpoint's
        // mtp.safetensors stores more (the Markov head, its mask, and the stage norms). Those
        // extras are allowed only in the MTP stack, never in the language or vision plan.
        let extra: Vec<&String> = layout
            .keys()
            .filter(|name| !arm.contains_key(*name))
            .collect();
        let stray: Vec<&&String> = extra
            .iter()
            .filter(|name| !name.starts_with("mtp."))
            .collect();
        assert!(
            stray.is_empty(),
            "TP{world}: the layout adds non-MTP tensors {stray:?}"
        );

        for (name, expected) in &arm {
            let Some(actual) = layout.get(name) else {
                continue;
            };
            assert_eq!(
                expected.0, actual.0,
                "TP{world}: {name} stored dtype differs"
            );
            assert_eq!(
                expected.1, actual.1,
                "TP{world}: {name} logical shape differs"
            );
            assert_eq!(expected.2, actual.2, "TP{world}: {name} rank shape differs");
            assert_eq!(expected.3, actual.3, "TP{world}: {name} rank run differs");
        }
    }
}
