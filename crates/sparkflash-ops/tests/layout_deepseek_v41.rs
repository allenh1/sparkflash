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

//! The DeepSeek-V4.1-Flash layout against the plan its arm builds today.
//!
//! The arm plans the language model and its MTP stages with
//! [sparkflash_model_deepseek_v41::TensorPlan], not with
//! [sparkflash_ops::ArchitectureTensorPlan]. This test is the oracle for that plan: every
//! language/MTP tensor the layout claims must be in the arm's plan with the same logical shape
//! and stored dtype, and every arm tensor must be claimed by the layout.
//!
//! The tower and aligner are not in the arm's plan (the arm loads them through its resident
//! image path), so they are checked for their expected counts and prefixes. The three image
//! delimiters are also outside the arm plan; the layout claims them unloaded because the resident
//! path reads them, and the test names that difference instead of hiding it.

use std::collections::BTreeSet;

use sparkflash_model::{Architecture, TpPlan};
use sparkflash_model_deepseek_v41::{ModelConfig, TensorPlan};
use sparkflash_ops::ArchitectureTensorPlan;
use sparkflash_ops::tensor_plan::TensorClaims;

const RELEASED: &str =
    include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");

/// The rank count the checkpoint is served at.
const WORLD: usize = 4;

/// The tensors the layout claims that the arm's TensorPlan does not contain.
const ARM_OMITS: [&str; 3] = ["image_start", "image_end", "image_newline"];

#[test]
fn the_layout_equals_the_arm_plan_at_tp4() {
    let architecture = Architecture::from_json(RELEASED).expect("the released configuration");
    let tp = TpPlan::for_architecture(&architecture, WORLD, 0).expect("a TP4 plan");
    let layout =
        ArchitectureTensorPlan::checkpoint(&architecture, tp).expect("the layout checkpoint plan");
    let arm = TensorPlan::language_model(
        &ModelConfig::from_released_json(RELEASED).expect("the arm's configuration"),
    );

    let mut wrong = Vec::new();
    let mut vision = 0usize;
    let mut aligner = 0usize;
    let mut extras = Vec::new();
    for index in 0..layout.len() {
        let spec = layout.spec_at(index);
        if spec.name.starts_with("vision.") {
            vision += 1;
            continue;
        }
        if spec.name.starts_with("aligner.") {
            aligner += 1;
            continue;
        }
        let Some(expected) = arm.get(&spec.name) else {
            if ARM_OMITS.contains(&spec.name.as_str()) {
                extras.push(spec.name.as_str());
            } else {
                wrong.push(format!(
                    "tensor {}: the layout claims it, the arm does not",
                    spec.name
                ));
            }
            continue;
        };
        if expected.shape != spec.shape {
            wrong.push(format!(
                "tensor {}: the arm builds shape {:?}, the layout built {:?}",
                spec.name, expected.shape, spec.shape
            ));
        }
        if expected.dtype.name() != spec.dtype.name() {
            wrong.push(format!(
                "tensor {}: the arm stores {}, the layout built {}",
                spec.name,
                expected.dtype.name(),
                spec.dtype.name()
            ));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");

    assert_eq!(vision, 259, "the tower's 259 indexed tensors are claimed");
    assert_eq!(aligner, 4, "the aligner's four indexed tensors are claimed");
    assert!(
        extras.is_empty(),
        "an unloaded delimiter is not a tensor the plan carries: {extras:?}"
    );

    let claimed: BTreeSet<&str> = (0..layout.len())
        .map(|index| layout.spec_at(index).name.as_str())
        .collect();
    let missing: Vec<&str> = arm.names().filter(|name| !claimed.contains(name)).collect();
    assert!(
        missing.is_empty(),
        "the layout does not claim these arm tensors: {missing:#?}"
    );
    assert_eq!(layout.len(), arm.len() + vision + aligner);

    // The three resident image delimiters are not plan members, but the index holds them: the
    // claim set must still name them at their stored shape or the scan calls them unclaimed.
    let claims = TensorClaims::from_checkpoint(&architecture, tp).expect("the claim set");
    for name in ARM_OMITS {
        let spec = claims
            .specs()
            .iter()
            .find(|spec| spec.name == name)
            .unwrap_or_else(|| panic!("the claim set omits {name}"));
        assert_eq!(spec.shape, vec![architecture.hidden_size], "{name}");
    }
}

/// The real snapshot the lead host holds. The test skips on a machine without it.
const SNAPSHOT_REPO: &str = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
const SNAPSHOT_REVISION: &str = "1bae0e1088993b9b82de08f59de5be5a34e42170";

fn snapshot_dir() -> Option<std::path::PathBuf> {
    let cache = sparkflash_weights::hub::default_cache_dir()?;
    let path = cache
        .join(sparkflash_weights::hub::repo_folder(SNAPSHOT_REPO))
        .join("snapshots")
        .join(SNAPSHOT_REVISION);
    path.is_dir().then_some(path)
}

/// The acceptance scan: the real index at the served rank count, vision on.
///
/// This is the same scan the console and onboard call, so it needs no CLI build. It skips
/// when the snapshot is absent; on the lead host it is the TP4 gate.
#[test]
fn the_real_snapshot_scans_clean_at_tp4() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: {SNAPSHOT_REPO} @ {SNAPSHOT_REVISION} is not in the local cache");
        return;
    };
    let config = std::fs::read_to_string(dir.join("config.json")).expect("the config");
    let index = sparkflash_weights::tensor_index::read_local(&dir).expect("the headers");
    let tensors = sparkflash_ops::TensorIndex::from_reader(&index);
    let report = sparkflash_ops::scan(sparkflash_ops::ScanInputs {
        config: &config,
        tensors: Some(&tensors),
        target: sparkflash_model::Target::Sm121,
        world: 4,
    });
    println!("{}", report.render());
    assert!(report.error.is_none(), "{:?}", report.error);
    let unresolved: Vec<&sparkflash_ops::ScanRow> = report
        .rows
        .iter()
        .filter(|row| !row.status.is_resolved())
        .collect();
    assert!(
        unresolved.is_empty(),
        "the TP4 scan still has {} unresolved rows",
        unresolved.len()
    );
    assert_eq!(
        report.coverage.tensors,
        sparkflash_ops::CoverageState::Clean
    );
    assert_eq!(
        report.coverage.config_keys,
        sparkflash_ops::CoverageState::Clean
    );
}
