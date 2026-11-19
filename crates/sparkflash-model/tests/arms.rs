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

//! The workspace checks that make the model registration one table.
//!
//! These are CPU-only and read the tree, so they run in the same per-commit CI as the model
//! crate: a model crate missing from a CI shard, a direct environment read in a model crate, or
//! an arm with no ledger row fails here rather than in review.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use sparkflash_model::{ARMS, ModelProfile, Strategy, arm_for_weights};

/// The model crates whose arms still read the environment at the point of use.
///
/// This table is phase 2's *first* commit, not the end state: the plan moves these reads
/// through `profile.rs` in phase 7. The counts are the audit's measured values (20 / 2 / 1,
/// plus the one `env::var_os` in Qwen's test helper) and are asserted exactly so a new read
/// cannot hide behind the exception.
const ENV_READ_EXCEPTIONS: &[(&str, usize)] = &[
    // The generated conformance checks name the dense arm's synthetic checkpoint by this
    // variable; it is the check's fixture, not a served setting, so it stays out of the profile.
    ("sparkflash-model-dense", 1),
    ("sparkflash-model-glm5-next", 2),
    ("sparkflash-model-inkling2", 20),
    ("sparkflash-model-qwen4-exp", 2),
];

/// The repository root, from this crate's manifest directory.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the model crate is under crates/")
        .to_path_buf()
}

/// Every `sparkflash-model-*` crate directory under `crates/`.
fn model_crates() -> Vec<PathBuf> {
    let mut crates = fs::read_dir(root().join("crates"))
        .expect("the workspace crates directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("sparkflash-model-"))
        })
        .collect::<Vec<_>>();
    crates.sort();
    crates
}

/// Count `env::var` and `env::var_os` calls in every `.rs` file under `directory`.
fn count_env_reads(directory: &Path) -> usize {
    let mut total = 0;
    let Ok(entries) = fs::read_dir(directory) else {
        return total;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            total += count_env_reads(&path);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let text = fs::read_to_string(&path).unwrap_or_default();
            total += text
                .lines()
                .filter(|line| line.contains("env::var") || line.contains("env::var_os"))
                .count();
        }
    }
    total
}

#[test]
fn every_weight_selector_resolves_in_the_registry() {
    for arm in ARMS {
        for weights in arm.weights() {
            assert!(
                sparkflash_model::find_model_by_weights(weights).is_some(),
                "{} names {weights}, which MODELS does not describe",
                arm.id()
            );
        }
    }
}

#[test]
fn every_registered_model_has_an_arm() {
    for model in sparkflash_model::MODELS.iter() {
        assert!(
            arm_for_weights(&model.weights_id).is_some(),
            "{} ({}) has no ARMS entry",
            model.id,
            model.weights_id
        );
    }
}

#[test]
fn every_arm_has_all_eleven_ledger_rows_and_a_numerics_status() {
    for arm in ARMS {
        assert_eq!(
            arm.ledger().rows.len(),
            Strategy::ALL.len(),
            "{} states a strategy twice",
            arm.id()
        );
        assert!(
            arm.ledger().unstated().is_empty(),
            "{} leaves a strategy out",
            arm.id()
        );
        for weights in arm.weights() {
            assert!(
                arm.ledger().weights_ids.contains(weights),
                "{}'s ledger does not name {weights}",
                arm.id()
            );
        }
        let status = sparkflash_model::numerics_status(*arm);
        assert!(
            status == "captured"
                || status == "none (registered, uncaptured)"
                || status == "none (no entry)",
            "{} has no numerics status: {status}",
            arm.id()
        );
    }
}

#[test]
fn every_carried_profile_parses() {
    for arm in ARMS {
        if let Some(text) = arm.profile_toml() {
            ModelProfile::from_toml(text)
                .unwrap_or_else(|error| panic!("{}'s carried profile: {error}", arm.id()));
        }
    }
}

#[test]
fn every_model_crate_is_named_in_both_ci_files() {
    let gitlab = fs::read_to_string(root().join(".gitlab-ci.yml")).expect("the GitLab pipeline");
    let github = fs::read_to_string(root().join(".github/workflows/ci.yml"))
        .expect("the GitHub Actions workflow");
    let mut crates = BTreeSet::new();
    for path in model_crates() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("a crate name")
            .to_owned();
        assert!(
            gitlab.contains(&name),
            "{name} is not named in .gitlab-ci.yml"
        );
        assert!(
            github.contains(&name),
            "{name} is not named in .github/workflows/ci.yml"
        );
        crates.insert(name);
    }
    assert_eq!(
        crates.len(),
        model_crates().len(),
        "the crate list is unique"
    );
}

#[test]
fn a_model_crate_reads_no_environment_beyond_the_recorded_table() {
    let allowed = ENV_READ_EXCEPTIONS
        .iter()
        .copied()
        .collect::<std::collections::BTreeMap<_, _>>();
    for path in model_crates() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("a crate name");
        let measured = count_env_reads(&path);
        let expected = allowed.get(name).copied().unwrap_or(0);
        assert_eq!(
            measured, expected,
            "{name} has {measured} direct environment reads; {expected} are in the exception              table (move the rest through profile.rs, or record them here)"
        );
    }
}
