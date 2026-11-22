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

//! The real snapshots the lead host holds, read header by header and checked against the shared
//! checkpoint plan.
//!
//! These are the interim expectations before the layout reader's stack tensor roles land: a plan
//! claims what its family planner names, and a tower no shared planner describes is reported as
//! unclaimed rather than dropped. The test is ignored by default because it needs the pinned
//! snapshots; an explicit `--ignored` run fails when none of them is present.
//!
//! This file is an integration test, deliberately outside `scan_report.rs`: the scan's vocabulary
//! must name no model family or crate, and a test that names released repositories would otherwise
//! be part of that source.

use sparkflash_model::Target;
use sparkflash_ops::{CoverageState, ScanInputs, ScanReport, ScanStatus, TensorIndex, scan};

/// One real snapshot the lead host holds. The test skips cleanly on a machine without it.
struct Snapshot {
    repo: &'static str,
    revision: &'static str,
    expectation: SnapshotExpectation,
    /// The vision parts with no kernel or no stated wiring, in report order.
    vision_gaps: &'static [&'static str],
}

enum SnapshotExpectation {
    /// Every tensor is claimed, but the tower's parts do not all resolve yet.
    Gaps,
    /// Every row resolves and both coverage checks are clean.
    Clean,
    /// These exact prefix groups are the only unclaimed tensors.
    Unclaimed(Vec<(&'static str, usize)>),
    /// The shared plan cannot be built; its tower tensors are the unclaimed ones and one
    /// layout row names the missing planner.
    PlanFailure {
        vision_tensors: usize,
        aligner_tensors: usize,
    },
}

fn snapshot_dir(snapshot: &Snapshot) -> Option<std::path::PathBuf> {
    let cache = sparkflash_weights::hub::default_cache_dir()?;
    let path = cache
        .join(sparkflash_weights::hub::repo_folder(snapshot.repo))
        .join("snapshots")
        .join(snapshot.revision);
    path.is_dir().then_some(path)
}

fn real_report(dir: &std::path::Path) -> ScanReport {
    let config = std::fs::read_to_string(dir.join("config.json")).expect("the config");
    let index = sparkflash_weights::tensor_index::read_local(dir).expect("the headers");
    assert_eq!(index.len(), index.tensors.len());
    let tensors = TensorIndex::from_reader(&index);
    scan(ScanInputs {
        config: &config,
        tensors: Some(&tensors),
        target: Target::Sm121,
        world: 1,
    })
}

/// The vision components whose rows did not resolve, excluding the config-key and tensor rows
/// whose part is the aggregate `vision`.
fn vision_gaps(report: &ScanReport) -> Vec<&'static str> {
    let mut gaps: Vec<&'static str> = report
        .rows
        .iter()
        .filter(|row| {
            row.stack == sparkflash_model::StackKind::Vision
                && row.part != sparkflash_model::Part::Vision
                && !row.status.is_resolved()
        })
        .map(|row| row.part.name())
        .collect();
    gaps.sort_unstable();
    gaps.dedup();
    gaps
}

fn unclaimed_groups(report: &ScanReport) -> Vec<(String, usize)> {
    report
        .rows
        .iter()
        .filter_map(|row| match &row.status {
            ScanStatus::UnclaimedTensor { tensor, .. } => Some(tensor),
            _ => None,
        })
        .map(|tensor| {
            // A group label is "prefix.* (N tensors)"; an individual row is its name.
            if let Some((prefix, count)) = tensor.split_once(".* (") {
                let count = count
                    .split(' ')
                    .next()
                    .and_then(|count| count.parse().ok())
                    .unwrap_or(1);
                (prefix.to_owned(), count)
            } else {
                (tensor.clone(), 1)
            }
        })
        .collect()
}

#[test]
#[ignore = "needs the pinned snapshots in the Hugging Face cache; run with --ignored"]
fn the_real_snapshots_report_their_interim_claim_state() {
    // Integration dependency: these expectations are the shared checkpoint plan's state
    // before piece A's stack tensor roles land. When A claims a tower, its group leaves this
    // table and the corresponding snapshot becomes Clean.
    let snapshots = [
        Snapshot {
            repo: "zai-org/GLM-5.3-Flash",
            revision: "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
            expectation: SnapshotExpectation::Gaps,
            vision_gaps: &[
                "vision_attention",
                "vision_downsample",
                "vision_merger",
                "vision_mlp",
                "vision_norm",
                "vision_patch_embed",
            ],
        },
        Snapshot {
            repo: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            revision: "1bae0e1088993b9b82de08f59de5be5a34e42170",
            expectation: SnapshotExpectation::PlanFailure {
                vision_tensors: 259,
                aligner_tensors: 4,
            },
            vision_gaps: &["vision_merger", "vision_mlp", "vision_norm"],
        },
        Snapshot {
            repo: "nvidia/Qwen3.8-Flash-Next-NVFP4",
            revision: "fc694b54fb0174e0913e6adf86691ef85a4ead47",
            expectation: SnapshotExpectation::Gaps,
            vision_gaps: &[
                "vision_attention",
                "vision_merger",
                "vision_mlp",
                "vision_norm",
                "vision_patch_embed",
                "vision_position",
            ],
        },
        Snapshot {
            repo: "thinkingmachines/Inkling-2-Small",
            revision: "885d78bfca61a7e9a498e33cfa7ad35850533b8c",
            expectation: SnapshotExpectation::Unclaimed(vec![("model.visual", 422)]),
            vision_gaps: &["vision_merger", "vision_mlp", "vision_position"],
        },
        Snapshot {
            repo: "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
            revision: "f6c832e6963ddb74f62c75813a3e8c74b1af67b1",
            // The layout names every tensor with the checkpoint's `backbone.*` spelling; the
            // arm's plan and the shared plan are equal at each rank the arm builds.
            expectation: SnapshotExpectation::Clean,
            vision_gaps: &[],
        },
    ];
    let mut checked = 0;
    for snapshot in &snapshots {
        let Some(dir) = snapshot_dir(snapshot) else {
            continue;
        };
        checked += 1;
        let report = real_report(&dir);
        assert!(
            report.error.is_none(),
            "{}: {}",
            snapshot.repo,
            report.error.as_deref().unwrap_or("")
        );
        assert_eq!(
            vision_gaps(&report),
            snapshot.vision_gaps,
            "{}: the tower's unresolved components moved\n{}",
            snapshot.repo,
            report.render()
        );
        match &snapshot.expectation {
            SnapshotExpectation::Gaps => {
                let groups = unclaimed_groups(&report);
                assert!(
                    groups.is_empty(),
                    "{}: unclaimed {groups:?}\n{}",
                    snapshot.repo,
                    report.render()
                );
                assert_eq!(
                    report.coverage.tensors,
                    CoverageState::Clean,
                    "{}: the plan and the index must agree\n{}",
                    snapshot.repo,
                    report.render()
                );
                assert!(
                    !report.is_resolved(),
                    "{}: a gap snapshot resolved every row\n{}",
                    snapshot.repo,
                    report.render()
                );
                if snapshot.repo == "zai-org/GLM-5.3-Flash" {
                    let unknown: Vec<&str> = report
                        .rows
                        .iter()
                        .filter_map(|row| match &row.status {
                            ScanStatus::UnknownLayout { detail, .. } => Some(detail.as_str()),
                            _ => None,
                        })
                        .collect();
                    assert!(
                        unknown.is_empty(),
                        "{}: the layout left wiring unknown: {unknown:?}\n{}",
                        snapshot.repo,
                        report.render()
                    );
                    for row in report
                        .rows
                        .iter()
                        .filter(|row| row.stack == sparkflash_model::StackKind::Vision)
                    {
                        assert!(
                            !row.shape.is_unknown() && row.format.is_some(),
                            "{}: vision part {} has no stored shape or format\n{}",
                            snapshot.repo,
                            row.part.name(),
                            report.render()
                        );
                        assert!(
                            matches!(
                                row.status,
                                ScanStatus::Resolved
                                    | ScanStatus::NotBuilt { .. }
                                    | ScanStatus::NoPartKind { .. }
                            ),
                            "{}: vision part {} has status {}\n{}",
                            snapshot.repo,
                            row.part.name(),
                            row.status.name(),
                            report.render()
                        );
                    }
                }
            }
            SnapshotExpectation::Clean => {
                let groups = unclaimed_groups(&report);
                assert!(
                    groups.is_empty(),
                    "{}: unclaimed {groups:?}\n{}",
                    snapshot.repo,
                    report.render()
                );
                let unknown: Vec<&str> = report
                    .rows
                    .iter()
                    .filter_map(|row| match &row.status {
                        ScanStatus::UnknownLayout { detail, .. } => Some(detail.as_str()),
                        _ => None,
                    })
                    .collect();
                assert!(
                    unknown.is_empty(),
                    "{}: the layout left wiring unknown: {unknown:?}\n{}",
                    snapshot.repo,
                    report.render()
                );
                assert!(
                    report.is_clean(),
                    "{}: not clean\n{}",
                    snapshot.repo,
                    report.render()
                );
            }
            SnapshotExpectation::Unclaimed(expected) => {
                let expected: Vec<(String, usize)> = expected
                    .iter()
                    .map(|(prefix, count)| ((*prefix).to_owned(), *count))
                    .collect();
                assert_eq!(
                    unclaimed_groups(&report),
                    expected,
                    "{}:\n{}",
                    snapshot.repo,
                    report.render()
                );
            }
            SnapshotExpectation::PlanFailure {
                vision_tensors,
                aligner_tensors,
            } => {
                let layout = report
                    .rows
                    .iter()
                    .filter(|row| matches!(row.status, ScanStatus::UnknownLayout { .. }))
                    .collect::<Vec<_>>();
                assert!(
                    layout.iter().any(|row| row
                        .status
                        .fix()
                        .is_some_and(|fix| fix.contains("stack or layout"))),
                    "{}: no row names the missing plan or layout\n{}",
                    snapshot.repo,
                    report.render()
                );
                assert!(
                    layout.iter().all(|row| row.status.fix().is_some()),
                    "{}: an unknown-layout row has no fix\n{}",
                    snapshot.repo,
                    report.render()
                );
                let groups = unclaimed_groups(&report);
                let vision = groups
                    .iter()
                    .find(|(prefix, _)| prefix == "vision")
                    .cloned();
                if *vision_tensors == 0 {
                    assert!(vision.is_none(), "{}:\n{}", snapshot.repo, report.render());
                } else {
                    assert_eq!(
                        vision,
                        Some(("vision".to_owned(), *vision_tensors)),
                        "{}:\n{}",
                        snapshot.repo,
                        report.render()
                    );
                }
                let aligner: usize = groups
                    .iter()
                    .filter(|(prefix, _)| prefix.starts_with("aligner"))
                    .map(|(_, count)| count)
                    .sum();
                assert_eq!(
                    aligner,
                    *aligner_tensors,
                    "{}:\n{}",
                    snapshot.repo,
                    report.render()
                );
            }
        }
    }
    assert!(
        checked > 0,
        "no pinned snapshot is present in this machine's Hub cache; the ignored test checked nothing"
    );
}
