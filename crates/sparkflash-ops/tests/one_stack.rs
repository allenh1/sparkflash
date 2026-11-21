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

//! The structural gates: no model CUDA tree, no model-to-model dependency (a family may build on
//! the shared dense host), one workspace member.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/sparkflash-ops is two levels below the workspace root")
        .to_path_buf()
}

fn model_crates() -> Vec<PathBuf> {
    let crates = workspace_root().join("crates");
    let mut out: Vec<PathBuf> = std::fs::read_dir(&crates)
        .expect("the workspace has a crates directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("sparkflash-model-"))
        })
        .collect();
    out.sort();
    out
}

fn cuda_sources(directory: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            cuda_sources(&path, out);
        } else if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| matches!(extension, "cu" | "cuh"))
        {
            out.push(path);
        }
    }
}

#[test]
fn no_model_crate_owns_cuda_or_a_kernel_build_script() {
    let root = workspace_root();
    for crate_dir in model_crates() {
        let mut sources = Vec::new();
        cuda_sources(&crate_dir.join("src"), &mut sources);
        if crate_dir.join("kernels").is_dir() {
            cuda_sources(&crate_dir.join("kernels"), &mut sources);
        }
        assert!(
            sources.is_empty(),
            "{} still owns CUDA sources: {sources:?}",
            crate_dir
                .strip_prefix(&root)
                .unwrap_or(&crate_dir)
                .display()
        );
        assert!(
            !crate_dir.join("build.rs").is_file(),
            "{} still has a kernel build script",
            crate_dir.display()
        );
    }
}

/// The one model crate another may depend on: the shared dense host.
const SHARED_HOST: &str = "sparkflash-model-dense";

#[test]
fn no_model_crate_depends_on_another_model_crate() {
    let root = workspace_root();
    for crate_dir in model_crates() {
        let manifest = std::fs::read_to_string(crate_dir.join("Cargo.toml")).expect("manifest");
        let name = crate_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        for other in model_crates() {
            let other_name = other
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            // The shared dense host is the base a family's own parts plug into
            // (`sparkflash onboard --emit` writes a crate on it), not a peer arm.
            if other_name == name || other_name == SHARED_HOST {
                continue;
            }
            assert!(
                !manifest.contains(other_name),
                "{} depends on {other_name}; shared code belongs in sparkflash-ops",
                crate_dir
                    .strip_prefix(&root)
                    .unwrap_or(&crate_dir)
                    .display()
            );
        }
    }
}

#[test]
fn the_language_stack_walk_matches_the_scan_and_the_plan_order() {
    use sparkflash_model::{Architecture, Part, Target};
    use sparkflash_ops::{ScanInputs, scan};

    let config = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":128,"max_position_embeddings":16}"#;
    let architecture = Architecture::from_json(config).expect("a synthetic config");
    let stacks = architecture.stacks();
    let plan = sparkflash_ops::EnginePlan::from_json(config, Target::Sm121, 1, 0).expect("a plan");
    assert_eq!(stacks.len(), 1);
    assert_eq!(stacks[0].blocks.len(), plan.layers().len());

    let report = scan(ScanInputs {
        config,
        tensors: None,
        target: Target::Sm121,
        world: 1,
    });
    for (block, layer) in stacks[0].blocks.iter().zip(plan.layers()) {
        let from_plan: Vec<Part> = layer
            .requirements
            .iter()
            .map(|requirement| requirement.part)
            .collect();
        let from_stack: Vec<Part> = block.parts.iter().map(|instance| instance.part).collect();
        assert_eq!(from_stack, from_plan, "block {}", block.index);

        let rows: Vec<_> = report
            .rows
            .iter()
            .filter(|row| row.block == Some(block.index))
            .collect();
        assert_eq!(rows.len(), block.parts.len(), "one row per instance");
        for (row, instance) in rows.iter().zip(&block.parts) {
            assert_eq!(row.part, instance.part);
            assert_eq!(
                row.shape, instance.shape,
                "the scan and the description must state the same shape for {}",
                instance.part
            );
        }
    }
}

fn model_families() -> Vec<String> {
    let mut families: Vec<String> = std::fs::read_dir(workspace_root().join("crates"))
        .expect("the workspace has a crates directory")
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .filter_map(|name| name.strip_prefix("sparkflash-model-").map(str::to_owned))
        .map(|name| {
            name.split('-')
                .next()
                .unwrap_or("")
                .trim_end_matches(|character: char| character.is_ascii_digit())
                .to_lowercase()
        })
        .filter(|name| !name.is_empty() && name != "dense")
        .collect();
    families.extend(["deepseek", "qwen", "glm", "inkling", "dory"].map(str::to_owned));
    families.sort();
    families.dedup();
    families
}

#[test]
fn the_scan_agrees_with_the_plan_on_every_released_config() {
    use sparkflash_model::Target;
    use sparkflash_ops::{Availability, EnginePlan, KernelPhase, KernelResolver, ScanInputs, scan};

    let crates = workspace_root().join("crates");
    let mut configs = Vec::new();
    for entry in std::fs::read_dir(&crates).expect("the crates directory") {
        let directory = entry.expect("a crate entry").path();
        let Some(name) = directory
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
        else {
            continue;
        };
        if !name.starts_with("sparkflash-model-") {
            continue;
        }
        let Ok(files) = std::fs::read_dir(directory.join("config")) else {
            continue;
        };
        for file in files {
            let path = file.expect("a config entry").path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name.ends_with(".released.json") || file_name.ends_with(".hf.json") {
                configs.push(path);
            }
        }
    }
    configs.sort();
    assert!(configs.len() >= 6, "{configs:?}");

    for path in configs {
        let config = std::fs::read_to_string(&path).expect("a released config");
        let plan = EnginePlan::from_json(&config, Target::Sm121, 1, 0)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let report = scan(ScanInputs {
            config: &config,
            tensors: None,
            target: Target::Sm121,
            world: 1,
        });
        assert!(
            report.error.is_none(),
            "{}: {:?}",
            path.display(),
            report.error
        );
        let kernel_rows: Vec<_> = report
            .rows
            .iter()
            .filter(|row| {
                !matches!(
                    row.status,
                    sparkflash_ops::scan::ScanStatus::UnknownKey { .. }
                        | sparkflash_ops::scan::ScanStatus::UnclaimedTensor { .. }
                        | sparkflash_ops::scan::ScanStatus::UnknownLayout { .. }
                )
            })
            .collect();
        assert_eq!(kernel_rows.len(), plan.requirements().count());
        let resolver = KernelResolver::new(Target::Sm121, Availability::Catalogue);
        for (row, requirement) in kernel_rows.into_iter().zip(plan.requirements()) {
            let resolved = resolver
                .resolve_for(requirement, KernelPhase::Prefill)
                .is_ok()
                && resolver
                    .resolve_for(requirement, KernelPhase::Decode)
                    .is_ok();
            assert_eq!(
                row.status.is_resolved(),
                resolved,
                "{} {:?} block {:?}: scan and the resolver disagree",
                path.display(),
                requirement.part,
                row.block
            );
            assert_eq!(row.part, requirement.part);
            if matches!(
                requirement.part,
                sparkflash_model::Part::GdnScan | sparkflash_model::Part::KdaScan
            ) {
                assert!(
                    row.format.is_some(),
                    "{} {:?} must state the attention format the plan resolves at",
                    path.display(),
                    requirement.part
                );
            }
        }
    }
}

#[test]
fn a_tp4_scan_row_states_the_rank_local_shape_the_stack_describes_in_full() {
    use sparkflash_model::stack::Shape;
    use sparkflash_model::{Architecture, Part, Target};
    use sparkflash_ops::{KernelPhase, ScanInputs, scan};

    let config = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":8,"num_key_value_heads":8,"head_dim":128,"vocab_size":128,"max_position_embeddings":16}"#;
    let architecture = Architecture::from_json(config).expect("a synthetic config");
    let stacks = architecture.stacks();
    let described = stacks[0].blocks[0]
        .parts
        .iter()
        .find(|instance| instance.part == Part::GqaAttention)
        .expect("the block's attention");
    assert_eq!(described.shape, Shape::new([8, 128]));

    let report = scan(ScanInputs {
        config,
        tensors: None,
        target: Target::Sm121,
        world: 4,
    });
    assert!(report.error.is_none(), "{report:?}");
    let row = report
        .rows
        .iter()
        .find(|row| row.block == Some(0) && row.part == Part::GqaAttention)
        .expect("the rank's attention row");
    assert_eq!(
        row.shape,
        Shape::new([2, 128]),
        "four ranks hold two heads each"
    );
    assert_eq!(row.pass, KernelPhase::Both);
}

#[test]
fn the_scan_vocabulary_names_no_model_family_or_crate() {
    let families = model_families();
    assert!(families.contains(&"deepseek".to_owned()));
    for (file, source) in [
        (
            "stack.rs",
            include_str!("../../sparkflash-model/src/stack.rs"),
        ),
        ("scan_report.rs", include_str!("../src/scan_report.rs")),
    ] {
        let lower = source.to_lowercase();
        for family in &families {
            assert!(!lower.contains(family.as_str()), "{file} names {family}");
        }
        assert!(
            !lower.contains("sparkflash-model-"),
            "{file} names a model crate"
        );
    }
}

#[test]
fn the_ops_crate_is_a_workspace_member() {
    let manifest = std::fs::read_to_string(workspace_root().join("Cargo.toml")).expect("manifest");
    assert!(manifest.contains("\"crates/sparkflash-ops\""));
}

/// The CLI source-line ratchet.
///
/// Phase 4 moved DeepSeek's device forward out of the CLI and set the gate at 30k. Phase 5's
/// zero-code dynamic HF seam (`serve/dynamic_host.rs`, `serve/dynamic.rs` and `dynamic.rs`,
/// about 2.4k lines) landed inside `sparkflash-cli/src` because it reads `ServeOptions`, the
/// pack registry and the shared architecture directly. Phase 6's first fold moved the packed
/// arms' group-rank openers into the arm crates (`sparkflash-model-glm5-next/src/rank.rs`,
/// `sparkflash-model-qwen4-exp/src/rank.rs`) and the rank seam into `sparkflash-serve::tp4`:
/// 36,987 -> 36,566 measured. The second fold moved the seam's step contract (`Tp4RankApi` and
/// the step shapes) into `sparkflash-serve::tp4` and the GLM and Qwen step adapters into their
/// arms: 36,566 -> 35,972 measured, so the gate is one above it. The one-serve-entry fold moved
/// the arms' serving glue into `sparkflash_runtime::WalkArm` and the registration's remaining
/// serving sites onto `sparkflash_model::ARMS`: 35,905 -> 35,855 measured, so the gate is one
/// above it. Dory's serve on its tick engine and the one HTTP driver for the typed and the
/// dynamic route leave 35,864. Still owed: DeepSeek's 14,545-line serving path (`device_chain.rs`, `tp4_host.rs`,
/// `tp4_chain.rs`, `tp4_device_runner.rs`, `serve/device.rs`) folds into the common entry. The
/// gate is lowered as those land; it never rises.
const CLI_SOURCE_LINE_LIMIT: usize = 35_865;

#[test]
fn the_deepseek_forward_is_out_of_the_cli_and_the_cli_is_below_34k_lines() {
    let root = workspace_root();
    let cli = root.join("crates/sparkflash-cli/src");
    let mut total = 0usize;
    let mut stack = vec![cli.clone()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(&directory)
            .expect("CLI source tree")
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                total += std::fs::read_to_string(&path)
                    .expect("source")
                    .lines()
                    .count();
            }
        }
    }
    assert!(
        total < CLI_SOURCE_LINE_LIMIT,
        "sparkflash-cli is {total} lines, over the {CLI_SOURCE_LINE_LIMIT} gate"
    );
    assert!(
        !cli.join("tp4_forward.rs").exists(),
        "DeepSeek's device forward is still in sparkflash-cli"
    );
    assert!(
        root.join("crates/sparkflash-deepseek-forward/src/tp4_forward.rs")
            .is_file(),
        "the DeepSeek device forward crate is missing"
    );
}

#[test]
fn each_released_vision_tower_is_a_stack_of_named_components() {
    use sparkflash_model::stack::StackKind;
    use sparkflash_model::{Part, Target};
    use sparkflash_ops::scan::{ScanInputs, scan};

    const DEEPSEEK: &str =
        include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
    const INKLING: &str =
        include_str!("../../sparkflash-model-inkling2/config/inkling2_small.hf.json");
    const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
    const QWEN: &str =
        include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");

    for (name, config, rows, resolved) in [
        ("deepseek", DEEPSEEK, 7, true),
        ("inkling", INKLING, 6, true),
        ("glm", GLM, 7, false),
        ("qwen", QWEN, 6, false),
    ] {
        let report = scan(ScanInputs {
            config,
            tensors: None,
            target: Target::Sm121,
            world: 1,
        });
        assert!(report.error.is_none(), "{name}: {:?}", report.error);
        let vision: Vec<_> = report
            .rows
            .iter()
            .filter(|row| row.stack == StackKind::Vision)
            .collect();
        assert_eq!(
            vision.len(),
            rows,
            "{name}: {}\n{}",
            vision.len(),
            report.render()
        );
        for row in &vision {
            assert!(
                row.status.is_resolved() || row.status.fix().is_some(),
                "{name}: {} has no fix: {}",
                row.part,
                report.render()
            );
        }
        assert_eq!(
            report
                .rows
                .iter()
                .filter(|row| row.stack == StackKind::Vision)
                .all(|row| row.status.is_resolved()),
            resolved,
            "{name}:\n{}",
            report.render()
        );
    }

    // The old opaque part is only the stack's name; it has no catalogue entry of its own.
    assert_eq!(
        sparkflash_ops::parts::kernels(Part::Vision, Target::Sm121).count(),
        0
    );
    assert!(sparkflash_ops::parts::kernels(Part::VisionPatchEmbed, Target::Sm121).count() > 0);
}
