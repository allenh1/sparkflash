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

//! A model crate does not look a kernel up by string.
//!
//! The catalogue and the resolver name the image and symbol a part runs; an arm that calls
//! `load_named`/`load_named_shared` itself has chosen a kernel outside that table, and a renamed
//! or misfiled symbol then fails at boot rather than in review. This test reads the tree and
//! holds every such site to the table below.
//!
//! Phase 7 converts one arm at a time. [EXEMPTIONS] is the measured list: one line per file,
//! its site count and why the string is still there. A conversion deletes lines and the test
//! fails until the table agrees, so the ban tightens with each arm rather than drifting. The
//! remaining lines are DeepSeek's and the two loaders each converted arm keeps in Bound.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The kernel-lookup calls a model crate may not make once it is converted.
const LOOKUPS: &[&str] = &["load_named(", "load_named_shared(", "load_function("];

/// One file's recorded string lookups: the crate, the path under it, the number of sites, and
/// why each is still a string lookup.
struct Exemption {
    crate_name: &'static str,
    file: &'static str,
    sites: usize,
    why: &'static str,
}

/// Every string kernel lookup in a model crate at this tip.
///
/// Every model arm now loads through the catalogue-bound handle in `sparkflash-ops`, so no
/// model crate holds a string lookup. The DeepSeek forward is the one remaining caller; phase 7
/// converts it and deletes its lines.
const EXEMPTIONS: &[Exemption] = &[
    Exemption {
        crate_name: "sparkflash-deepseek-forward",
        file: "src/tp4_forward.rs",
        sites: 13,
        why: "not converted yet: phase 7 runs by arm",
    },
    Exemption {
        crate_name: "sparkflash-deepseek-forward",
        file: "src/tp4_forward/rows.rs",
        sites: 3,
        why: "not converted yet: phase 7 runs by arm",
    },
    Exemption {
        crate_name: "sparkflash-deepseek-forward",
        file: "src/moe_device.rs",
        sites: 1,
        why: "the row-scaled expert body is loaded by symbol until the catalogue names it",
    },
    Exemption {
        crate_name: "sparkflash-deepseek-forward",
        file: "src/nvfp4_prefill.rs",
        sites: 8,
        why: "the NVFP4 wrappers are new bodies the catalogue does not name yet; each loads its own cubin symbol",
    },
    Exemption {
        crate_name: "sparkflash-model-qwen4-exp",
        file: "examples/nvfp4_expert_qwen.rs",
        sites: 4,
        why: "a single-operation check loads the cubin it measures by symbol, not through a serving arm",
    },
    Exemption {
        crate_name: "sparkflash-model-inkling2",
        file: "examples/nvfp4_expert_inkling.rs",
        sites: 3,
        why: "a single-operation check loads the cubin it measures by symbol, not through a serving arm",
    },
    Exemption {
        crate_name: "sparkflash-deepseek-forward",
        file: "src/tp4_forward/vision.rs",
        sites: 1,
        why: "not converted yet: phase 7 runs by arm",
    },
    Exemption {
        crate_name: "sparkflash-deepseek-forward",
        file: "src/tp4_forward/draft.rs",
        sites: 1,
        why: "not converted yet: phase 7 runs by arm",
    },
    Exemption {
        crate_name: "sparkflash-deepseek-forward",
        file: "src/tp4_forward/bulk.rs",
        sites: 1,
        why: "not converted yet: phase 7 runs by arm",
    },
    Exemption {
        crate_name: "sparkflash-deepseek-forward",
        file: "src/tp4_tail.rs",
        sites: 1,
        why: "not converted yet: phase 7 runs by arm",
    },
];

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root")
}

/// Every `.rs` file under `root`, relative to it.
fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files
}

/// Each model crate's source, and the lookup count per file.
fn lookups() -> BTreeMap<(String, String), usize> {
    let workspace = workspace();
    let crates = workspace.join("crates");
    let mut found = BTreeMap::new();
    let mut names: Vec<(String, PathBuf)> = std::fs::read_dir(&crates)
        .expect("the crates directory")
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("sparkflash-model-") || name == "sparkflash-deepseek-forward" {
                Some((name, entry.path()))
            } else {
                None
            }
        })
        .collect();
    names.sort();
    for (name, directory) in names {
        for path in rust_files(&directory) {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let sites: usize = text
                .lines()
                .filter(|line| LOOKUPS.iter().any(|lookup| line.contains(lookup)))
                .count();
            if sites == 0 {
                continue;
            }
            let relative = path
                .strip_prefix(&directory)
                .expect("a file under its crate")
                .to_string_lossy()
                .replace('\\', "/");
            // A test builds its own operand fixture and is not a serving launch site: it may load
            // the body it measures directly. The ban is about the arm's boot table and launches.
            if relative.starts_with("tests/") {
                continue;
            }
            found.insert((name.clone(), relative), sites);
        }
    }
    found
}

#[test]
fn a_converted_model_crate_does_not_look_a_kernel_up_by_string() {
    let found = lookups();
    let mut table: BTreeMap<(String, String), &Exemption> = BTreeMap::new();
    for exemption in EXEMPTIONS {
        assert!(
            !exemption.why.trim().is_empty(),
            "{} has an exemption with no reason",
            exemption.crate_name
        );
        assert!(exemption.sites > 0, "{} names no site", exemption.file);
        table.insert(
            (exemption.crate_name.to_owned(), exemption.file.to_owned()),
            exemption,
        );
    }

    let mut problems = Vec::new();
    for ((crate_name, file), sites) in &found {
        match table.get(&(crate_name.clone(), file.clone())) {
            None => problems.push(format!(
                "{crate_name}/{file} looks a kernel up by string at {sites} site(s), and the \
                 exemption table does not name it: convert them to a bound handle or add a line \
                 with the count and the reason"
            )),
            Some(recorded) if recorded.sites != *sites => problems.push(format!(
                "{crate_name}/{file} has {sites} string lookup(s), and the table records {} \
                 ({}): tighten the table as the arm converts",
                recorded.sites, recorded.why
            )),
            Some(_) => {}
        }
    }
    for ((crate_name, file), recorded) in &table {
        if !found.contains_key(&(crate_name.clone(), file.clone())) {
            problems.push(format!(
                "{crate_name}/{file} records {} string lookup(s), and the tree has none: delete \
                 the line",
                recorded.sites
            ));
        }
        let path = workspace().join("crates").join(crate_name).join(file);
        assert!(path.is_file(), "{} does not exist", path.display());
    }
    assert!(
        problems.is_empty(),
        "the string-lookup ban has drifted from the tree:\n  {}",
        problems.join("\n  ")
    );
}

#[test]
fn a_converted_arm_holds_no_string_lookup() {
    let found = lookups();
    for arm in [
        "sparkflash-model-dense",
        "sparkflash-model-glm5-next",
        "sparkflash-model-qwen4-exp",
        "sparkflash-model-inkling2",
        "sparkflash-model-dory",
    ] {
        // A file under examples/ is a check tool, not a serving launch: it loads the cubin it
        // measures by name and holds no handle of its own. The ban is on launches.
        assert!(
            !found
                .keys()
                .any(|(crate_name, file)| crate_name == arm && !file.starts_with("examples/")),
            "{arm} looks a kernel up by string; its launches must hold a bound handle"
        );
    }
}
