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

//! The seal on `FusedArm`: the runtime owns every strategy, and a model crate supplies a walk.

use std::path::{Path, PathBuf};

/// Every `.rs` file under `root`, in a stable order.
fn rust_files(root: &Path, into: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap_or_else(|error| panic!("{}: {error}", root.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_files(&path, into);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            into.push(path);
        }
    }
}

/// The workspace's crates directory: the test runs from `crates/sparkflash-runtime`.
fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the runtime crate is under crates/")
        .to_path_buf()
}

/// `impl ... FusedArm for ...`, the only way to write a serving strategy.
fn implements_fused_arm(line: &str) -> bool {
    let line = line.trim_start();
    !line.starts_with("//")
        && line.contains("impl")
        && line.contains("FusedArm for")
        && !line.contains("//")
}

#[test]
fn only_the_runtime_implements_fused_arm() {
    let mut files = Vec::new();
    rust_files(&crates_dir(), &mut files);
    let mut implementors = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file)
            .unwrap_or_else(|error| panic!("{}: {error}", file.display()));
        for (number, line) in text.lines().enumerate() {
            if implements_fused_arm(line) {
                implementors.push((
                    file.strip_prefix(crates_dir())
                        .unwrap_or(file)
                        .to_path_buf(),
                    number + 1,
                    line.trim().to_owned(),
                ));
            }
        }
    }
    assert!(
        !implementors.is_empty(),
        "the runtime's own strategies should implement FusedArm"
    );
    for (file, number, line) in &implementors {
        assert!(
            file.starts_with("sparkflash-runtime"),
            "{}:{number} implements FusedArm outside the runtime: {line}",
            file.display()
        );
        assert!(
            *file == Path::new("sparkflash-runtime/src/fused.rs")
                || *file == Path::new("sparkflash-runtime/src/group.rs")
                || *file == Path::new("sparkflash-runtime/src/runtime.rs"),
            "{}:{number} is a strategy the runtime does not own: {line}",
            file.display()
        );
    }
}

#[test]
fn a_model_crate_implements_a_walk_not_a_strategy() {
    for crate_dir in std::fs::read_dir(crates_dir())
        .expect("the crates directory")
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("sparkflash-model-"))
        })
    {
        let mut files = Vec::new();
        rust_files(&crate_dir, &mut files);
        for file in files {
            let text = std::fs::read_to_string(&file)
                .unwrap_or_else(|error| panic!("{}: {error}", file.display()));
            for (number, line) in text.lines().enumerate() {
                assert!(
                    !implements_fused_arm(line),
                    "{}:{} implements FusedArm; a model crate supplies a Walk: {line}",
                    file.display(),
                    number + 1
                );
            }
        }
    }
}
