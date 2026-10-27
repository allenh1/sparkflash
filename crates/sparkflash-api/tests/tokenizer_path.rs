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

//! The deployed tokenizer default must not depend on the build host's checkout path (M2 window run
//! of 2026-10-18: the binary was built in a /tmp directory and copied to a node, and the default
//! resolved to the build-time CARGO_MANIFEST_DIR, which does not exist there).

use std::fs;
use std::path::Path;

use sparkflash_model_deepseek_v41::encoding::{
    reference_tokenizer_path, resolve_reference_tokenizer,
};

/// Puts the vendored relative path under `root` so the resolver sees a staged file.
fn stage(root: &Path) -> std::path::PathBuf {
    let path = root.join("reference/tokenizer.json");
    fs::create_dir_all(path.parent().expect("the staged file has a parent"))
        .expect("the staged directory is creatable");
    fs::write(&path, "{}").expect("the staged file is writable");
    path
}

#[test]
fn the_executable_directory_wins_over_the_cwd() {
    let exe = tempfile::tempdir().expect("a temporary exe directory");
    let cwd = tempfile::tempdir().expect("a temporary cwd");
    let staged = stage(exe.path());
    // A different file in the cwd proves the first candidate is the one that was chosen.
    stage(cwd.path());

    let resolved = resolve_reference_tokenizer(Some(exe.path()), Some(cwd.path()))
        .expect("the staged tokenizer is found");
    assert_eq!(resolved, staged);
    assert_ne!(
        resolved,
        reference_tokenizer_path(),
        "a deployed binary must not fall back to the build host's checkout path"
    );
}

#[test]
fn the_cwd_is_the_fallback_when_the_executable_directory_has_nothing() {
    let exe = tempfile::tempdir().expect("a temporary exe directory");
    let cwd = tempfile::tempdir().expect("a temporary cwd");
    let staged = stage(cwd.path());

    let resolved = resolve_reference_tokenizer(Some(exe.path()), Some(cwd.path()))
        .expect("the cwd tokenizer is found");
    assert_eq!(resolved, staged);
}

#[test]
fn neither_candidate_is_an_error_that_names_the_flag() {
    let exe = tempfile::tempdir().expect("a temporary exe directory");
    let cwd = tempfile::tempdir().expect("a temporary cwd");
    let error = resolve_reference_tokenizer(Some(exe.path()), Some(cwd.path()))
        .expect_err("nothing is staged, so the default is an error");
    assert!(error.contains("--tokenizer"), "{error}");
    assert!(
        error.contains("reference/tokenizer.json"),
        "the error names the file that was looked for: {error}"
    );
    // Both candidates are reported so an operator knows exactly what to stage.
    assert!(error.contains(&exe.path().display().to_string()), "{error}");
    assert!(error.contains(&cwd.path().display().to_string()), "{error}");
}

#[test]
fn missing_process_directories_are_reported_like_any_other_miss() {
    let error = resolve_reference_tokenizer(None, None)
        .expect_err("with no candidate directories there is nothing to find");
    assert!(error.contains("--tokenizer"), "{error}");
}
