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

//! Embeds the commit this crate was built from, so a served card can name its build.
//!
//! The build runs git in this crate's source tree. A missing git, a source tree without a HEAD,
//! and a repository that cannot be read all fall back to `unknown`, because a build must not fail
//! over a convenience the card reports rather than requires.

use std::path::Path;
use std::process::Command;

/// Runs git in `root`, returning its trimmed stdout only when the command succeeds.
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let root = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"),
    );
    // A new commit or a staged change must rerun this script. Emitting a rerun directive replaces
    // Cargo's default package scan, so name the package and both files the identity depends on.
    println!("cargo:rerun-if-changed=.");
    if let Some(git_dir) = git(&root, &["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
        println!("cargo:rerun-if-changed={git_dir}/index");
    }
    let commit = git(&root, &["rev-parse", "HEAD"])
        .filter(|commit| !commit.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    let dirty = if commit == "unknown" {
        "0"
    } else {
        match git(
            &root,
            &["status", "--porcelain", "--untracked-files=normal"],
        ) {
            Some(status) if !status.is_empty() => "1",
            Some(_) => "0",
            None => "0",
        }
    };
    println!("cargo:rustc-env=SPARKFLASH_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=SPARKFLASH_BUILD_DIRTY={dirty}");
}
