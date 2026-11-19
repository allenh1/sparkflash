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

//! The build identity a served card and the boot line report.
//!
//! `build.rs` embeds the commit and a dirty marker at compile time. `/v1/models` reports the
//! result as `build.id`, and the serve boot line names the same value, so a capture recorded
//! from a running server can say which build produced it. `build.version` is the package
//! version, unchanged.

/// The commit `build.rs` embedded, or `unknown` when the build ran without a git HEAD.
pub const COMMIT: &str = env!("SPARKFLASH_BUILD_COMMIT");

/// `1` when the source tree was dirty at build time, `0` otherwise.
pub const DIRTY: &str = env!("SPARKFLASH_BUILD_DIRTY");

/// The value a build with no discoverable git revision reports.
pub const UNKNOWN: &str = "unknown";

/// The build's identity: the embedded commit, with `-dirty` when the tree was dirty at build
/// time, or [UNKNOWN] when the build could not name a revision.
#[must_use]
pub fn id() -> String {
    if DIRTY == "1" && COMMIT != UNKNOWN {
        format!("{COMMIT}-dirty")
    } else {
        COMMIT.to_owned()
    }
}

/// The package version, which is what `/v1/models` reports as `build.version`.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_commit_is_a_revision_or_unknown() {
        let id = id();
        assert!(!id.is_empty());
        if COMMIT == UNKNOWN {
            assert_eq!(id, UNKNOWN);
            return;
        }
        let commit = id.strip_suffix("-dirty").unwrap_or(&id);
        assert_eq!(commit, COMMIT);
        assert!(
            (7..=64).contains(&commit.len()),
            "the embedded commit {commit:?} is not a git revision"
        );
        assert!(
            commit.chars().all(|c| c.is_ascii_hexdigit()),
            "the embedded commit {commit:?} is not hexadecimal"
        );
        assert_eq!(id.ends_with("-dirty"), DIRTY == "1");
    }

    #[test]
    fn the_build_version_is_the_package_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }
}
