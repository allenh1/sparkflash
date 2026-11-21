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

//! The one place on a node where SparkFlash keeps what every checkout shares.
//!
//! Each checkout keeps its own build, logs and telemetry. The control plane's configuration, the
//! converted rank files and the launch scripts' site file live under one cache directory, so a
//! second checkout of the repository runs against the same environment and the same converted
//! weights, with nothing copied:
//!
//! ```text
//! ~/.cache/sparkflash/sparkflash.toml           the control plane's configuration
//! ~/.cache/sparkflash/site.env                  the launch scripts' site settings
//! ~/.cache/sparkflash/converted/<scope>/        one checkpoint's rank-R-of-W.bin files
//! ```
//!
//! The path is spelled with `~` so the configuration names the same place on every node, and each
//! node expands it against its own home ([expand_home]).

use std::path::{Path, PathBuf};

/// The shared directory, as the configuration spells it for every node.
pub const CACHE_DIR: &str = "~/.cache/sparkflash";

/// The control plane's configuration under [CACHE_DIR].
pub const CONFIG_FILE: &str = "~/.cache/sparkflash/sparkflash.toml";

/// The root of the converted rank files under [CACHE_DIR]; each checkpoint gets its own child
/// ([converted_scope]).
pub const CONVERTED_DIR: &str = "~/.cache/sparkflash/converted";

/// `path` with a leading `~` or `~/` replaced by `$HOME`; anything else unchanged.
#[must_use]
pub fn expand_home(path: &str) -> PathBuf {
    let home = || std::env::var_os("HOME").map(PathBuf::from);
    if path == "~" {
        return home().unwrap_or_else(|| PathBuf::from(path));
    }
    match (path.strip_prefix("~/"), home()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

/// The converted directory for one checkpoint: `name`, then `--` and a hash of the resolved
/// checkpoint path, unless `name` already ends in that marker.
///
/// `name` is the hub folder (`models--org--repo`) for a hub model, otherwise a sanitized name for
/// the checkpoint. Hashing the resolved path gives a re-download, another revision or a
/// same-named explicit checkpoint its own directory, so one checkpoint never silently reads
/// another's rank files.
#[must_use]
pub fn converted_scope(name: &str, checkpoint: &Path) -> String {
    let mut scope = sanitize_scope(name);
    if scope.is_empty() {
        scope.push_str("model");
    }
    let marker = short_hash(&checkpoint.to_string_lossy());
    let suffix = format!("--{marker}");
    if !scope
        .to_ascii_lowercase()
        .ends_with(&suffix.to_ascii_lowercase())
    {
        scope.push_str(&suffix);
    }
    scope
}

/// This node's converted directory for `checkpoint` (named `name`, as [converted_scope] takes it):
/// [CONVERTED_DIR] expanded, then the scope of the checkpoint's absolute path. `sparkflash
/// convert` writes here when no `--output-dir` names another, where the console looks for it.
#[must_use]
pub fn converted_dir(name: &str, checkpoint: &Path) -> PathBuf {
    let checkpoint = std::path::absolute(checkpoint).unwrap_or_else(|_| checkpoint.to_path_buf());
    expand_home(CONVERTED_DIR).join(converted_scope(name, &checkpoint))
}

/// `value` kept to `[A-Za-z0-9._-]`, every other run of characters one `-`, with no leading or
/// trailing `-` or `.`.
#[must_use]
pub fn sanitize_scope(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut last_was_dash = false;
    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
            result.push(character);
            last_was_dash = false;
        } else if !last_was_dash {
            result.push('-');
            last_was_dash = true;
        }
    }
    result.trim_matches(['-', '.']).to_owned()
}

/// The 16-hex-digit marker of a checkpoint path. The directories already on the nodes carry this
/// exact spelling, so the hash must not change.
fn short_hash(value: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scope_keeps_the_marker_the_nodes_already_hold() {
        // A placeholder home; the marker of this spelling is pinned so the hash cannot drift.
        let checkpoint =
            Path::new("/home/user/.cache/huggingface/hub/models--org--repo/snapshots/abc");
        let scope = converted_scope("models--org--repo", checkpoint);
        assert_eq!(
            scope,
            format!(
                "models--org--repo--{}",
                short_hash(&checkpoint.to_string_lossy())
            )
        );
        assert_eq!(
            converted_scope(&scope, checkpoint),
            scope,
            "a marked name is not marked again"
        );
        assert_eq!(
            converted_scope("", checkpoint).split("--").next(),
            Some("model")
        );
        assert_eq!(sanitize_scope(" a/b c "), "a-b-c");
    }

    #[test]
    fn the_shared_paths_sit_under_one_directory() {
        for path in [CONFIG_FILE, CONVERTED_DIR] {
            assert!(path.starts_with(CACHE_DIR), "{path}");
        }
        assert!(expand_home(CACHE_DIR).is_absolute() || std::env::var_os("HOME").is_none());
    }
}
