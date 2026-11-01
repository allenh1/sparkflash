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

//! The Hugging Face hub cache layout, for the packs this crate loads.
//!
//! The cache is the canonical home of a downloaded checkpoint:
//!
//! ```text
//! <cache>/models--<org>--<name>/refs/main          commit hash main points at
//! <cache>/models--<org>--<name>/snapshots/<hash>/  the checkout
//! ```
//!
//! A pack resolves its own repository id here instead of making the operator copy a snapshot
//! path into an environment variable. The lookup is deliberately one repo: a model id maps to one
//! directory, and reading `refs/main` is enough to find the snapshot (the newest snapshot is the
//! fallback for a cache written without a ref).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The hub cache directory from this process's environment, in `huggingface_hub`'s order.
#[must_use]
pub fn cache_dir() -> Option<PathBuf> {
    cache_dir_from(|key| std::env::var_os(key))
}

fn cache_dir_from(var: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    let set = |key: &str| {
        var(key)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    if let Some(cache) = set("HF_HUB_CACHE").or_else(|| set("HUGGINGFACE_HUB_CACHE")) {
        return Some(cache);
    }
    if let Some(home) = set("HF_HOME") {
        return Some(home.join("hub"));
    }
    if let Some(cache) = set("XDG_CACHE_HOME") {
        return Some(cache.join("huggingface").join("hub"));
    }
    set("HOME").map(|home| home.join(".cache").join("huggingface").join("hub"))
}

/// The snapshot directory for `repo_id` in the process's hub cache.
#[must_use]
pub fn cached_model_dir(repo_id: &str) -> Option<PathBuf> {
    cache_dir().and_then(|cache| cached_model_in(&cache, repo_id))
}

/// The snapshot directory for `repo_id` in `cache`.
///
/// `refs/main` wins; a cache with no ref falls back to the newest snapshot. A missing repository
/// or snapshot is `None`, which lets a caller fall back to its own default or report a cache miss.
#[must_use]
pub fn cached_model_in(cache: &Path, repo_id: &str) -> Option<PathBuf> {
    let folder = repo_folder(repo_id)?;
    let repo = cache.join(folder);
    if let Ok(revision) = std::fs::read_to_string(repo.join("refs").join("main")) {
        let revision = revision.trim();
        let snapshot = repo.join("snapshots").join(revision);
        if !revision.is_empty() && snapshot.is_dir() {
            return Some(snapshot);
        }
    }
    std::fs::read_dir(repo.join("snapshots"))
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .max_by_key(|entry| entry.metadata().and_then(|meta| meta.modified()).ok())
        .map(|entry| entry.path())
}

/// `org/name` as the `models--org--name` cache folder.
fn repo_folder(repo_id: &str) -> Option<String> {
    let mut parts = repo_id.split('/');
    let org = parts.next()?;
    let name = parts.next()?;
    if parts.next().is_some()
        || org.is_empty()
        || name.is_empty()
        || repo_id.contains("..")
        || repo_id.contains("--")
    {
        return None;
    }
    Some(format!("models--{org}--{name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_directory_follows_the_huggingface_hub_order() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |key: &str| {
                pairs
                    .iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, value)| OsString::from(value))
            }
        };
        static ALL: [(&str, &str); 4] = [
            ("HF_HUB_CACHE", "/hub"),
            ("HF_HOME", "/hf"),
            ("XDG_CACHE_HOME", "/xdg"),
            ("HOME", "/home/user"),
        ];
        assert_eq!(cache_dir_from(env(&ALL)), Some(PathBuf::from("/hub")));
        assert_eq!(
            cache_dir_from(env(&ALL[1..])),
            Some(PathBuf::from("/hf/hub"))
        );
        assert_eq!(
            cache_dir_from(env(&ALL[2..])),
            Some(PathBuf::from("/xdg/huggingface/hub"))
        );
        assert_eq!(
            cache_dir_from(env(&ALL[3..])),
            Some(PathBuf::from("/home/user/.cache/huggingface/hub"))
        );
    }

    #[test]
    fn a_snapshot_resolves_through_the_ref_or_the_newest_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("models--org--model");
        let old = repo.join("snapshots").join("old");
        let new = repo.join("snapshots").join("new");
        std::fs::create_dir_all(&new).expect("new");
        assert_eq!(
            cached_model_in(dir.path(), "org/model"),
            Some(new),
            "no ref uses the newest snapshot"
        );
        std::fs::create_dir_all(&old).expect("old");
        std::fs::create_dir_all(repo.join("refs")).expect("refs");
        std::fs::write(repo.join("refs").join("main"), "old").expect("ref");
        assert_eq!(cached_model_in(dir.path(), "org/model"), Some(old));
        assert!(cached_model_in(dir.path(), "org/absent").is_none());
        assert!(cached_model_in(dir.path(), "not-an-id").is_none());
    }
}
