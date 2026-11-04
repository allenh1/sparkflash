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

//! The Hugging Face hub cache: from a model id to its checkpoint directory.
//!
//! The hub cache has one fixed layout, the one `huggingface_hub` writes and every HF tool reads:
//!
//! ```text
//! <cache>/models--<org>--<name>/refs/main          the commit hash `main` points at
//! <cache>/models--<org>--<name>/snapshots/<hash>/  the checkout (symlinks into blobs/)
//! <cache>/models--<org>--<name>/blobs/             the file contents
//! ```
//!
//! The cache directory is `HF_HUB_CACHE`, else `$HF_HOME/hub`, else
//! `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`, the same order
//! `huggingface_hub` uses. A model id therefore names its checkpoint without a configured path:
//! every model pack, the serving binary and the control plane resolve it here, so they cannot
//! disagree about which snapshot an id is.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The hub cache directory from this process's environment, as `huggingface_hub` resolves it.
#[must_use]
pub fn default_cache_dir() -> Option<PathBuf> {
    cache_dir_from(|key| std::env::var_os(key))
}

/// The hub cache directory from the environment `var` reads, in `huggingface_hub`'s order. An
/// empty variable counts as unset.
#[must_use]
pub fn cache_dir_from(var: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
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

/// Whether `id` is a Hugging Face model id: `name` or `org/name`, each part letters, digits,
/// `-`, `_` or `.`, not starting or ending with `-` or `.`, and no `--` or `..` anywhere.
#[must_use]
pub fn is_model_id(id: &str) -> bool {
    let parts: Vec<&str> = id.split('/').collect();
    (1..=2).contains(&parts.len())
        && !id.contains("--")
        && !id.contains("..")
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.len() <= 96
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                && !part.starts_with(['-', '.'])
                && !part.ends_with(['-', '.'])
        })
}

/// The cache folder for a model id: `org/name` is `models--org--name`.
#[must_use]
pub fn repo_folder(id: &str) -> String {
    format!("models--{}", id.replace('/', "--"))
}

/// The model id a cache folder holds, the inverse of [`repo_folder`]. Ids never contain `--`,
/// so the mapping is unambiguous.
#[must_use]
pub fn folder_id(folder: &str) -> Option<String> {
    folder
        .strip_prefix("models--")
        .filter(|rest| !rest.is_empty())
        .map(|rest| rest.replace("--", "/"))
}

/// A model id resolved to a checkout in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// The snapshot directory, used as the checkpoint.
    pub path: PathBuf,
    /// The commit hash the snapshot is named after.
    pub revision: String,
}

/// The snapshot `id` resolves to in `cache`: the commit `refs/main` names, or the newest snapshot
/// when the download recorded no ref (a download pinned to a commit writes none).
///
/// # Errors
///
/// A message naming the cache when the model or its snapshot is not there.
pub fn resolve(cache: &Path, id: &str) -> Result<Snapshot, String> {
    if !is_model_id(id) {
        return Err(format!("{id:?} is not a Hugging Face model id"));
    }
    let repo = cache.join(repo_folder(id));
    if !repo.is_dir() {
        return Err(format!(
            "{id} is not in the Hugging Face cache at {}",
            cache.display()
        ));
    }
    snapshot_of(&repo).ok_or_else(|| format!("{id} has no snapshot in {}", repo.display()))
}

/// The snapshot of one cached repository folder: `refs/main`'s commit, else the newest snapshot.
///
/// The ref is text a download wrote, so it names a snapshot only as one plain path component (not
/// `../..` or an absolute path) that is a directory entry of `snapshots/`, not a symlink, and a
/// symlinked entry is never the newest either. The cache's own folders are followed as the
/// filesystem has them: the hub links every file of a snapshot into `blobs/`, and a cache moved to
/// another disk is often a link. Whoever can write the cache chooses what it serves, with or
/// without a link out of it, so write access to the cache is the trust boundary, not this lookup.
#[must_use]
pub fn snapshot_of(repo: &Path) -> Option<Snapshot> {
    let snapshots = repo.join("snapshots");
    if let Ok(text) = std::fs::read_to_string(repo.join("refs").join("main")) {
        let revision = text.trim().to_owned();
        let path = snapshots.join(&revision);
        if is_component(&revision)
            && std::fs::symlink_metadata(&path).is_ok_and(|meta| meta.is_dir())
        {
            return Some(Snapshot { path, revision });
        }
    }
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .max_by_key(|entry| entry.metadata().and_then(|meta| meta.modified()).ok())
        .map(|entry| Snapshot {
            path: entry.path(),
            revision: entry.file_name().to_string_lossy().into_owned(),
        })
}

/// Whether `name` is one plain path component: not empty, `.` or `..`, and no separator.
fn is_component(name: &str) -> bool {
    !matches!(name, "" | "." | "..") && !name.contains(['/', '\\'])
}

/// The snapshot directory for `id` in `cache`, or `None` when the id is not a model id or the
/// cache holds no snapshot of it, which lets a caller fall back to its own default or report the
/// miss in its own words.
#[must_use]
pub fn cached_model_in(cache: &Path, id: &str) -> Option<PathBuf> {
    if !is_model_id(id) {
        return None;
    }
    snapshot_of(&cache.join(repo_folder(id))).map(|snapshot| snapshot.path)
}

/// The snapshot directory for `id` in this process's hub cache.
#[must_use]
pub fn cached_model_dir(id: &str) -> Option<PathBuf> {
    default_cache_dir().and_then(|cache| cached_model_in(&cache, id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_snapshot(cache: &Path, id: &str, revision: &str) -> PathBuf {
        let repo = cache.join(repo_folder(id));
        let snapshot = repo.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        std::fs::create_dir_all(repo.join("refs")).expect("refs dir");
        std::fs::write(repo.join("refs").join("main"), format!("{revision}\n")).expect("ref");
        snapshot
    }

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
        assert_eq!(
            cache_dir_from(env(&[("HF_HUB_CACHE", ""), ("HOME", "/home/user")])),
            Some(PathBuf::from("/home/user/.cache/huggingface/hub"))
        );
        assert_eq!(cache_dir_from(env(&[])), None);
    }

    #[test]
    fn model_ids_map_to_cache_folders_and_back() {
        assert_eq!(
            repo_folder("deepseek-ai/DeepSeek-V4.1-Flash"),
            "models--deepseek-ai--DeepSeek-V4.1-Flash"
        );
        assert_eq!(
            folder_id("models--deepseek-ai--DeepSeek-V4.1-Flash").as_deref(),
            Some("deepseek-ai/DeepSeek-V4.1-Flash")
        );
        assert_eq!(folder_id("models--gpt2").as_deref(), Some("gpt2"));
        assert_eq!(folder_id("datasets--org--name"), None);
        assert!(is_model_id("deepseek-ai/DeepSeek-V4.1-Flash"));
        assert!(is_model_id("gpt2"));
        for bad in [
            "", "a/b/c", "a--b/c", "../x", "org/-x", "org/x.", "org/x y", "/x",
        ] {
            assert!(!is_model_id(bad), "{bad:?}");
        }
    }

    #[test]
    fn resolve_follows_refs_main_to_its_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = cache_snapshot(dir.path(), "org/model", "abc123");
        // A second, newer snapshot that main does not point at must not win.
        std::fs::create_dir_all(snapshot.parent().expect("snapshots").join("fff999"))
            .expect("other snapshot");
        let resolved = resolve(dir.path(), "org/model").expect("resolves");
        assert_eq!(resolved.path, snapshot);
        assert_eq!(resolved.revision, "abc123");
        assert_eq!(cached_model_in(dir.path(), "org/model"), Some(snapshot));
    }

    #[test]
    fn without_a_ref_the_newest_snapshot_is_used() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("models--org--model");
        let new = repo.join("snapshots").join("new");
        std::fs::create_dir_all(&new).expect("new");
        assert_eq!(
            resolve(dir.path(), "org/model").expect("resolves").path,
            new
        );
        assert_eq!(cached_model_in(dir.path(), "org/model"), Some(new));
        // A ref wins over the newest snapshot once the download records one.
        let old = repo.join("snapshots").join("old");
        std::fs::create_dir_all(&old).expect("old");
        std::fs::create_dir_all(repo.join("refs")).expect("refs");
        std::fs::write(repo.join("refs").join("main"), "old").expect("ref");
        assert_eq!(cached_model_in(dir.path(), "org/model"), Some(old));
    }

    #[test]
    fn a_ref_that_is_a_path_stays_inside_the_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = cache_snapshot(dir.path(), "org/model", "abc123");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).expect("outside");
        let refs = dir
            .path()
            .join("models--org--model")
            .join("refs")
            .join("main");
        let escapes = [
            "../../outside".to_owned(),
            outside.display().to_string(),
            "..".to_owned(),
            ".".to_owned(),
            "abc123/..".to_owned(),
            "..\\..\\outside".to_owned(),
        ];
        for escape in escapes {
            std::fs::write(&refs, &escape).expect("ref");
            assert_eq!(
                cached_model_in(dir.path(), "org/model"),
                Some(snapshot.clone()),
                "{escape:?} is a ref to no snapshot, so the newest real one is used"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_snapshot_is_never_resolved() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = cache_snapshot(dir.path(), "org/model", "abc123");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).expect("outside");
        let link = snapshot.parent().expect("snapshots").join("linked");
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");
        // Neither as refs/main's target nor as the newest entry.
        let refs = dir
            .path()
            .join("models--org--model")
            .join("refs")
            .join("main");
        std::fs::write(&refs, "linked").expect("ref");
        assert_eq!(
            cached_model_in(dir.path(), "org/model"),
            Some(snapshot.clone())
        );
        std::fs::remove_file(&refs).expect("no ref");
        assert_eq!(cached_model_in(dir.path(), "org/model"), Some(snapshot));
    }

    #[test]
    fn a_missing_model_names_the_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let error = resolve(dir.path(), "org/absent").expect_err("absent");
        assert!(error.contains("org/absent is not in the Hugging Face cache at"));
        assert!(
            resolve(dir.path(), "not an id")
                .expect_err("bad id")
                .contains("not a Hugging Face model id")
        );
        assert!(cached_model_in(dir.path(), "org/absent").is_none());
        assert!(cached_model_in(dir.path(), "/a/path").is_none());
    }
}
