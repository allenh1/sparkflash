//! Hugging Face hub cache lookup: from a model id to the checkpoint directory.
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
//! `huggingface_hub` uses. A model id therefore names its checkpoint without a configured path,
//! and listing the `models--*` folders lists every model a node can serve.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// The `config.json` `model_type` values the supported weights load.
pub const SUPPORTED_MODEL_TYPES: &[&str] = &["deepseek_v41"];

/// The hub cache directory from this process's environment, as `huggingface_hub` resolves it.
#[must_use]
pub fn default_cache_dir() -> Option<PathBuf> {
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
fn folder_id(folder: &str) -> Option<String> {
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

fn snapshot_of(repo: &Path) -> Option<Snapshot> {
    let snapshots = repo.join("snapshots");
    if let Ok(text) = std::fs::read_to_string(repo.join("refs").join("main")) {
        let revision = text.trim().to_owned();
        let path = snapshots.join(&revision);
        if !revision.is_empty() && path.is_dir() {
            return Some(Snapshot { path, revision });
        }
    }
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .max_by_key(|entry| entry.metadata().and_then(|meta| meta.modified()).ok())
        .map(|entry| Snapshot {
            path: entry.path(),
            revision: entry.file_name().to_string_lossy().into_owned(),
        })
}

/// One model found in the cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CachedModel {
    /// The Hugging Face model id.
    pub id: String,
    /// The snapshot's commit hash.
    pub revision: String,
    /// The snapshot directory.
    pub snapshot: String,
    /// `config.json`'s `model_type`, when the snapshot has one.
    pub model_type: Option<String>,
    /// Whether this build's engine loads the model type.
    pub supported: bool,
    /// Whether the snapshot carries `tokenizer.json`.
    pub has_tokenizer: bool,
    /// The bytes downloaded so far.
    pub bytes: u64,
    /// False while a download left `.incomplete` blobs behind.
    pub complete: bool,
}

/// Every model in `cache` that has a snapshot, sorted by id.
///
/// # Errors
///
/// A message when the cache directory cannot be read.
pub fn list(cache: &Path) -> Result<Vec<CachedModel>, String> {
    let entries = std::fs::read_dir(cache).map_err(|error| {
        format!(
            "cannot read the Hugging Face cache at {}: {error}",
            cache.display()
        )
    })?;
    let mut models = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let folder = entry.file_name().to_string_lossy().into_owned();
        let Some(id) = folder_id(&folder) else {
            continue;
        };
        let repo = entry.path();
        let Some(snapshot) = snapshot_of(&repo) else {
            continue;
        };
        let model_type = std::fs::read(snapshot.path.join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|config| config["model_type"].as_str().map(str::to_owned));
        let (bytes, complete) = blob_totals(&repo.join("blobs"));
        models.push(CachedModel {
            supported: model_type
                .as_deref()
                .is_some_and(|kind| SUPPORTED_MODEL_TYPES.contains(&kind)),
            has_tokenizer: snapshot.path.join("tokenizer.json").is_file(),
            snapshot: snapshot.path.display().to_string(),
            revision: snapshot.revision,
            id,
            model_type,
            bytes,
            complete,
        });
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

fn blob_totals(blobs: &Path) -> (u64, bool) {
    let Ok(entries) = std::fs::read_dir(blobs) else {
        return (0, true);
    };
    let mut bytes = 0;
    let mut complete = true;
    for entry in entries.filter_map(Result::ok) {
        if entry.file_name().to_string_lossy().ends_with(".incomplete") {
            complete = false;
        }
        bytes += entry.metadata().map_or(0, |meta| meta.len());
    }
    (bytes, complete)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Writes a cached model the way `huggingface_hub` does, and returns its snapshot path.
    pub(crate) fn cache_model(cache: &Path, id: &str, revision: &str, model_type: &str) -> PathBuf {
        let repo = cache.join(repo_folder(id));
        let snapshot = repo.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        std::fs::create_dir_all(repo.join("refs")).expect("refs dir");
        std::fs::create_dir_all(repo.join("blobs")).expect("blobs dir");
        std::fs::write(repo.join("refs").join("main"), format!("{revision}\n")).expect("ref");
        let config = format!("{{\"model_type\": \"{model_type}\"}}");
        std::fs::write(repo.join("blobs").join("config-blob"), &config).expect("blob");
        std::fs::write(snapshot.join("config.json"), &config).expect("config");
        std::fs::write(snapshot.join("tokenizer.json"), "{}").expect("tokenizer");
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
        let snapshot = cache_model(dir.path(), "org/model", "abc123", "deepseek_v41");
        // A second, newer snapshot that main does not point at must not win.
        std::fs::create_dir_all(snapshot.parent().expect("snapshots").join("fff999"))
            .expect("other snapshot");
        let resolved = resolve(dir.path(), "org/model").expect("resolves");
        assert_eq!(resolved.path, snapshot);
        assert_eq!(resolved.revision, "abc123");
    }

    #[test]
    fn resolve_uses_the_only_snapshot_without_a_ref() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = cache_model(dir.path(), "org/model", "abc123", "deepseek_v41");
        std::fs::remove_file(dir.path().join("models--org--model/refs/main")).expect("drop ref");
        assert_eq!(
            resolve(dir.path(), "org/model").expect("resolves").path,
            snapshot
        );
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
    }

    #[test]
    fn list_reports_every_cached_model_and_whether_it_is_supported() {
        let dir = tempfile::tempdir().expect("tempdir");
        cache_model(dir.path(), "org/flash", "abc123", "deepseek_v41");
        cache_model(dir.path(), "other/llama", "def456", "llama");
        std::fs::write(
            dir.path()
                .join("models--other--llama/blobs/part.incomplete"),
            "xx",
        )
        .expect("partial blob");
        std::fs::create_dir_all(dir.path().join("models--empty--repo")).expect("no snapshot");
        std::fs::create_dir_all(dir.path().join("datasets--org--data/snapshots/a"))
            .expect("dataset");
        let models = list(dir.path()).expect("list");
        let ids: Vec<&str> = models.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(ids, ["org/flash", "other/llama"]);
        assert!(models[0].supported);
        assert!(models[0].complete);
        assert!(models[0].has_tokenizer);
        assert_eq!(models[0].revision, "abc123");
        assert_eq!(models[0].model_type.as_deref(), Some("deepseek_v41"));
        assert!(models[0].bytes > 0);
        assert!(!models[1].supported);
        assert!(!models[1].complete);
    }
}
