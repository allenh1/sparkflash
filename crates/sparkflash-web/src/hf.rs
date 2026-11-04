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
//! The cache directory and the id-to-snapshot resolution are [sparkflash_weights::hub]'s, the
//! one resolver the serving binary and every model pack use too. This module adds what only the
//! control plane needs: listing the `models--*` folders, which lists every model a node can serve,
//! and choosing among them.

use std::path::Path;

use serde::Serialize;
pub use sparkflash_weights::hub::{default_cache_dir, is_model_id, repo_folder, resolve};
use sparkflash_weights::hub::{folder_id, snapshot_of};

use crate::config::{profile_for_id, profile_for_model_type};

/// The `config.json` `model_type` values the supported weights load.
pub const SUPPORTED_MODEL_TYPES: &[&str] =
    &["deepseek_v41", "qwen4_exp", "qwen4_exp_text", "glm5_next"];

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
    /// The control-plane weight selector that runs this checkpoint, when supported.
    pub weights: Option<String>,
    /// The engine selector that runs this checkpoint, when supported.
    pub engine: Option<String>,
    /// The checkpoint's context ceiling, when supported.
    pub context_tokens: Option<usize>,
    /// Whether a launch needs per-rank converted weights.
    pub requires_conversion: Option<bool>,
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
        if let Some(model) = read_model(&entry.path(), &id) {
            models.push(model);
        }
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

/// The best complete cached model for a weight family, preferring the canonical repository and
/// then the closest name to it.
///
/// This is what the console selects on a fresh checkout: a machine that already has
/// `org/model-1017` should not offer to download `org/model` again.
#[must_use]
pub fn preferred_cached<'a>(
    models: &'a [CachedModel],
    weights: &str,
    canonical_id: &str,
) -> Option<&'a CachedModel> {
    let rank = |model: &CachedModel| {
        if model.id == canonical_id {
            0
        } else if model.id.starts_with(canonical_id) {
            1
        } else if model.id.starts_with("deepseek-ai/") {
            2
        } else {
            3
        }
    };
    models
        .iter()
        .filter(|model| model.complete && model.supported)
        .filter(|model| model.weights.as_deref() == Some(weights))
        .min_by(|left, right| {
            (rank(left), left.id.len(), left.id.as_str()).cmp(&(
                rank(right),
                right.id.len(),
                right.id.as_str(),
            ))
        })
}

/// The cached entry for one model id, if its snapshot exists.
///
/// This is the cheap targeted lookup the control plane uses before offering a download: it reads
/// one repo folder, not every model in the cache.
#[must_use]
pub fn lookup(cache: &Path, id: &str) -> Option<CachedModel> {
    if !is_model_id(id) {
        return None;
    }
    read_model(&cache.join(repo_folder(id)), id)
}

fn read_model(repo: &Path, id: &str) -> Option<CachedModel> {
    let snapshot = snapshot_of(repo)?;
    let model_type = std::fs::read(snapshot.path.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|config| config["model_type"].as_str().map(str::to_owned));
    let (bytes, complete) = blob_totals(&repo.join("blobs"));
    // An in-progress download may not have config.json yet. A canonical repository id still
    // identifies the family, so the selected model does not appear unsupported mid-download.
    let profile = match model_type.as_deref() {
        Some(kind) => profile_for_model_type(kind),
        None => profile_for_id(id),
    };
    Some(CachedModel {
        supported: profile.is_some(),
        weights: profile.map(|profile| profile.weights.to_owned()),
        engine: profile.map(|profile| profile.engine.to_owned()),
        context_tokens: profile.map(|profile| profile.context_tokens),
        requires_conversion: profile.map(|profile| profile.requires_conversion),
        has_tokenizer: snapshot.path.join("tokenizer.json").is_file(),
        snapshot: snapshot.path.display().to_string(),
        revision: snapshot.revision,
        id: id.to_owned(),
        model_type,
        bytes,
        complete,
    })
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
    use std::path::PathBuf;

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
    fn a_cached_snapshot_wins_over_an_absent_canonical_repo() {
        let dir = tempfile::tempdir().expect("tempdir");
        cache_model(
            dir.path(),
            "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "aaa",
            "deepseek_v41",
        );
        cache_model(
            dir.path(),
            "pqhaz/DeepSeek-V4.1-Flash-1017-heretic",
            "bbb",
            "deepseek_v41",
        );
        cache_model(dir.path(), "other/unrelated", "ccc", "llama");
        let models = list(dir.path()).expect("list");
        let preferred = preferred_cached(
            &models,
            crate::config::DEFAULT_WEIGHTS_ID,
            crate::config::DEFAULT_MODEL_ID,
        )
        .expect("a cached model");
        assert_eq!(preferred.id, "deepseek-ai/DeepSeek-V4.1-Flash-1017");
    }

    #[test]
    fn a_canonical_model_is_supported_while_its_config_is_still_downloading() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = cache_model(
            dir.path(),
            crate::config::DEFAULT_MODEL_ID,
            "abc123",
            "deepseek_v41",
        );
        std::fs::remove_file(snapshot.join("config.json")).expect("drop config");
        let model = lookup(dir.path(), crate::config::DEFAULT_MODEL_ID).expect("cache entry");
        assert!(model.supported);
        assert_eq!(model.model_type, None);
        assert_eq!(
            model.weights.as_deref(),
            Some(crate::config::DEFAULT_WEIGHTS_ID)
        );
        assert_eq!(model.requires_conversion, Some(true));
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
        assert_eq!(
            models[0].weights.as_deref(),
            Some(crate::config::DEFAULT_WEIGHTS_ID)
        );
        assert_eq!(
            models[0].engine.as_deref(),
            Some(crate::config::DEFAULT_ENGINE_ID)
        );
        assert_eq!(
            models[0].context_tokens,
            Some(crate::config::MODEL_CONTEXT_TOKENS)
        );
        assert_eq!(models[0].requires_conversion, Some(true));
        assert!(models[0].bytes > 0);
        assert!(!models[1].supported);
        assert!(models[1].weights.is_none());
        assert!(!models[1].complete);

        let flash = lookup(dir.path(), "org/flash").expect("targeted lookup");
        assert_eq!(flash.id, "org/flash");
        assert!(flash.complete);
        assert_eq!(flash.revision, "abc123");
        assert!(lookup(dir.path(), "org/absent").is_none());
        assert!(lookup(dir.path(), "not an id").is_none());
    }

    #[test]
    fn a_cached_glm_checkpoint_is_selectable() {
        let dir = tempfile::tempdir().expect("tempdir");
        cache_model(
            dir.path(),
            crate::config::GLM_MODEL_ID,
            "abc123",
            "glm5_next",
        );
        let model = lookup(dir.path(), crate::config::GLM_MODEL_ID).expect("cache entry");
        assert!(model.supported);
        assert_eq!(model.model_type.as_deref(), Some("glm5_next"));
        assert_eq!(
            model.weights.as_deref(),
            Some(crate::config::GLM_WEIGHTS_ID)
        );
        assert_eq!(model.engine.as_deref(), Some(crate::config::GLM_ENGINE_ID));
        assert_eq!(
            model.context_tokens,
            Some(crate::config::GLM_CONTEXT_TOKENS)
        );
        assert_eq!(model.requires_conversion, Some(true));
    }
}
