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

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
pub use sparkflash_weights::hub::{default_cache_dir, is_model_id, repo_folder, resolve};
use sparkflash_weights::hub::{folder_id, snapshot_of};

use crate::config::{ModelProfile, profile_for_id, profile_for_model_type};

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

/// A model the model browser can offer: one from the local cache, one from the Hub search, or
/// the selected Hub model after its `config.json` was inspected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RemoteModel {
    /// The Hugging Face model id.
    pub id: String,
    /// Hub download count, zero for a cached-only entry.
    pub downloads: u64,
    /// Hub likes, zero for a cached-only entry.
    pub likes: u64,
    /// Whether the snapshot is in the local cache.
    pub cached: bool,
    /// Whether the cached snapshot is complete; false for a Hub-only entry.
    pub complete: bool,
    /// Whether this build can load the checkpoint.
    pub supported: bool,
    /// The control-plane weight selector, when known.
    pub weights: Option<String>,
    /// The engine selector, when known.
    pub engine: Option<String>,
    /// The checkpoint's context ceiling, when known.
    pub context_tokens: Option<usize>,
    /// Whether a launch needs per-rank converted weights, when known.
    pub requires_conversion: Option<bool>,
    /// The family label, when the model type is known.
    pub family: Option<String>,
}

/// The families the browser lists on its side, from the build's model profiles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Family {
    /// The label shown in the sidebar, for example `DeepSeek`.
    pub label: String,
    /// The default search term for the family.
    pub search: String,
    /// The organization the family's checkpoints usually live under.
    pub org: String,
    /// The family's canonical checkpoint id.
    pub id: String,
    /// The weight selector that runs it.
    pub weights: String,
    /// How many ranks the family runs on.
    pub world_size: usize,
}

/// The families this build can configure, in sidebar order.
#[must_use]
pub fn families() -> Vec<Family> {
    crate::config::MODEL_PROFILES
        .iter()
        .map(|profile| Family {
            label: profile.label.to_owned(),
            search: profile.search.to_owned(),
            org: profile.org.to_owned(),
            id: profile.id.to_owned(),
            weights: profile.weights.to_owned(),
            world_size: profile.world_size,
        })
        .collect()
}

/// A Hub model search result, before the local cache is folded in.
#[derive(Debug, Deserialize)]
struct HubHit {
    #[serde(rename = "modelId")]
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
}

/// Search the Hub and the local cache for models the browser can offer.
///
/// The local cache is searched first and wins on an id collision, because a cached model has a
/// real size, completion state and `config.json`. A Hub hit that is not cached is still offered:
/// selecting it makes the Setup download step its target. When `family` is set, an empty query
/// searches for that family's default term; a query overrides it.
///
/// # Errors
/// A message only when the Hub request fails and there is no local result to fall back to.
pub fn search(
    cache: Option<&Path>,
    query: &str,
    family: Option<&ModelProfile>,
    limit: usize,
) -> Result<Vec<RemoteModel>, String> {
    let query = query.trim();
    let local = cache.and_then(|cache| list(cache).ok()).unwrap_or_default();
    let folded = |model: &CachedModel| -> RemoteModel { cached_remote(model) };
    let matches = |model: &CachedModel| {
        let id = model.id.to_ascii_lowercase();
        let text = query.to_ascii_lowercase();
        (text.is_empty() || id.contains(&text))
            && family.is_none_or(|profile| {
                model.weights.as_deref() == Some(profile.weights)
                    || model
                        .id
                        .to_ascii_lowercase()
                        .contains(&profile.search.to_ascii_lowercase())
            })
    };
    let mut results: BTreeMap<String, RemoteModel> = BTreeMap::new();
    for model in local.iter().filter(|model| matches(model)) {
        results.insert(model.id.clone(), folded(model));
    }

    let term = if query.is_empty() {
        family.map_or("", |profile| profile.search)
    } else {
        query
    };
    if term.is_empty() {
        let mut out: Vec<RemoteModel> = results.into_values().collect();
        sort_models(&mut out);
        return Ok(out);
    }
    match hub_search(term, limit) {
        Ok(hits) => {
            for hit in hits
                .into_iter()
                .filter(|hit| family.is_none_or(|profile| family_matches(&hit.id, profile)))
            {
                results.entry(hit.id.clone()).or_insert(RemoteModel {
                    id: hit.id,
                    downloads: hit.downloads,
                    likes: hit.likes,
                    cached: false,
                    complete: false,
                    supported: false,
                    weights: None,
                    engine: None,
                    context_tokens: None,
                    requires_conversion: None,
                    family: None,
                });
            }
            let mut out: Vec<RemoteModel> = results.into_values().collect();
            sort_models(&mut out);
            Ok(out)
        }
        Err(error) if results.is_empty() => Err(error),
        Err(_) => {
            let mut out: Vec<RemoteModel> = results.into_values().collect();
            sort_models(&mut out);
            Ok(out)
        }
    }
}

/// Inspect one model id: the cached entry when it exists, otherwise its Hub `config.json`.
///
/// This is what the browser calls when the operator selects a Hub-only result, so the form can
/// set `model.weights` and `engine.kind` from the checkpoint's real `model_type` instead of a
/// guess from its name.
///
/// # Errors
/// A message when the id is not a model id, is not cached, and its `config.json` cannot be read.
pub fn inspect(cache: Option<&Path>, id: &str) -> Result<RemoteModel, String> {
    if let Some(model) = cache.and_then(|cache| lookup(cache, id)) {
        return Ok(cached_remote(&model));
    }
    if !is_model_id(id) {
        return Err(format!("{id:?} is not a Hugging Face model id"));
    }
    let body = http_get(&format!(
        "https://huggingface.co/{id}/resolve/main/config.json"
    ))?;
    let config: serde_json::Value =
        serde_json::from_str(&body).map_err(|error| format!("{id} config.json: {error}"))?;
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|text| text.get("model_type"))
                .and_then(serde_json::Value::as_str)
        })
        .ok_or_else(|| format!("{id} config.json names no model_type"))?;
    let profile = profile_for_model_type(model_type);
    Ok(RemoteModel {
        id: id.to_owned(),
        downloads: 0,
        likes: 0,
        cached: false,
        complete: false,
        supported: profile.is_some(),
        weights: profile.map(|profile| profile.weights.to_owned()),
        engine: profile.map(|profile| profile.engine.to_owned()),
        context_tokens: profile.map(|profile| profile.context_tokens),
        requires_conversion: profile.map(|profile| profile.requires_conversion),
        family: profile.map(|profile| profile.label.to_owned()),
    })
}

fn cached_remote(model: &CachedModel) -> RemoteModel {
    RemoteModel {
        id: model.id.clone(),
        downloads: 0,
        likes: 0,
        cached: true,
        complete: model.complete,
        supported: model.supported,
        weights: model.weights.clone(),
        engine: model.engine.clone(),
        context_tokens: model.context_tokens,
        requires_conversion: model.requires_conversion,
        family: model
            .weights
            .as_deref()
            .and_then(crate::config::profile_for_weights)
            .map(|profile| profile.label.to_owned()),
    }
}

fn family_matches(id: &str, family: &ModelProfile) -> bool {
    let id = id.to_ascii_lowercase();
    id.contains(&family.org.to_ascii_lowercase())
        || id.contains(&family.search.to_ascii_lowercase())
}

fn sort_models(models: &mut [RemoteModel]) {
    models.sort_by(|left, right| {
        right
            .cached
            .cmp(&left.cached)
            .then_with(|| right.downloads.cmp(&left.downloads))
            .then_with(|| left.id.cmp(&right.id))
    });
}

fn hub_search(term: &str, limit: usize) -> Result<Vec<HubHit>, String> {
    let limit = limit.clamp(1, 50);
    let url = format!(
        "https://huggingface.co/api/models?search={}&limit={limit}&sort=downloads&direction=-1",
        encode_query(term)
    );
    let body = http_get(&url)?;
    serde_json::from_str(&body).map_err(|error| format!("the Hub search response: {error}"))
}

fn http_get(url: &str) -> Result<String, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(15))
        .build();
    let mut request = agent
        .get(url)
        .set("Accept", "application/json")
        .set("User-Agent", "sparkflash-web");
    if let Some(token) = hub_token() {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    match request.call() {
        Ok(response) => response
            .into_string()
            .map_err(|error| format!("reading {url}: {error}")),
        Err(ureq::Error::Status(status, _)) => Err(format!("{url} returned HTTP {status}")),
        Err(ureq::Error::Transport(error)) => Err(format!("{url}: {error}")),
    }
}

fn hub_token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
}

fn encode_query(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
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

    #[test]
    fn the_browser_lists_families_from_the_profiles() {
        let families = families();
        assert_eq!(families.len(), crate::config::MODEL_PROFILES.len());
        assert!(families.iter().any(|family| family.label == "DeepSeek"));
        assert!(families.iter().any(|family| family.label == "Qwen"));
        assert!(families.iter().any(|family| family.label == "GLM"));
        assert!(families.iter().any(|family| family.world_size == 1));
    }

    #[test]
    fn an_empty_query_returns_the_cached_models_without_the_network() {
        let dir = tempfile::tempdir().expect("tempdir");
        cache_model(
            dir.path(),
            crate::config::DEFAULT_MODEL_ID,
            "abc123",
            "deepseek_v41",
        );
        let models = search(Some(dir.path()), "", None, 10).expect("cache search");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, crate::config::DEFAULT_MODEL_ID);
        assert!(models[0].cached);
        assert!(models[0].supported);
        assert_eq!(models[0].family.as_deref(), Some("DeepSeek"));
        assert_eq!(
            models[0].weights.as_deref(),
            Some(crate::config::DEFAULT_WEIGHTS_ID)
        );
    }

    #[test]
    fn inspecting_a_cached_model_does_not_reach_the_network() {
        let dir = tempfile::tempdir().expect("tempdir");
        cache_model(
            dir.path(),
            crate::config::GLM_MODEL_ID,
            "abc123",
            "glm5_next",
        );
        let model = inspect(Some(dir.path()), crate::config::GLM_MODEL_ID).expect("cached inspect");
        assert!(model.cached);
        assert!(model.supported);
        assert_eq!(model.family.as_deref(), Some("GLM"));
        assert_eq!(model.engine.as_deref(), Some(crate::config::GLM_ENGINE_ID));
    }

    #[test]
    fn a_search_term_is_percent_encoded_for_the_hub() {
        assert_eq!(encode_query("Qwen3.8 Flash"), "Qwen3.8%20Flash");
        assert_eq!(encode_query("a/b-c_d.e~f"), "a%2Fb-c_d.e~f");
        assert_eq!(encode_query("GLM-5.3"), "GLM-5.3");
    }
}
