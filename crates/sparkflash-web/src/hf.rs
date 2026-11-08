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

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
pub use sparkflash_weights::hub::{default_cache_dir, is_model_id, repo_folder, resolve};
use sparkflash_weights::hub::{folder_id, snapshot_of};

use sparkflash_model::Architecture;
use sparkflash_ops::parts::{
    host_reference_missing, host_reference_quantization_refusal, host_reference_rope_refusal,
    host_reference_storage_quantization_refusal,
};

use crate::config::{ModelProfile, profile_for_id, profile_for_model_type};
use crate::fit::{self, Cluster, Evidence, Fit, WeightBytes};

/// The `config.json` `model_type` values the released presets load.
///
/// Kept as documentation for the canonical families; it is no longer the gate. A cached model is
/// supported when a device engine serves its own configuration at the width the console launches
/// its family at (see [fit::launch_check]).
pub const SUPPORTED_MODEL_TYPES: &[&str] = &[
    "deepseek_v41",
    "qwen4_exp",
    "qwen4_exp_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "glm5_next",
];

/// What the CLI's CPU host reference cannot execute in a checkpoint; empty when it can run it.
///
/// The host reference is the dynamic engine's correctness path for small test checkpoints. The
/// console never offers it as a way to serve a model; this is only the predicate an explicit
/// `engine.kind = "dynamic"` plan is validated against, so the console refuses what the CLI
/// would refuse. It is the same check the CLI backend runs: parts, rotary scaling, declared
/// quantization, and, when the shards are on disk, storage-only quantization.
fn host_reference_gaps(text: &str, checkpoint: Option<&Path>) -> Vec<String> {
    let architecture = match Architecture::from_json(text) {
        Ok(architecture) => architecture,
        Err(error) => {
            return vec![format!(
                "a config.json the host reference can read ({error})"
            )];
        }
    };
    let mut missing: Vec<String> = host_reference_missing(&architecture)
        .iter()
        .map(ToString::to_string)
        .collect();
    missing.extend(host_reference_rope_refusal(&architecture));
    missing.extend(host_reference_quantization_refusal(&architecture));
    if let Some(directory) = checkpoint
        && let Ok(source) = sparkflash_weights::safetensors::open_checkpoint(directory)
    {
        missing.extend(host_reference_storage_quantization_refusal(
            &architecture,
            &source,
        ));
    }
    missing.sort();
    missing.dedup();
    missing
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
    /// Whether an engine serves the checkpoint at the width the console launches its family at.
    /// A registered host-reference family (`engine = "dynamic"`) counts, because its profile
    /// names the one width that engine runs; it is labelled in the fit summary.
    pub supported: bool,
    /// The control-plane weight selector of the family's device arm.
    pub weights: Option<String>,
    /// The engine selector of the family's device arm.
    pub engine: Option<String>,
    /// The checkpoint's context ceiling.
    pub context_tokens: Option<usize>,
    /// Whether a launch needs per-rank converted weights.
    pub requires_conversion: Option<bool>,
    /// What the device path is missing at the launch width, when the model is unsupported.
    pub missing_parts: Vec<String>,
    /// Whether the snapshot carries `tokenizer.json`.
    pub has_tokenizer: bool,
    /// The bytes downloaded so far.
    pub bytes: u64,
    /// False while a download left `.incomplete` blobs behind.
    pub complete: bool,
    /// The cluster fit, when the caller judged one (see [fit_cached]).
    pub fit: Option<Fit>,
    /// What the CLI's CPU host reference cannot execute, for a family with no device arm:
    /// `Some(empty)` when it can. Only an explicit `engine.kind = "dynamic"` plan reads it.
    #[serde(skip)]
    pub host_reference: Option<Vec<String>>,
}

/// Every model in `cache` that has a snapshot, sorted by id.
///
/// # Errors
///
/// A message when the cache directory cannot be read.
pub fn list(cache: &Path) -> Result<Vec<CachedModel>, String> {
    scan(cache, |_| true)
}

/// The cached models whose id `keep` accepts, sorted by id.
///
/// The id is read from the folder name, so a model the filter rejects is never opened.
fn scan(cache: &Path, keep: impl Fn(&str) -> bool) -> Result<Vec<CachedModel>, String> {
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
        if !keep(&id) {
            continue;
        }
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
    let text = std::fs::read_to_string(snapshot.path.join("config.json")).ok();
    // A download may have written part of config.json; only a whole JSON document is judged.
    let document = text
        .as_deref()
        .filter(|text| serde_json::from_str::<Value>(text).is_ok());
    let model_type = document
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .and_then(|config| config["model_type"].as_str().map(str::to_owned));
    let (bytes, complete) = blob_totals(&repo.join("blobs"));
    // An in-progress download may not have config.json yet. A canonical repository id still
    // identifies the family, so the selected model does not appear unsupported mid-download; a
    // complete document is judged by the device path at its family's launch width.
    let profile = match model_type.as_deref() {
        Some(kind) => profile_for_model_type(kind),
        None => profile_for_id(id),
    };
    let check = document.map(fit::launch_check);
    let supported = match &check {
        Some(check) => check.gaps.is_empty(),
        None => profile.is_some(),
    };
    let context_tokens = check
        .as_ref()
        .and_then(|check| check.window)
        .or_else(|| profile.map(|profile| profile.context_tokens));
    let host_reference = match (profile, document) {
        // A family with no arm, and a registered host-reference family (engine "dynamic"), are
        // both judged by the host predicate so an explicit dynamic plan can be validated.
        (None, Some(text)) => Some(host_reference_gaps(text, Some(&snapshot.path))),
        (Some(profile), Some(text)) if profile.engine == crate::config::DYNAMIC_ENGINE_ID => {
            Some(host_reference_gaps(text, Some(&snapshot.path)))
        }
        _ => None,
    };
    Some(CachedModel {
        supported,
        weights: profile.map(|profile| profile.weights.to_owned()),
        engine: profile.map(|profile| profile.engine.to_owned()),
        context_tokens,
        requires_conversion: profile.map(|profile| profile.requires_conversion),
        missing_parts: check.map(|check| check.gaps).unwrap_or_default(),
        has_tokenizer: snapshot.path.join("tokenizer.json").is_file(),
        snapshot: snapshot.path.display().to_string(),
        revision: snapshot.revision,
        id: id.to_owned(),
        model_type,
        bytes,
        complete,
        fit: None,
        host_reference,
    })
}

/// The cluster fit of a cached checkpoint, from its own `config.json` and the safetensors bytes
/// on disk.
///
/// A download that has not finished has neither a reliable document nor its final size; its
/// canonical family (when the id names one) still says which engine would serve it, and the
/// verdict stays [fit::Verdict::Unknown] until the files land.
#[must_use]
pub fn fit_cached(model: &CachedModel, cluster: &Cluster) -> Fit {
    let snapshot = Path::new(&model.snapshot);
    let text = std::fs::read_to_string(snapshot.join("config.json")).ok();
    let document = text
        .as_deref()
        .filter(|text| serde_json::from_str::<Value>(text).is_ok());
    let declared = profile_for_id(&model.id)
        .filter(|_| document.is_none())
        .map(|profile| serde_json::json!({ "model_type": profile.model_type }));
    let weights = safetensors_bytes(snapshot)
        .filter(|_| model.complete)
        .map(|bytes| WeightBytes {
            bytes,
            measured: true,
        });
    let lacks: Vec<String> = if model.complete && document.is_some() && weights.is_none() {
        vec![NO_SAFETENSORS.to_owned()]
    } else {
        Vec::new()
    };
    fit::judge(
        &Evidence {
            config: document,
            declared: declared.as_ref(),
            weights,
            missing: &lacks,
        },
        cluster,
    )
}

/// Attaches the cluster fit to every model in `models`.
pub fn judge_cached(models: &mut [CachedModel], cluster: &Cluster) {
    for model in models {
        model.fit = Some(fit_cached(model, cluster));
    }
}

/// What a checkpoint with no safetensors shards is missing.
const NO_SAFETENSORS: &str =
    "safetensors weights (this build loads safetensors only; the repository has none)";

/// What a gated repository whose `config.json` could not be read is missing.
const GATED: &str = "access to the gated repository (accept its terms on huggingface.co, then \
                     set HF_TOKEN for the console)";

/// The bytes of the snapshot's top-level `*.safetensors` files, following the hub's symlinks.
///
/// Subfolders (an `original/` export, a drafter) are not what the engine loads, so they are not
/// counted. `None` when there are no safetensors files.
fn safetensors_bytes(snapshot: &Path) -> Option<u64> {
    let entries = std::fs::read_dir(snapshot).ok()?;
    let mut total = None;
    for entry in entries.filter_map(Result::ok) {
        if !entry
            .file_name()
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            continue;
        }
        let size = std::fs::metadata(entry.path()).map_or(0, |meta| meta.len());
        total = Some(total.unwrap_or(0u64).saturating_add(size));
    }
    total
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
    /// Whether a device engine serves it at the width the console launches its family at.
    pub supported: bool,
    /// The control-plane weight selector, when a device arm serves the family.
    pub weights: Option<String>,
    /// The engine selector, when a device arm serves the family.
    pub engine: Option<String>,
    /// The checkpoint's context ceiling, when known.
    pub context_tokens: Option<usize>,
    /// Whether a launch needs per-rank converted weights, when known.
    pub requires_conversion: Option<bool>,
    /// What the device path is missing at the launch width, when the model is unsupported.
    pub missing_parts: Vec<String>,
    /// The family label, when the model type is known.
    pub family: Option<String>,
    /// Whether the Hub gates the repository behind accepted terms.
    pub gated: bool,
    /// Whether, where and how it can be served on this cluster.
    pub fit: Option<Fit>,
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
    /// How many ranks the console launches the family on.
    pub world_size: usize,
    /// Every tensor-parallel width the family's device arm runs.
    pub worlds: Vec<usize>,
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
            worlds: profile.worlds.to_vec(),
        })
        .collect()
}

/// A Hub model, as the search and model endpoints return it with the expanded fields.
#[derive(Debug, Clone, Deserialize)]
struct HubHit {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    /// `false`, or the gating mode (`"auto"`, `"manual"`).
    #[serde(default)]
    gated: Value,
    /// The commit `main` points at.
    #[serde(default)]
    sha: Option<String>,
    /// The `config.json` subset the Hub indexes (`model_type`, `quantization_config`).
    #[serde(default)]
    config: Option<Value>,
    /// Per-dtype parameter counts; absent when the repository has no safetensors.
    #[serde(default)]
    safetensors: Option<Value>,
}

/// The fields every Hub request expands, percent-encoded.
const HUB_EXPAND: &str = "expand%5B%5D=downloads&expand%5B%5D=likes&expand%5B%5D=gated\
                          &expand%5B%5D=sha&expand%5B%5D=config&expand%5B%5D=safetensors";

impl HubHit {
    fn is_gated(&self) -> bool {
        match &self.gated {
            Value::Bool(gated) => *gated,
            Value::String(mode) => !mode.is_empty() && mode != "false",
            _ => false,
        }
    }

    fn declared_model_type(&self) -> Option<&str> {
        let config = self.config.as_ref()?;
        config
            .get("model_type")
            .or_else(|| {
                config
                    .get("text_config")
                    .and_then(|text| text.get("model_type"))
            })
            .and_then(Value::as_str)
    }

    /// Judges the hit. `fetch_any` fetches `config.json` for any family (an inspection); a
    /// search fetches it only for a family a device arm serves, since a family with none is
    /// missing its engine whatever the document says.
    fn judge(&self, cluster: &Cluster, fetch_any: bool) -> RemoteModel {
        let profile = self.declared_model_type().and_then(profile_for_model_type);
        let mut lacks = Vec::new();
        if self.safetensors.is_none() {
            lacks.push(NO_SAFETENSORS.to_owned());
        }
        let text = if profile.is_some() || fetch_any {
            match hub_config(&self.id, self.sha.as_deref()) {
                Ok(text) => Some(text),
                Err(HttpError {
                    status: Some(401 | 403),
                    ..
                }) => {
                    lacks.push(GATED.to_owned());
                    None
                }
                Err(HttpError {
                    status: Some(404), ..
                }) => {
                    lacks.push("a config.json in the repository".to_owned());
                    None
                }
                Err(_) => None,
            }
        } else {
            None
        };
        let weights = self
            .safetensors
            .as_ref()
            .and_then(|safetensors| safetensors.get("parameters"))
            .and_then(|parameters| fit::hub_weight_bytes(parameters, self.config.as_ref()))
            .map(|bytes| WeightBytes {
                bytes,
                measured: false,
            });
        let fit = fit::judge(
            &Evidence {
                config: text.as_deref(),
                declared: self.config.as_ref(),
                weights,
                missing: &lacks,
            },
            cluster,
        );
        let supported = fit
            .launch()
            .is_some_and(|launch| launch.engine_ready == Some(true));
        let missing_parts = if supported {
            Vec::new()
        } else {
            let mut missing = fit.missing.clone();
            if let Some(launch) = fit.launch() {
                missing.extend(launch.reasons.iter().cloned());
            }
            missing
        };
        RemoteModel {
            id: self.id.clone(),
            downloads: self.downloads,
            likes: self.likes,
            cached: false,
            complete: false,
            supported,
            weights: profile.map(|profile| profile.weights.to_owned()),
            engine: profile.map(|profile| profile.engine.to_owned()),
            context_tokens: fit.window,
            requires_conversion: profile.map(|profile| profile.requires_conversion),
            missing_parts,
            family: profile.map(|profile| profile.label.to_owned()),
            gated: self.is_gated(),
            fit: Some(fit),
        }
    }
}

/// Search the Hub and the local cache for models the browser can offer, each judged on
/// `cluster`.
///
/// The local cache is searched first and wins on an id collision, because a cached model has a
/// real size, completion state and `config.json`. A Hub hit that is not cached is judged from
/// the search's expanded fields; for a family a device arm serves, its `config.json` is fetched
/// too (in parallel, and remembered by commit), so the plan, the kernels and the caches are
/// judged on the real document. When `family` is set, an empty query searches for that
/// family's default term; a query overrides it.
///
/// # Errors
/// A message only when the Hub request fails and there is no local result to fall back to.
pub fn search(
    cache: Option<&Path>,
    query: &str,
    family: Option<&ModelProfile>,
    limit: usize,
    cluster: &Cluster,
) -> Result<Vec<RemoteModel>, String> {
    let query = query.trim();
    let text = query.to_ascii_lowercase();
    let local = cache
        .and_then(|cache| {
            scan(cache, |id| {
                text.is_empty() || id.to_ascii_lowercase().contains(&text)
            })
            .ok()
        })
        .unwrap_or_default();
    let in_family = |model: &CachedModel| {
        family.is_none_or(|profile| {
            model.weights.as_deref() == Some(profile.weights)
                || model
                    .id
                    .to_ascii_lowercase()
                    .contains(&profile.search.to_ascii_lowercase())
        })
    };
    let mut results: BTreeMap<String, RemoteModel> = BTreeMap::new();
    for model in local.iter().filter(|model| in_family(model)) {
        results.insert(model.id.clone(), cached_remote(model, cluster));
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
            let fresh: Vec<HubHit> = hits
                .into_iter()
                .filter(|hit| family.is_none_or(|profile| family_matches(&hit.id, profile)))
                .filter(|hit| !results.contains_key(&hit.id))
                .collect();
            let judged: Vec<RemoteModel> = std::thread::scope(|scope| {
                let handles: Vec<_> = fresh
                    .iter()
                    .map(|hit| scope.spawn(move || hit.judge(cluster, false)))
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|handle| handle.join().ok())
                    .collect()
            });
            for model in judged {
                results.entry(model.id.clone()).or_insert(model);
            }
            let mut out: Vec<RemoteModel> = results.into_values().collect();
            sort_models(&mut out);
            Ok(out)
        }
        Err(error) if results.is_empty() => Err(error.to_string()),
        Err(_) => {
            let mut out: Vec<RemoteModel> = results.into_values().collect();
            sort_models(&mut out);
            Ok(out)
        }
    }
}

/// Inspect one model id on `cluster`: the cached entry when it exists, otherwise its Hub record
/// and `config.json`, whatever its family.
///
/// This is what the browser calls when the operator opens a Hub-only result, so a family with
/// no device arm still gets its kernel-level gaps, its caches and its window from the real
/// document.
///
/// # Errors
/// A message when the id is not a model id, is not cached, and the Hub does not know it.
pub fn inspect(cache: Option<&Path>, id: &str, cluster: &Cluster) -> Result<RemoteModel, String> {
    if let Some(model) = cache.and_then(|cache| lookup(cache, id)) {
        return Ok(cached_remote(&model, cluster));
    }
    if !is_model_id(id) {
        return Err(format!("{id:?} is not a Hugging Face model id"));
    }
    let body = http_get(&format!(
        "https://huggingface.co/api/models/{id}?{HUB_EXPAND}"
    ))
    .map_err(|error| error.to_string())?;
    let hit: HubHit =
        serde_json::from_str(&body).map_err(|error| format!("the Hub record for {id}: {error}"))?;
    Ok(hit.judge(cluster, true))
}

fn cached_remote(model: &CachedModel, cluster: &Cluster) -> RemoteModel {
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
        missing_parts: model.missing_parts.clone(),
        family: model
            .weights
            .as_deref()
            .and_then(crate::config::profile_for_weights)
            .map(|profile| profile.label.to_owned()),
        gated: false,
        fit: Some(
            model
                .fit
                .clone()
                .unwrap_or_else(|| fit_cached(model, cluster)),
        ),
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

fn hub_search(term: &str, limit: usize) -> Result<Vec<HubHit>, HttpError> {
    let limit = limit.clamp(1, 50);
    let url = format!(
        "https://huggingface.co/api/models?search={}&limit={limit}&sort=downloads&direction=-1&{HUB_EXPAND}",
        encode_query(term)
    );
    let body = http_get(&url)?;
    serde_json::from_str(&body).map_err(|error| HttpError {
        status: None,
        message: format!("the Hub search response: {error}"),
    })
}

/// Fetched `config.json` documents by `id@commit`, and the HTTP status of the ones that failed.
///
/// A commit's document never changes, so a search does not refetch the documents the last
/// keystroke already read. A transport failure is not remembered.
static HUB_CONFIGS: LazyLock<Mutex<HashMap<String, Result<String, u16>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Documents [HUB_CONFIGS] keeps before it starts over.
const HUB_CONFIG_ENTRIES: usize = 256;

fn hub_config(id: &str, sha: Option<&str>) -> Result<String, HttpError> {
    let revision = sha.unwrap_or("main");
    let key = format!("{id}@{revision}");
    let remembered = HUB_CONFIGS
        .lock()
        .ok()
        .and_then(|memo| memo.get(&key).cloned());
    let fetched = match remembered {
        Some(Ok(text)) => return Ok(text),
        Some(Err(status)) => {
            return Err(HttpError {
                status: Some(status),
                message: format!("{id} config.json returned HTTP {status}"),
            });
        }
        None => http_get(&format!(
            "https://huggingface.co/{id}/resolve/{revision}/config.json"
        )),
    };
    let remember = match &fetched {
        Ok(text) => Some(Ok(text.clone())),
        Err(HttpError {
            status: Some(status),
            ..
        }) => Some(Err(*status)),
        Err(_) => None,
    };
    // `main` moves, so only a pinned commit is remembered.
    if let Some(entry) = remember.filter(|_| sha.is_some())
        && let Ok(mut memo) = HUB_CONFIGS.lock()
    {
        if memo.len() >= HUB_CONFIG_ENTRIES {
            memo.clear();
        }
        memo.insert(key, entry);
    }
    fetched
}

/// A failed Hub request: its HTTP status when the Hub answered.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HttpError {
    status: Option<u16>,
    message: String,
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

fn http_get(url: &str) -> Result<String, HttpError> {
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
        Ok(response) => response.into_string().map_err(|error| HttpError {
            status: None,
            message: format!("reading {url}: {error}"),
        }),
        Err(ureq::Error::Status(status, _)) => Err(HttpError {
            status: Some(status),
            message: format!("{url} returned HTTP {status}"),
        }),
        Err(ureq::Error::Transport(error)) => Err(HttpError {
            status: None,
            message: format!("{url}: {error}"),
        }),
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

    use serde_json::{Value, json};

    use super::*;

    /// The released document for a family a device arm serves, so a cached fixture is judged on
    /// the real geometry; any other `model_type` gets a bare document.
    fn released_document(model_type: &str) -> String {
        match model_type {
            "deepseek_v41" => include_str!(
                "../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
            )
            .to_owned(),
            "glm5_next" => {
                include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json")
                    .to_owned()
            }
            "qwen4_exp" => include_str!(
                "../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json"
            )
            .to_owned(),
            other => format!("{{\"model_type\": \"{other}\"}}"),
        }
    }

    fn nominal() -> Cluster {
        Cluster::nominal(4)
    }

    /// Writes a cached model the way `huggingface_hub` does, and returns its snapshot path.
    pub(crate) fn cache_model(cache: &Path, id: &str, revision: &str, model_type: &str) -> PathBuf {
        let repo = cache.join(repo_folder(id));
        let snapshot = repo.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        std::fs::create_dir_all(repo.join("refs")).expect("refs dir");
        std::fs::create_dir_all(repo.join("blobs")).expect("blobs dir");
        std::fs::write(repo.join("refs").join("main"), format!("{revision}\n")).expect("ref");
        let config = released_document(model_type);
        std::fs::write(repo.join("blobs").join("config-blob"), &config).expect("blob");
        std::fs::write(snapshot.join("config.json"), &config).expect("config");
        std::fs::write(snapshot.join("tokenizer.json"), "{}").expect("tokenizer");
        std::fs::write(snapshot.join("model.safetensors"), [0u8; 8]).expect("shard");
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
        let models = search(Some(dir.path()), "", None, 10, &nominal()).expect("cache search");
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
        let model = inspect(Some(dir.path()), crate::config::GLM_MODEL_ID, &nominal())
            .expect("cached inspect");
        assert!(model.cached);
        assert!(model.supported);
        assert_eq!(model.family.as_deref(), Some("GLM"));
        assert_eq!(model.engine.as_deref(), Some(crate::config::GLM_ENGINE_ID));
        let fit = model.fit.expect("an inspection carries its fit");
        assert_eq!(fit.verdict, fit::Verdict::Runs, "{fit:#?}");
        assert!(fit.launch().is_some_and(|launch| launch.world == 4));
    }

    pub(crate) fn custom_config() -> Value {
        serde_json::json!({
            "model_type": "acme_custom_flash",
            "hidden_size": 2560,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "eos_token_id": [0],
            "layer_types": ["full_attention"]
        })
    }

    pub(crate) fn cache_config(cache: &Path, id: &str, revision: &str, config: &Value) -> PathBuf {
        let repo = cache.join(repo_folder(id));
        let snapshot = repo.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        std::fs::create_dir_all(repo.join("refs")).expect("refs dir");
        std::fs::write(repo.join("refs").join("main"), format!("{revision}\n")).expect("ref");
        std::fs::write(
            snapshot.join("config.json"),
            serde_json::to_vec(config).expect("config JSON"),
        )
        .expect("config");
        snapshot
    }

    #[test]
    fn the_released_configs_resolve_through_the_shared_plan() {
        for (name, text) in [
            (
                "qwen",
                include_str!(
                    "../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json"
                ),
            ),
            (
                "glm",
                include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json"),
            ),
            (
                "qwen fp8",
                include_str!(
                    "../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_fp8.hf.json"
                ),
            ),
            (
                "deepseek",
                include_str!(
                    "../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
                ),
            ),
        ] {
            let check = fit::launch_check(text);
            assert!(check.gaps.is_empty(), "{name}: {:?}", check.gaps);
            assert!(check.window.is_some(), "{name} declares no window");
        }
    }

    #[test]
    fn a_custom_cached_model_is_not_served_by_the_cpu_reference() {
        let dir = tempfile::tempdir().expect("tempdir");
        cache_config(
            dir.path(),
            "acme/Custom-Flash-Test",
            "abc123",
            &custom_config(),
        );
        let model = lookup(dir.path(), "acme/Custom-Flash-Test").expect("cache entry");
        // The host reference can execute it, but that is a test oracle: no device engine serves
        // the family, so the browser must not offer it.
        assert!(!model.supported);
        assert_eq!(model.host_reference.as_deref(), Some(&[][..]));
        assert_eq!(model.model_type.as_deref(), Some("acme_custom_flash"));
        assert_eq!(model.context_tokens, Some(128));
        assert!(
            model
                .missing_parts
                .iter()
                .any(|part| part.contains("device engine")),
            "{:?}",
            model.missing_parts
        );
        assert_eq!(model.engine, None);
        assert_eq!(model.weights, None);
        let fit = fit_cached(&model, &nominal());
        assert_eq!(fit.verdict, fit::Verdict::Missing);
        assert!(fit.summary.contains("device engine"), "{}", fit.summary);
        let models = list(dir.path()).expect("list");
        assert_eq!(models.len(), 1);
        assert!(!models[0].supported);
    }

    #[test]
    fn a_declared_quantized_custom_checkpoint_is_not_supported() {
        // The CLI host backend refuses a declared NVFP4/FP8 document at boot because its
        // BF16/F32 decode would not apply the block scales. The console must judge the same
        // document by the same predicate, or it offers a model the serve command refuses.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = custom_config();
        config["quantization_config"] = json!({
            "quant_method": "modelopt",
            "quant_algo": "mixed_precision",
            "expert_weight": {"quant_algo": "nvfp4", "group_size": 16}
        });
        cache_config(dir.path(), "acme/Quantized-Flash", "abc123", &config);
        let model = lookup(dir.path(), "acme/Quantized-Flash").expect("cache entry");
        assert!(!model.supported);
        let host = model
            .host_reference
            .expect("a novel family is host-checked");
        assert!(
            host.iter().any(|part| part.contains("quantized")),
            "the host refusal names quantization: {host:?}"
        );
    }

    #[test]
    fn a_storage_only_quantized_custom_checkpoint_is_not_supported() {
        // A document can omit quantization_config while its shards store U8 codes. The CLI
        // backend names the tensor from its header; the console must refuse the same checkpoint
        // instead of offering a model the launch would reject.
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = cache_config(
            dir.path(),
            "acme/Storage-Quantized-Flash",
            "abc123",
            &custom_config(),
        );
        sparkflash_weights::safetensors::write_checkpoint(
            &snapshot,
            &[(
                sparkflash_weights::TensorSpec {
                    name: "model.embed_tokens.weight".to_owned(),
                    dtype: sparkflash_weights::DType::U8,
                    shape: vec![16],
                },
                vec![0x21; 16],
            )],
            1,
        )
        .expect("the storage-only quantized shard writes");
        let model = lookup(dir.path(), "acme/Storage-Quantized-Flash").expect("cache entry");
        assert!(!model.supported);
        let host = model
            .host_reference
            .expect("a novel family is host-checked");
        assert!(
            host.iter()
                .any(|part| part.contains("U8") && part.contains("quantized")),
            "the host refusal names the stored dtype: {host:?}"
        );
    }

    #[test]
    fn a_yarn_rope_custom_checkpoint_is_not_supported() {
        // The host forward applies plain and positive-linear RoPE only. A yarn rule is accepted
        // by the shared architecture and the plan, so the web gate must apply the same boot
        // predicate the CLI backend uses rather than accepting it and failing per request.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = custom_config();
        config["rope_scaling"] = json!({
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128
        });
        cache_config(dir.path(), "acme/Yarn-Flash", "abc123", &config);
        let model = lookup(dir.path(), "acme/Yarn-Flash").expect("cache entry");
        assert!(!model.supported);
        let host = model
            .host_reference
            .expect("a novel family is host-checked");
        assert!(
            host.iter().any(|part| part.contains("rope.scaling")),
            "the host refusal names the rotary rule: {host:?}"
        );
    }

    #[test]
    fn an_unresolved_custom_kernel_is_named_in_the_fit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = custom_config();
        // MLA at a 2560-wide hidden state: the catalogue's MLA bodies are compiled for 4096 or
        // 5120, so the plan resolves but no kernel does.
        config["kv_lora_rank"] = serde_json::json!(16);
        cache_config(dir.path(), "acme/Custom-Mla", "abc123", &config);
        let model = lookup(dir.path(), "acme/Custom-Mla").expect("cache entry");
        assert!(!model.supported);
        let fit = fit_cached(&model, &nominal());
        let tp1 = &fit.configurations[0];
        assert_eq!(tp1.world, 1);
        assert!(
            tp1.reasons
                .iter()
                .any(|reason| reason.contains("mla_attention")),
            "TP=1 reasons: {:?}",
            tp1.reasons
        );
    }

    #[test]
    fn a_catalogue_shaped_mla_is_host_unsupported() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = custom_config();
        // A 5120 hidden state is DeepSeek's released MLA width: the CUDA catalogue has an entry
        // for that geometry, but the host reference still cannot execute an MLA layer.
        config["hidden_size"] = serde_json::json!(5120);
        config["kv_lora_rank"] = serde_json::json!(16);
        cache_config(dir.path(), "acme/Custom-Mla", "abc123", &config);
        let model = lookup(dir.path(), "acme/Custom-Mla").expect("cache entry");
        assert!(!model.supported);
        let host = model
            .host_reference
            .expect("a novel family is host-checked");
        assert!(
            host.iter().any(|part| part == "mla_attention"),
            "host gaps: {host:?}"
        );
    }

    #[test]
    fn a_host_runnable_gqa_is_still_not_served() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = custom_config();
        config["hidden_size"] = serde_json::json!(1024);
        cache_config(dir.path(), "acme/Custom-Small", "abc123", &config);
        let model = lookup(dir.path(), "acme/Custom-Small").expect("cache entry");
        assert!(!model.supported);
        assert_eq!(model.host_reference.as_deref(), Some(&[][..]));
        assert_eq!(model.engine, None);
        assert_eq!(model.weights, None);
        assert_eq!(model.requires_conversion, None);
    }

    #[test]
    fn a_cached_fit_measures_the_shards_and_waits_for_a_download() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = cache_model(
            dir.path(),
            "zai-org/GLM-5.3-Flash-Tune",
            "abc123",
            "glm5_next",
        );
        std::fs::remove_file(snapshot.join("model.safetensors")).expect("fixture shard");
        std::fs::write(
            snapshot.join("model-00001-of-00002.safetensors"),
            [0u8; 300],
        )
        .expect("shard");
        std::fs::write(
            snapshot.join("model-00002-of-00002.safetensors"),
            [0u8; 200],
        )
        .expect("shard");
        std::fs::create_dir_all(snapshot.join("original")).expect("subfolder");
        std::fs::write(snapshot.join("original/model.safetensors"), [0u8; 999]).expect("shard");
        let model = lookup(dir.path(), "zai-org/GLM-5.3-Flash-Tune").expect("cache entry");
        assert!(model.supported, "{:?}", model.missing_parts);
        let fit = fit_cached(&model, &nominal());
        assert_eq!(fit.weights_bytes, Some(500));
        assert!(fit.weights_measured);
        assert_eq!(fit.verdict, fit::Verdict::Runs, "{fit:#?}");

        // Mid-download the canonical family still names its engine, but nothing is claimed.
        let canonical = cache_model(
            dir.path(),
            crate::config::GLM_MODEL_ID,
            "def456",
            "glm5_next",
        );
        std::fs::remove_file(canonical.join("config.json")).expect("drop config");
        std::fs::write(
            dir.path()
                .join(repo_folder(crate::config::GLM_MODEL_ID))
                .join("blobs/shard.incomplete"),
            "xx",
        )
        .expect("partial blob");
        let downloading = lookup(dir.path(), crate::config::GLM_MODEL_ID).expect("cache entry");
        assert!(downloading.supported);
        let fit = fit_cached(&downloading, &nominal());
        assert_eq!(fit.verdict, fit::Verdict::Unknown, "{fit:#?}");
        assert_eq!(fit.engine.as_deref(), Some(crate::config::GLM_ENGINE_ID));
        assert_eq!(fit.weights_bytes, None);
    }

    #[test]
    fn a_hub_hit_is_judged_from_its_expanded_fields() {
        // A family with no device arm needs no document fetch to be judged missing its engine,
        // and the expanded safetensors counts still size it.
        let hit: HubHit = serde_json::from_value(json!({
            "id": "meta-llama/Llama-3.1-8B",
            "downloads": 487_572,
            "likes": 2764,
            "gated": "manual",
            "sha": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
            "config": {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            "safetensors": {"parameters": {"BF16": 8_030_261_248u64}, "total": 8_030_261_248u64}
        }))
        .expect("the Hub record parses");
        let model = hit.judge(&nominal(), false);
        assert!(model.gated);
        assert!(!model.supported);
        assert!(!model.cached);
        let fit = model.fit.expect("judged");
        assert_eq!(fit.verdict, fit::Verdict::Missing);
        assert_eq!(fit.weights_bytes, Some(16_060_522_496));
        assert!(!fit.weights_measured);
        assert!(fit.summary.contains("llama"), "{}", fit.summary);

        // A GGUF-only repository has no safetensors, which is named as missing.
        let gguf: HubHit = serde_json::from_value(json!({
            "id": "TheBloke/Llama-2-7B-GGUF",
            "config": {"model_type": "llama"}
        }))
        .expect("the Hub record parses");
        let fit = gguf.judge(&nominal(), false).fit.expect("judged");
        assert!(
            fit.missing
                .iter()
                .any(|reason| reason.contains("safetensors")),
            "{:?}",
            fit.missing
        );
    }

    #[test]
    fn a_search_term_is_percent_encoded_for_the_hub() {
        assert_eq!(encode_query("Qwen3.8 Flash"), "Qwen3.8%20Flash");
        assert_eq!(encode_query("a/b-c_d.e~f"), "a%2Fb-c_d.e~f");
        assert_eq!(encode_query("GLM-5.3"), "GLM-5.3");
    }
}
