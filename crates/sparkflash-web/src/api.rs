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

//! HTTP routes and embedded assets for the control plane.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use axum::extract::{Path as AxumPath, Query, Request, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::commands::{ServeAction, convert_command, local_hostname, serve_action_command};
use crate::config::{
    MAX_CONCURRENCY, MODEL_CONTEXT_TOKENS, SUPPORTED_ENGINES, SUPPORTED_WEIGHTS, SparkConfig,
    WORLD_SIZE,
};
use crate::discovery::{DiscoveryReport, NodeDiscovery, SshProbe, report_nodes};
use crate::hf::{self, CachedModel};
use crate::jobs::{JobManager, JobSummary};
use crate::stats::{MAX_SERIES_POINTS, StatsStore, TelemetrySource, now_ms};

/// The header every state-changing request must carry (the UI sends it). A browser adds a custom
/// header to a cross-origin request only after a CORS preflight, which this server never grants,
/// so another site open in the operator's browser cannot launch, stop or reconfigure the cluster.
pub const ACTION_HEADER: &str = "x-sparkflash-web";

/// The cluster actions that change whether the group runs; one at a time.
const CLUSTER_STATE_KINDS: &[&str] = &["launch", "stop"];

const INDEX_HTML: &str = include_str!("../assets/index.html");
const APP_CSS: &str = include_str!("../assets/app.css");
const APP_JS: &str = include_str!("../assets/app.js");

/// The dependencies and startup facts used to build a [`WebState`].
pub struct WebStateInit {
    /// The validated configuration.
    pub config: SparkConfig,
    /// The file the config was loaded from and is saved back to.
    pub config_path: PathBuf,
    /// A load error to surface until the operator saves a fix.
    pub load_error: Option<String>,
    /// Whether the config file existed when the process started.
    pub exists: bool,
    /// The mDNS browser/advertiser.
    pub discovery: Arc<dyn NodeDiscovery>,
    /// The passwordless-SSH probe.
    pub ssh: Arc<dyn SshProbe>,
    /// The process job manager.
    pub jobs: Arc<JobManager>,
    /// The telemetry store.
    pub stats: Arc<StatsStore>,
    /// The rank-0 scrape source.
    pub telemetry: Arc<TelemetrySource>,
    /// The full mDNS name this process advertised, when registration succeeded.
    pub advertised: Option<String>,
}

/// Everything the routes share.
#[derive(Clone)]
pub struct WebState {
    config: Arc<RwLock<ConfigState>>,
    config_path: PathBuf,
    discovery: Arc<dyn NodeDiscovery>,
    ssh: Arc<dyn SshProbe>,
    jobs: Arc<JobManager>,
    stats: Arc<StatsStore>,
    telemetry: Arc<RwLock<Arc<TelemetrySource>>>,
    last_discovery: Arc<RwLock<Option<DiscoveryReport>>>,
    discovery_lock: Arc<tokio::sync::Mutex<()>>,
    advertised: Option<String>,
    hostname: String,
}

struct ConfigState {
    config: SparkConfig,
    load_error: Option<String>,
    exists: bool,
}

impl WebState {
    /// Builds a state value from the already-loaded configuration.
    #[must_use]
    pub fn new(init: WebStateInit) -> Self {
        Self {
            config: Arc::new(RwLock::new(ConfigState {
                config: init.config,
                load_error: init.load_error,
                exists: init.exists,
            })),
            config_path: init.config_path,
            discovery: init.discovery,
            ssh: init.ssh,
            jobs: init.jobs,
            stats: init.stats,
            telemetry: Arc::new(RwLock::new(init.telemetry)),
            last_discovery: Arc::new(RwLock::new(None)),
            discovery_lock: Arc::new(tokio::sync::Mutex::new(())),
            advertised: init.advertised,
            hostname: local_hostname(),
        }
    }

    /// The current configuration.
    #[must_use]
    pub fn config(&self) -> SparkConfig {
        self.config.read().expect("config lock").config.clone()
    }

    /// The configuration an action may run with.
    ///
    /// A missing file is fine: the built-in defaults are a valid starting point and the operator
    /// can save them. An *invalid file that exists* is not: silently launching it with defaults
    /// would be the opposite of what the file says.
    ///
    /// # Errors
    /// Returns the load error when the file exists but did not validate.
    pub fn usable_config(&self) -> Result<SparkConfig, String> {
        let state = self.config.read().expect("config lock");
        if state.exists && state.load_error.is_some() {
            return Err(format!(
                "the configuration file did not validate: {}",
                state.load_error.as_deref().unwrap_or("unknown error")
            ));
        }
        Ok(state.config.clone())
    }

    /// The configuration a launch or conversion runs with: [`Self::usable_config`] with the
    /// checkpoint resolved, so a model missing from the Hugging Face cache is refused up front
    /// instead of failing on every rank.
    ///
    /// # Errors
    /// Returns the load error, or why the checkpoint does not resolve.
    pub fn runnable_config(&self) -> Result<SparkConfig, String> {
        self.usable_config()?
            .resolved()
            .map_err(|error| error.to_string())
    }

    /// The configuration file path.
    #[must_use]
    pub fn config_path(&self) -> &std::path::Path {
        &self.config_path
    }

    /// The stats store, for the sampler.
    #[must_use]
    pub fn stats(&self) -> Arc<StatsStore> {
        Arc::clone(&self.stats)
    }

    /// The telemetry source, for the sampler.
    ///
    /// The source is replaced when a saved config changes `web.serve_url`, so the next scrape
    /// follows the new rank-0 address without restarting the control plane.
    #[must_use]
    pub fn telemetry(&self) -> Arc<TelemetrySource> {
        Arc::clone(&self.telemetry.read().expect("telemetry lock"))
    }
}

/// Builds the web router.
pub fn router(state: WebState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/assets/app.css", get(css))
        .route("/assets/app.js", get(js))
        .route("/api/health", get(health))
        .route("/api/state", get(state_snapshot))
        .route("/api/config", get(get_config).put(put_config))
        .route("/api/config/reload", post(reload_config))
        .route("/api/discover", post(discover))
        .route("/api/models", get(models))
        .route("/api/series", get(series))
        .route("/api/jobs", get(list_jobs))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/actions/convert", post(convert))
        .route("/api/actions/launch", post(launch))
        .route("/api/actions/stop", post(stop))
        .route("/api/actions/status", post(status))
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

/// Refuses a request whose `Host` this listener does not answer to (a DNS-rebound page), and a
/// state-changing request without [`ACTION_HEADER`] (a cross-site form or `fetch`).
async fn guard(State(web): State<WebState>, request: Request, next: Next) -> Response {
    let allowed = web.config().web.allowed_hosts;
    if let Some(host) = request_host(request.headers())
        && !host_allowed(host, &web.hostname, &allowed)
    {
        return api_error(
            StatusCode::MISDIRECTED_REQUEST,
            format!(
                "this control plane does not answer to host {host:?}; add it to web.allowed_hosts"
            ),
        );
    }
    if !matches!(*request.method(), Method::GET | Method::HEAD)
        && !request.headers().contains_key(ACTION_HEADER)
    {
        return api_error(
            StatusCode::FORBIDDEN,
            format!("state-changing requests must carry the {ACTION_HEADER} header"),
        );
    }
    next.run(request).await
}

/// The request's host name without its port; `None` when it sent no usable `Host`.
fn request_host(headers: &HeaderMap) -> Option<&str> {
    let host = headers.get(header::HOST)?.to_str().ok()?;
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next();
    }
    Some(match host.rsplit_once(':') {
        Some((name, port)) if port.bytes().all(|byte| byte.is_ascii_digit()) => name,
        _ => host,
    })
}

/// An IP literal, `localhost`, this machine's own name, or a configured name.
fn host_allowed(host: &str, hostname: &str, allowed: &[String]) -> bool {
    let host = host.trim_end_matches('.');
    host.parse::<std::net::IpAddr>().is_ok()
        || host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
        || host.eq_ignore_ascii_case(hostname)
        || host
            .strip_suffix(".local")
            .is_some_and(|name| name.eq_ignore_ascii_case(hostname))
        || allowed.iter().any(|name| name.eq_ignore_ascii_case(host))
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn css() -> Response {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS).into_response()
}

async fn js() -> Response {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        APP_JS,
    )
        .into_response()
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"ok": true, "now_ms": now_ms()}))
}

#[derive(Serialize)]
struct ConfigView {
    path: String,
    exists: bool,
    load_error: Option<String>,
    config: SparkConfig,
}

#[derive(Serialize)]
struct StateView {
    now_ms: u64,
    config_path: String,
    config: SparkConfig,
    config_error: Option<String>,
    config_exists: bool,
    latest: Option<crate::stats::Sample>,
    totals: crate::stats::Totals,
    cluster: Option<crate::stats::ClusterTelemetry>,
    last_error: Option<String>,
    jobs: Vec<JobSummary>,
    discovery: Option<DiscoveryReport>,
    advertised: Option<String>,
    checkpoint: CheckpointView,
    limits: Limits,
    supported: Supported,
}

/// Where the next launch or conversion reads the weights from.
#[derive(Serialize)]
struct CheckpointView {
    /// The checkpoint directory, when it resolves.
    path: Option<String>,
    /// Whether it is looked up in the Hugging Face cache rather than set in `model.checkpoint`.
    from_cache: bool,
    /// The hub cache `model.id` is looked up in.
    cache: Option<String>,
    /// Why it does not resolve.
    error: Option<String>,
}

impl CheckpointView {
    fn of(config: &SparkConfig) -> Self {
        let resolved = config.resolve_checkpoint();
        Self {
            path: resolved
                .as_ref()
                .ok()
                .map(|path| path.display().to_string()),
            from_cache: config.model.checkpoint.trim().is_empty(),
            cache: config.hub_cache_dir().map(|dir| dir.display().to_string()),
            error: resolved.err().map(|error| error.to_string()),
        }
    }
}

#[derive(Serialize)]
struct Limits {
    max_context_tokens: usize,
    max_concurrency: usize,
    world_size: usize,
}

#[derive(Serialize)]
struct Supported {
    weights: &'static [&'static str],
    engines: &'static [&'static str],
}

async fn state_snapshot(State(web): State<WebState>) -> Json<StateView> {
    let (config, config_error, config_exists) = {
        let state = web.config.read().expect("config lock");
        (state.config.clone(), state.load_error.clone(), state.exists)
    };
    let discovery = web.last_discovery.read().expect("discovery lock").clone();
    let checkpoint = CheckpointView::of(&config);
    Json(StateView {
        now_ms: now_ms(),
        config_path: web.config_path.display().to_string(),
        config,
        config_error,
        config_exists,
        latest: web.stats.latest(),
        totals: web.stats.totals(),
        cluster: web.stats.cluster(),
        last_error: web.stats.last_error(),
        jobs: web.jobs.list(),
        discovery,
        advertised: web.advertised.clone(),
        checkpoint,
        limits: Limits {
            max_context_tokens: MODEL_CONTEXT_TOKENS,
            max_concurrency: MAX_CONCURRENCY,
            world_size: WORLD_SIZE,
        },
        supported: Supported {
            weights: SUPPORTED_WEIGHTS,
            engines: SUPPORTED_ENGINES,
        },
    })
}

async fn get_config(State(web): State<WebState>) -> Json<ConfigView> {
    let state = web.config.read().expect("config lock");
    Json(ConfigView {
        path: web.config_path.display().to_string(),
        exists: state.exists,
        load_error: state.load_error.clone(),
        config: state.config.clone(),
    })
}

async fn put_config(State(web): State<WebState>, Json(body): Json<serde_json::Value>) -> Response {
    let config_value = body.get("config").cloned().unwrap_or(body);
    let config: SparkConfig = match serde_json::from_value(config_value) {
        Ok(config) => config,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("configuration is not the expected shape: {error}"),
            );
        }
    };
    if let Err(error) = config.validate() {
        return api_error(StatusCode::BAD_REQUEST, error.message);
    }
    if let Err(error) = config.save(&web.config_path) {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, error.message);
    }
    {
        let mut state = web.config.write().expect("config lock");
        state.config = config.clone();
        state.load_error = None;
        state.exists = true;
    }
    *web.telemetry.write().expect("telemetry lock") =
        Arc::new(TelemetrySource::new(&config.web.serve_url));
    Json(ConfigView {
        path: web.config_path.display().to_string(),
        exists: true,
        load_error: None,
        config,
    })
    .into_response()
}

async fn reload_config(State(web): State<WebState>) -> Response {
    match SparkConfig::load(&web.config_path) {
        Ok(config) => {
            {
                let mut state = web.config.write().expect("config lock");
                state.config = config.clone();
                state.load_error = None;
                state.exists = true;
            }
            *web.telemetry.write().expect("telemetry lock") =
                Arc::new(TelemetrySource::new(&config.web.serve_url));
            Json(ConfigView {
                path: web.config_path.display().to_string(),
                exists: true,
                load_error: None,
                config,
            })
            .into_response()
        }
        Err(error) => {
            let mut state = web.config.write().expect("config lock");
            state.load_error = Some(error.message.clone());
            api_error(StatusCode::BAD_REQUEST, error.message)
        }
    }
}

async fn discover(State(web): State<WebState>) -> Response {
    let _guard = web.discovery_lock.lock().await;
    let config = web.config();
    let discovery = Arc::clone(&web.discovery);
    let probe = Arc::clone(&web.ssh);
    let timeout = std::time::Duration::from_secs(config.web.discovery_timeout_secs);
    let result = tokio::task::spawn_blocking(move || {
        let found = discovery.discover(timeout);
        let (answers, error) = match found {
            Ok(answers) => (answers, None),
            Err(error) => (Vec::new(), Some(error)),
        };
        let mut report = report_nodes(&config, &answers, probe.as_ref());
        report.mdns_error = error;
        report
    })
    .await;
    match result {
        Ok(report) => {
            *web.last_discovery.write().expect("discovery lock") = Some(report.clone());
            Json(report).into_response()
        }
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("discovery task failed: {error}"),
        ),
    }
}

/// The models in the Hugging Face cache, for the model selector.
#[derive(Serialize)]
struct ModelsView {
    cache: Option<String>,
    models: Vec<CachedModel>,
    error: Option<String>,
}

async fn models(State(web): State<WebState>) -> Json<ModelsView> {
    let cache = web.config().hub_cache_dir();
    let listed = match cache.clone() {
        Some(dir) => tokio::task::spawn_blocking(move || hf::list(&dir))
            .await
            .unwrap_or_else(|error| Err(format!("the cache scan failed: {error}"))),
        None => {
            Err("no Hugging Face cache directory: set HF_HUB_CACHE or model.hub_cache".to_owned())
        }
    };
    let (models, error) = match listed {
        Ok(models) => (models, None),
        Err(error) => (Vec::new(), Some(error)),
    };
    Json(ModelsView {
        cache: cache.map(|dir| dir.display().to_string()),
        models,
        error,
    })
}

#[derive(Deserialize)]
struct SeriesQuery {
    #[serde(default = "default_window")]
    window_secs: u64,
}

const fn default_window() -> u64 {
    3_600
}

async fn series(
    State(web): State<WebState>,
    Query(query): Query<SeriesQuery>,
) -> Json<crate::stats::SeriesResponse> {
    let config = web.config();
    let window = query.window_secs.clamp(60, config.stats.retention_secs);
    Json(web.stats.series(window, now_ms(), MAX_SERIES_POINTS))
}

async fn list_jobs(State(web): State<WebState>) -> Json<Vec<JobSummary>> {
    Json(web.jobs.list())
}

async fn get_job(State(web): State<WebState>, AxumPath(id): AxumPath<u64>) -> Response {
    match web.jobs.get(id) {
        Some(job) => Json(job).into_response(),
        None => api_error(StatusCode::NOT_FOUND, format!("job {id} does not exist")),
    }
}

#[derive(Deserialize, Default)]
struct ConvertRequest {
    ranks: Option<Vec<usize>>,
}

async fn convert(State(web): State<WebState>, Json(request): Json<ConvertRequest>) -> Response {
    let config = match web.runnable_config() {
        Ok(config) => config,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    if web.stats.latest().is_some_and(|sample| sample.online) {
        return api_error(
            StatusCode::CONFLICT,
            "rank 0 is online; stop the cluster before converting its weights",
        );
    }
    let ranks = request.ranks.unwrap_or_else(|| (0..WORLD_SIZE).collect());
    if ranks.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "ranks must not be empty");
    }
    for rank in &ranks {
        if *rank >= WORLD_SIZE {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("rank {rank} is outside 0..{WORLD_SIZE}"),
            );
        }
        if web.jobs.convert_running(*rank) {
            return api_error(
                StatusCode::CONFLICT,
                format!("a conversion for rank {rank} is already running"),
            );
        }
    }
    let mut started = Vec::new();
    for rank in ranks {
        let command = convert_command(&config, rank);
        // Refused while this rank converts or the group is being launched or stopped.
        match web.jobs.start_exclusive(command, CLUSTER_STATE_KINDS).await {
            Ok(id) => started.push(id),
            Err(error) => {
                return api_error(StatusCode::CONFLICT, error.message);
            }
        }
    }
    accepted_jobs(&web, &started)
}

async fn launch(State(web): State<WebState>) -> Response {
    if web.stats.latest().is_some_and(|sample| sample.online) {
        return api_error(
            StatusCode::CONFLICT,
            "rank 0 is already reachable; stop the cluster before launching it",
        );
    }
    start_action(&web, ServeAction::Launch).await
}

async fn stop(State(web): State<WebState>) -> Response {
    start_action(&web, ServeAction::Stop).await
}

async fn status(State(web): State<WebState>) -> Response {
    start_action(&web, ServeAction::Status).await
}

async fn start_action(web: &WebState, action: ServeAction) -> Response {
    let config = match action {
        ServeAction::Launch => web.runnable_config(),
        // Stop and status never read the weights, so a model missing from the cache must not
        // block them.
        ServeAction::Stop | ServeAction::Status => web
            .usable_config()
            .map(|config| config.resolved().unwrap_or(config)),
    };
    let config = match config {
        Ok(config) => config,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    let command = serve_action_command(&config, action);
    // Launch and stop exclude each other (and a second of themselves); status runs alongside.
    let exclusive: &[&str] = match action {
        ServeAction::Launch | ServeAction::Stop => CLUSTER_STATE_KINDS,
        ServeAction::Status => &[],
    };
    match web.jobs.start_exclusive(command, exclusive).await {
        Ok(id) => accepted_jobs(web, &[id]),
        Err(error) => api_error(StatusCode::CONFLICT, error.message),
    }
}

fn accepted_jobs(web: &WebState, ids: &[u64]) -> Response {
    let jobs: Vec<JobSummary> = ids
        .iter()
        .filter_map(|id| web.jobs.get(*id).map(|detail| detail.summary))
        .collect();
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({"jobs": jobs})),
    )
        .into_response()
}

fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({"error": message.into()}))).into_response()
}

/// A test-only convenience: build routes over explicit fakes.
#[cfg(test)]
pub(crate) fn test_router(state: WebState) -> Router {
    router(state)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;
    use crate::commands::CommandSpec;
    use crate::config::{SparkConfig, StatsConfig};
    use crate::discovery::{DiscoveredNode, NodeDiscovery, ProbeOutcome, SshProbe};
    use crate::jobs::JobManager;
    use crate::stats::StatsStore;

    struct FakeDiscovery;
    impl NodeDiscovery for FakeDiscovery {
        fn discover(&self, _timeout: std::time::Duration) -> Result<Vec<DiscoveredNode>, String> {
            Ok(vec![DiscoveredNode {
                hostname: "spark7".to_owned(),
                addresses: vec!["10.0.0.7".to_owned()],
                port: Some(22),
                services: vec!["_ssh._tcp.local.".to_owned()],
                properties: BTreeMap::new(),
            }])
        }
    }

    struct FakeSsh;
    impl SshProbe for FakeSsh {
        fn probe(&self, _command: &CommandSpec) -> ProbeOutcome {
            ProbeOutcome {
                ok: true,
                message: String::new(),
            }
        }
    }

    fn test_state(dir: &std::path::Path) -> WebState {
        test_state_with(dir, false, None)
    }

    fn test_state_with(
        dir: &std::path::Path,
        exists: bool,
        load_error: Option<String>,
    ) -> WebState {
        let mut config = SparkConfig::default();
        // Never the real ~/.cache: each test's hub cache starts empty.
        config.model.hub_cache = dir.join("hub").display().to_string();
        let stats_config = StatsConfig {
            persist_path: dir.join("stats.json").display().to_string(),
            ..StatsConfig::default()
        };
        let stats = Arc::new(StatsStore::new(&stats_config, dir.join("stats.json")));
        WebState::new(WebStateInit {
            config,
            config_path: dir.join("sparkflash.toml"),
            load_error,
            exists,
            discovery: Arc::new(FakeDiscovery),
            ssh: Arc::new(FakeSsh),
            jobs: Arc::new(JobManager::new()),
            stats,
            telemetry: Arc::new(TelemetrySource::new("http://127.0.0.1:9")),
            advertised: Some("sparkflash-test._sparkflash._tcp.local.".to_owned()),
        })
    }

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("JSON body")
    }

    #[tokio::test]
    async fn serves_the_dark_ui_and_its_assets() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        let index = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(index.status(), StatusCode::OK);
        let html = String::from_utf8(
            index
                .into_body()
                .collect()
                .await
                .expect("body")
                .to_bytes()
                .to_vec(),
        )
        .expect("utf8");
        assert!(html.contains("SparkFlash Control"));
        assert!(html.contains("prefers-color-scheme") || html.contains("dark"));
        let css = app
            .oneshot(
                Request::builder()
                    .uri("/assets/app.css")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(css.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn state_reports_limits_and_configuration() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/state")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["limits"]["world_size"], 4);
        assert_eq!(body["limits"]["max_concurrency"], 32);
        assert_eq!(body["config"]["runtime"]["max_context_tokens"], 1_048_576);
        assert_eq!(body["supported"]["engines"][0], "deepseek-v4.1-flash");
        assert_eq!(body["config_exists"], false);
    }

    #[tokio::test]
    async fn saving_config_validates_and_writes_the_toml() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state(dir.path());
        let app = test_router(state);
        let mut config: serde_json::Value =
            serde_json::to_value(SparkConfig::default()).expect("config JSON");
        config["runtime"]["max_concurrency"] = serde_json::json!(4);
        let response = app
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .header(ACTION_HEADER, "1")
                    .uri("/api/config")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({"config": config}).to_string(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let path = dir.path().join("sparkflash.toml");
        let text = std::fs::read_to_string(&path).expect("saved TOML");
        assert!(text.contains("max_concurrency = 4"));
        let loaded = SparkConfig::load(&path).expect("reload");
        assert_eq!(loaded.runtime.max_concurrency, 4);
    }

    #[tokio::test]
    async fn an_invalid_update_is_refused_without_touching_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        let mut config: serde_json::Value =
            serde_json::to_value(SparkConfig::default()).expect("config JSON");
        config["runtime"]["max_concurrency"] = serde_json::json!(999);
        let response = app
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .header(ACTION_HEADER, "1")
                    .uri("/api/config")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({"config": config}).to_string(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(!dir.path().join("sparkflash.toml").exists());
    }

    #[tokio::test]
    async fn discovery_returns_configured_and_new_nodes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .header(ACTION_HEADER, "1")
                    .uri("/api/discover")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let nodes = body["nodes"].as_array().expect("nodes");
        assert_eq!(nodes.len(), 5);
        assert_eq!(body["discovered"], 1);
        assert!(nodes.iter().any(|node| node["ssh"] == "10.0.0.7"));
    }

    #[tokio::test]
    async fn launch_refuses_while_rank_zero_is_online() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state(dir.path());
        state.stats.ingest(
            &crate::stats::RawSnapshot {
                online: true,
                ready: true,
                metrics: crate::stats::MetricsSnapshot {
                    running: Some(0),
                    waiting: Some(0),
                    serving: Some(1),
                    ..crate::stats::MetricsSnapshot::default()
                },
                cluster: None,
                error: None,
                unhealthy: None,
            },
            1_000,
        );
        let app = test_router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .header(ACTION_HEADER, "1")
                    .uri("/api/actions/launch")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn an_invalid_existing_file_blocks_actions_instead_of_using_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state_with(dir.path(), true, Some("missing field `runtime`".to_owned()));
        let app = test_router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .header(ACTION_HEADER, "1")
                    .uri("/api/actions/launch")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = body_json(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("did not validate")
        );
    }

    #[tokio::test]
    async fn the_model_list_and_checkpoint_come_from_the_hub_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        crate::hf::tests::cache_model(&dir.path().join("hub"), "org/other", "abc", "llama");
        let app = test_router(test_state(dir.path()));
        let get = |uri: &str| {
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("request")
        };
        let models = body_json(
            app.clone()
                .oneshot(get("/api/models"))
                .await
                .expect("models"),
        )
        .await;
        assert_eq!(models["error"], serde_json::Value::Null);
        assert_eq!(models["models"][0]["id"], crate::config::DEFAULT_MODEL_ID);
        assert_eq!(models["models"][0]["supported"], true);
        assert_eq!(models["models"][1]["id"], "org/other");
        assert_eq!(models["models"][1]["supported"], false);
        let state = body_json(app.oneshot(get("/api/state")).await.expect("state")).await;
        assert_eq!(state["checkpoint"]["from_cache"], true);
        assert_eq!(
            state["checkpoint"]["path"],
            snapshot.display().to_string().as_str()
        );
    }

    #[tokio::test]
    async fn launch_and_convert_refuse_a_model_missing_from_the_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        for uri in ["/api/actions/launch", "/api/actions/convert"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .header(ACTION_HEADER, "1")
                        .header(header::CONTENT_TYPE, "application/json")
                        .uri(uri)
                        .body(Body::from("{}"))
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(response.status(), StatusCode::CONFLICT, "{uri}");
            let body = body_json(response).await;
            assert!(
                body["error"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("is not in the Hugging Face cache"),
                "{uri}: {body}"
            );
        }
    }

    async fn status_of(app: Router, request: Request<Body>) -> StatusCode {
        app.oneshot(request).await.expect("response").status()
    }

    #[tokio::test]
    async fn an_action_without_the_ui_header_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        for (method, uri) in [
            ("POST", "/api/actions/stop"),
            ("POST", "/api/actions/launch"),
            ("POST", "/api/discover"),
            ("POST", "/api/config/reload"),
            ("PUT", "/api/config"),
        ] {
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .body(Body::empty())
                .expect("request");
            assert_eq!(
                status_of(app.clone(), request).await,
                StatusCode::FORBIDDEN,
                "{method} {uri}"
            );
        }
        // Reads need no header.
        let read = Request::builder()
            .uri("/api/state")
            .body(Body::empty())
            .expect("request");
        assert_eq!(status_of(app, read).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn only_local_and_configured_host_names_are_answered() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state(dir.path());
        let hostname = state.hostname.clone();
        let app = test_router(state.clone());
        let get = |host: &str| {
            Request::builder()
                .uri("/api/state")
                .header(header::HOST, host)
                .body(Body::empty())
                .expect("request")
        };
        for host in [
            "127.0.0.1:8790".to_owned(),
            "[::1]:8790".to_owned(),
            "localhost:8790".to_owned(),
            hostname.clone(),
            format!("{hostname}.local:8790"),
        ] {
            assert_eq!(
                status_of(app.clone(), get(&host)).await,
                StatusCode::OK,
                "{host}"
            );
        }
        assert_eq!(
            status_of(app.clone(), get("attacker.example:8790")).await,
            StatusCode::MISDIRECTED_REQUEST
        );
        state
            .config
            .write()
            .expect("config lock")
            .config
            .web
            .allowed_hosts = vec!["spark-console.example".to_owned()];
        assert_eq!(
            status_of(app.clone(), get("spark-console.example")).await,
            StatusCode::OK
        );
        assert_eq!(
            status_of(app, get("attacker.example")).await,
            StatusCode::MISDIRECTED_REQUEST
        );
    }
}
