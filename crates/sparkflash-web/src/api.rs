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

use crate::commands::{
    CommandSpec, ServeAction, SyncTarget, SyncWhat, convert_command, download_command,
    fabric_probe_command, inspect_command, local_hostname, network_fix_command, prepare_command,
    prepared_probe_command, serve_action_command, sync_command,
};
use crate::config::{
    MAX_CONCURRENCY, MODEL_PROFILES, QWEN_CONTEXT_TOKENS, QWEN_ENGINE_ID, QWEN_MODEL_ID,
    QWEN_WEIGHTS_ID, SUPPORTED_ENGINES, SUPPORTED_WEIGHTS, SparkConfig,
};
use crate::discovery::{DiscoveryReport, NodeDiscovery, SshProbe, report_nodes};
use crate::fit::Cluster;
use crate::hf::{self, CachedModel, RemoteModel};
use crate::inspect::{self, Check, ProbeResult, SetupReport, Severity};
use crate::jobs::{JobManager, JobStatus, JobSummary};
use crate::stats::{MAX_SERIES_POINTS, StatsStore, TelemetrySource, now_ms};

/// The header every state-changing request must carry (the UI sends it). A browser adds a custom
/// header to a cross-origin request only after a CORS preflight, which this server never grants,
/// so another site open in the operator's browser cannot launch, stop or reconfigure the cluster.
pub const ACTION_HEADER: &str = "x-sparkflash-web";

/// The cluster actions that change whether the group runs; one at a time. A swap is the
/// in-process stop, switch and launch sequence, so it is exclusive with both. A fast-port setup
/// can drop the links a running group talks over, so it excludes these as well.
const CLUSTER_STATE_KINDS: &[&str] = &[
    "launch",
    "stop",
    "swap",
    "network-setup-r0",
    "network-setup-r1",
    "network-setup-r2",
    "network-setup-r3",
];

/// What a conversion must not overlap: a launch, stop or swap, or the checkpoint changing
/// under it.
const CONVERT_EXCLUSIVE: &[&str] = &[
    "launch",
    "stop",
    "swap",
    "download",
    "sync-model",
    "network-setup-r0",
    "network-setup-r1",
    "network-setup-r2",
    "network-setup-r3",
];

/// What a launch must not overlap: anything that changes the files the ranks start from.
/// Background `prepare-*` jobs are deliberately absent: sharding another model may continue
/// while the current one serves and a normal launch must not be blocked by it.
const LAUNCH_EXCLUSIVE: &[&str] = &[
    "launch",
    "stop",
    "swap",
    "download",
    "sync-model",
    "sync-software",
    "convert-r0",
    "convert-r1",
    "convert-r2",
    "convert-r3",
    // A port job can drop the link the ranks are starting over.
    "network-setup-r0",
    "network-setup-r1",
    "network-setup-r2",
    "network-setup-r3",
];

/// What a model download or copy must not overlap: the other one, or a conversion reading it.
const MODEL_EXCLUSIVE: &[&str] = &[
    "download",
    "sync-model",
    "swap",
    "convert-r0",
    "convert-r1",
    "convert-r2",
    "convert-r3",
    "prepare-r0",
    "prepare-r1",
    "prepare-r2",
    "prepare-r3",
    // A port job can drop the link a model transfer is running over.
    "network-setup-r0",
    "network-setup-r1",
    "network-setup-r2",
    "network-setup-r3",
];

/// How long one node check may take, on top of the SSH connect timeout.
const INSPECT_SECS: u64 = 60;

/// How long the per-rank link probe may take, on top of the SSH connect timeout. The probe
/// ARPs and pings every fast address of every peer from every fast port, and each unanswered
/// attempt can burn its one-second wait: two ports, two addresses per peer and three peers is
/// twelve ARP attempts plus twelve pings in the worst case. The deadline must outlast that, or a
/// disconnected fabric would be reported as an unfinished probe instead of a measured one.
const FABRIC_SECS: u64 = 60;

/// What a fast-port setup must not overlap: a launch or stop, and the jobs that move model bytes,
/// since reconfiguring the ports can drop in-flight transfers.
const NETWORK_SETUP_EXCLUSIVE: &[&str] = &[
    "launch",
    "stop",
    "swap",
    "download",
    "sync-model",
    "sync-software",
    // They read model bytes over the same links and can be cut by a netplan apply.
    "convert-r0",
    "convert-r1",
    "convert-r2",
    "convert-r3",
    "prepare-r0",
    "prepare-r1",
    "prepare-r2",
    "prepare-r3",
];

/// How old the last node check may be when its port fixes are applied. The fixes carry the
/// addresses and routes the check measured and run as root, so a stale report must not be
/// replayed onto a cluster that has since changed under it.
const NETWORK_SETUP_MAX_AGE_MS: u64 = 10 * 60 * 1000;

/// How long after a launch or swap the fast ports are left alone. The launcher job finishes when
/// the script returns, while rank 0 binds about half a minute later and the telemetry takes
/// another sample, so "rank 0 is online" is not yet a usable gate in that window.
const LAUNCH_SETTLE_MS: u64 = 5 * 60 * 1000;

/// How long a stop or its confirmation through the launcher may take.
const STOP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// How long the launcher's start-up survival check may take.
const LAUNCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

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
    last_inspect: Arc<RwLock<Option<SetupReport>>>,
    /// The configuration the cached report was taken for. A saved change makes the automatic
    /// provisioner inspect once, instead of reusing a report that describes other nodes or paths.
    last_inspect_config: Arc<RwLock<Option<SparkConfig>>>,
    inspect_lock: Arc<tokio::sync::Mutex<()>>,
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
            last_inspect: Arc::new(RwLock::new(None)),
            last_inspect_config: Arc::new(RwLock::new(None)),
            inspect_lock: Arc::new(tokio::sync::Mutex::new(())),
            advertised: init.advertised,
            hostname: local_hostname(),
        }
    }

    /// The current configuration.
    #[must_use]
    pub fn config(&self) -> SparkConfig {
        self.config.read().expect("config lock").config.clone()
    }

    /// The cluster a model is judged against.
    ///
    /// Its size is the most Sparks any source names: the plan, the last discovery (configured
    /// and newly found hosts) or the telemetry, because a TP=1 plan still sits on a larger
    /// cluster. Each node is sized by the smallest unified-memory total the telemetry reports,
    /// or the GB10 ceiling before any rank has reported.
    #[must_use]
    pub fn cluster(&self, config: &SparkConfig) -> Cluster {
        let telemetry = self.stats.cluster();
        let discovered = self
            .last_discovery
            .read()
            .expect("discovery lock")
            .as_ref()
            .map_or(0, |report| report.nodes.len());
        let reported = telemetry.as_ref().map_or(0, |cluster| cluster.nodes.len());
        let nodes = config.cluster.nodes.len().max(discovered).max(reported);
        Cluster::nominal(nodes).measured(
            telemetry
                .iter()
                .flat_map(|cluster| cluster.nodes.iter().map(|node| node.uma_total_gib)),
        )
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

    /// The process manager, for the automatic provisioner.
    #[must_use]
    pub fn jobs(&self) -> Arc<JobManager> {
        Arc::clone(&self.jobs)
    }

    /// Whether a configuration file exists and validated.
    #[must_use]
    pub fn config_exists(&self) -> bool {
        let state = self.config.read().expect("config lock");
        state.exists && state.load_error.is_none()
    }

    /// Runs the read-only node check and keeps the report for the state view.
    ///
    /// # Errors
    /// Returns why the configuration cannot be used or the check task failed.
    pub async fn inspect_now(&self) -> Result<SetupReport, String> {
        let _guard = self.inspect_lock.lock().await;
        let config = self.usable_config()?;
        let probe = Arc::clone(&self.ssh);
        let inspected_for = config.clone();
        let result = tokio::task::spawn_blocking(move || run_inspect(&config, probe.as_ref()))
            .await
            .map_err(|error| format!("the node check failed: {error}"))?;
        *self.last_inspect.write().expect("inspect lock") = Some(result.clone());
        *self.last_inspect_config.write().expect("inspect lock") = Some(inspected_for);
        Ok(result)
    }

    /// One automatic provisioning pass:
    ///
    /// 1. build the local GPU kernels and fetch NCCL when they are missing, so a fresh checkout
    ///    does not need `scripts/setup.sh`;
    /// 2. inspect every Spark;
    /// 3. copy this process's binary, kernels and NCCL to any reachable Spark whose copy differs.
    ///
    /// The pass is deliberately idempotent and never loops on a failed attempt: a bootstrap job
    /// that already exists (running or failed) is left for the operator to retry from the UI.
    ///
    /// # Errors
    /// Only when the configuration cannot be used or the node check itself fails.
    pub async fn provision_once(&self) -> Result<(), String> {
        if !self.config_exists() {
            return Ok(());
        }
        let config = self.usable_config()?;
        if config.is_solo() || !config.binary_path().is_file() {
            return Ok(());
        }
        let jobs = self.jobs.list();
        let seen = |kind: &str| jobs.iter().any(|job| job.kind == kind);
        // A failed automatic deploy is not retried in a loop; the Setup page's Deploy button is
        // the deliberate retry.
        if jobs
            .iter()
            .find(|job| job.kind == "sync-software")
            .is_some_and(|job| job.status == JobStatus::Failed)
        {
            return Ok(());
        }
        if config.requires_cubins()
            && !has_cubins(&config.cubins_path())
            && !seen("bootstrap-cubins")
        {
            let script = config.repo_root().join("scripts/build-cubins.sh");
            if script.is_file() {
                let command = CommandSpec {
                    program: "bash".to_owned(),
                    args: vec![
                        script.display().to_string(),
                        config.cubins_path().display().to_string(),
                    ],
                    env: std::collections::BTreeMap::new(),
                    cwd: Some(config.repo_root()),
                    label: "Build the GPU kernels".to_owned(),
                    kind: "bootstrap-cubins".to_owned(),
                };
                let _ = self.jobs.start_exclusive(command, &["launch"]).await;
                return Ok(());
            }
        }
        if let Some(nccl) = config.nccl_lib_path()
            && !nccl.join("libnccl.so.2").is_file()
            && !seen("bootstrap-nccl")
        {
            let script = config.repo_root().join("scripts/fetch-nccl.sh");
            if script.is_file() {
                let command = CommandSpec {
                    program: "bash".to_owned(),
                    args: vec![script.display().to_string(), nccl.display().to_string()],
                    env: std::collections::BTreeMap::new(),
                    cwd: Some(config.repo_root()),
                    label: "Fetch the NCCL library".to_owned(),
                    kind: "bootstrap-nccl".to_owned(),
                };
                let _ = self.jobs.start_exclusive(command, &["launch"]).await;
                return Ok(());
            }
        }

        // The check hashes the server binary, the cubin tree and NCCL on every Spark. Do that
        // once at boot and reuse the report until a software job changes something or the saved
        // configuration changes; otherwise this ten-second loop puts a SHA-256 pass over a
        // several-hundred-megabyte binary on the CPU forever.
        let cached = {
            let report = self.last_inspect.read().expect("inspect lock").clone();
            let inspected_for = self
                .last_inspect_config
                .read()
                .expect("inspect lock")
                .clone();
            report.filter(|_| inspected_for.as_ref() == Some(&config))
        };
        // A port pass rewrites the fast ports; a check in the middle of it records a cluster that
        // is half-configured. The next ten-second pass takes the report instead.
        let report = match cached {
            Some(report) if !software_changed_since(&jobs, report.checked_ms) => report,
            _ if self
                .jobs
                .running_in(&[
                    "network-setup-r0",
                    "network-setup-r1",
                    "network-setup-r2",
                    "network-setup-r3",
                ])
                .is_some() =>
            {
                return Ok(());
            }
            _ => self.inspect_now().await?,
        };
        let lead_ready = report
            .nodes
            .first()
            .is_some_and(|node| node.software == Check::Ready);
        let needs_copy = report.nodes.iter().skip(1).any(|node| {
            node.reachable
                && matches!(
                    node.software,
                    Check::Missing | Check::Outdated | Check::Partial
                )
        });
        if !lead_ready || !needs_copy {
            return Ok(());
        }
        let command = sync_command(&config, SyncWhat::Software, &sync_targets(self, &config))?;
        // The copy runs over the fast addresses, so a port pass that resets those ports must stop
        // it: the same exclusion the manual Deploy button uses, in both directions.
        let _ = self
            .jobs
            .start_exclusive(
                command,
                &[
                    "launch",
                    "sync-software",
                    "network-setup-r0",
                    "network-setup-r1",
                    "network-setup-r2",
                    "network-setup-r3",
                ],
            )
            .await;
        Ok(())
    }

    /// Retry the local preparation the automatic provisioner does: build missing cubins and fetch
    /// a missing NCCL library. Unlike the automatic pass, this deliberately retries a previous
    /// failed attempt because an operator asked for it.
    ///
    /// # Errors
    /// When the configuration cannot be used or there is nothing left to prepare.
    pub async fn prepare_local(&self) -> Result<Vec<u64>, String> {
        let config = self.usable_config()?;
        let mut started = Vec::new();
        let mut failures = Vec::new();
        if config.requires_cubins() && !has_cubins(&config.cubins_path()) {
            let script = config.repo_root().join("scripts/build-cubins.sh");
            if script.is_file() {
                let command = CommandSpec {
                    program: "bash".to_owned(),
                    args: vec![
                        script.display().to_string(),
                        config.cubins_path().display().to_string(),
                    ],
                    env: std::collections::BTreeMap::new(),
                    cwd: Some(config.repo_root()),
                    label: "Build the GPU kernels".to_owned(),
                    kind: "bootstrap-cubins".to_owned(),
                };
                match self.jobs.start_exclusive(command, &["launch"]).await {
                    Ok(id) => started.push(id),
                    Err(error) => failures.push(error.message),
                }
            } else {
                failures.push(format!("{} is missing", script.display()));
            }
        }
        if let Some(nccl) = config.nccl_lib_path()
            && !nccl.join("libnccl.so.2").is_file()
        {
            let script = config.repo_root().join("scripts/fetch-nccl.sh");
            if script.is_file() {
                let command = CommandSpec {
                    program: "bash".to_owned(),
                    args: vec![script.display().to_string(), nccl.display().to_string()],
                    env: std::collections::BTreeMap::new(),
                    cwd: Some(config.repo_root()),
                    label: "Fetch the NCCL library".to_owned(),
                    kind: "bootstrap-nccl".to_owned(),
                };
                match self.jobs.start_exclusive(command, &["launch"]).await {
                    Ok(id) => started.push(id),
                    Err(error) => failures.push(error.message),
                }
            } else {
                failures.push(format!("{} is missing", script.display()));
            }
        }
        if started.is_empty() {
            return Err(if failures.is_empty() {
                "the local server is already prepared; use Deploy for the other Sparks".to_owned()
            } else {
                failures.join("; ")
            });
        }
        Ok(started)
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

/// Whether a job that changes the local software finished after `checked_ms`.
///
/// The automatic provisioner re-runs the node check only for these jobs. A cached report is
/// otherwise reused, so the SHA-256 pass over the server binary is a startup cost, not a poll.
fn software_changed_since(jobs: &[JobSummary], checked_ms: u64) -> bool {
    jobs.iter().any(|job| {
        matches!(
            job.kind.as_str(),
            "bootstrap-cubins" | "bootstrap-nccl" | "sync-software"
        ) && job
            .finished_ms
            .is_some_and(|finished| finished > checked_ms)
    })
}

/// Whether `directory` holds at least one `.cubin` file.
fn has_cubins(directory: &std::path::Path) -> bool {
    std::fs::read_dir(directory).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("cubin"))
        })
    })
}

/// The other Sparks a copy goes to, addressed over the fast network once the last check proved it.
fn sync_targets(web: &WebState, config: &SparkConfig) -> Vec<SyncTarget> {
    let fast = web
        .last_inspect
        .read()
        .expect("inspect lock")
        .as_ref()
        .is_some_and(|report| {
            report.network_ok()
                && report
                    .plan
                    .as_ref()
                    .is_some_and(|plan| plan.matches(config))
        });
    config
        .cluster
        .nodes
        .iter()
        .filter(|node| !node.local)
        .map(|node| SyncTarget {
            address: if fast {
                node.peer.clone()
            } else {
                node.ssh.clone()
            },
            ssh: node.ssh.clone(),
        })
        .collect()
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
        .route("/api/inspect", post(inspect_nodes))
        .route("/api/network/apply", post(apply_network))
        .route("/api/network/setup", post(network_setup))
        .route("/api/models", get(models))
        .route("/api/models/search", get(models_search))
        .route("/api/models/inspect", get(models_inspect))
        .route("/api/series", get(series))
        .route("/api/jobs", get(list_jobs))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/actions/convert", post(convert))
        .route("/api/actions/download", post(download))
        .route("/api/actions/prepare", post(prepare))
        .route("/api/actions/prepare-model", post(prepare_model))
        .route("/api/actions/swap", post(swap))
        .route("/api/actions/sync", post(sync))
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
    /// This node's hostname, which the setup page names rank 0 after.
    hostname: String,
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
    setup: Option<SetupReport>,
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
    /// Whether that directory exists right now.
    exists: bool,
    /// Whether it is looked up in the Hugging Face cache rather than set in `model.checkpoint`.
    from_cache: bool,
    /// The hub cache `model.id` is looked up in.
    cache: Option<String>,
    /// The cached entry for `model.id`, when it has one. `complete` says whether the snapshot
    /// is ready to serve; `bytes` says how much is already downloaded.
    cached: Option<CachedModel>,
    /// Why it does not resolve.
    error: Option<String>,
}

impl CheckpointView {
    fn of(config: &SparkConfig) -> Self {
        let from_cache = config.model.checkpoint.trim().is_empty();
        let cache = config.hub_cache_dir();
        let cached = if from_cache {
            cache
                .as_deref()
                .and_then(|dir| hf::lookup(dir, &config.model.id))
        } else {
            None
        };
        let resolved = config.resolve_checkpoint();
        let exists = resolved.as_ref().map(|path| path.is_dir()).unwrap_or(false);
        Self {
            path: resolved
                .as_ref()
                .ok()
                .map(|path| path.display().to_string()),
            exists,
            from_cache,
            cache: cache.map(|dir| dir.display().to_string()),
            cached,
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
    /// Every model family this build can configure, including one not downloaded yet.
    models: Vec<SupportedModel>,
    /// The one-node TP=1 profile the Setup page offers as "run on this Spark only".
    solo: SoloDefaults,
}

#[derive(Serialize)]
struct SupportedModel {
    id: &'static str,
    weights: &'static str,
    engine: &'static str,
    context_tokens: usize,
    requires_conversion: bool,
}

#[derive(Serialize)]
struct SoloDefaults {
    weights: &'static str,
    engine: &'static str,
    id: &'static str,
    context_tokens: usize,
}

async fn state_snapshot(State(web): State<WebState>) -> Json<StateView> {
    let (config, config_error, config_exists) = {
        let state = web.config.read().expect("config lock");
        (state.config.clone(), state.load_error.clone(), state.exists)
    };
    let discovery = web.last_discovery.read().expect("discovery lock").clone();
    let setup = web.last_inspect.read().expect("inspect lock").clone();
    let checkpoint = CheckpointView::of(&config);
    let limits = Limits {
        max_context_tokens: config.max_context_tokens(),
        max_concurrency: MAX_CONCURRENCY,
        world_size: config.world_size(),
    };
    Json(StateView {
        now_ms: now_ms(),
        hostname: web.hostname.clone(),
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
        setup,
        advertised: web.advertised.clone(),
        checkpoint,
        limits,
        supported: Supported {
            weights: SUPPORTED_WEIGHTS,
            engines: SUPPORTED_ENGINES,
            models: MODEL_PROFILES
                .iter()
                .map(|profile| SupportedModel {
                    id: profile.id,
                    weights: profile.weights,
                    engine: profile.engine,
                    context_tokens: profile.context_tokens,
                    requires_conversion: profile.requires_conversion,
                })
                .collect(),
            solo: SoloDefaults {
                weights: QWEN_WEIGHTS_ID,
                engine: QWEN_ENGINE_ID,
                id: QWEN_MODEL_ID,
                context_tokens: QWEN_CONTEXT_TOKENS,
            },
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
    store_config(&web, config)
}

/// Validates, saves and adopts `config`, answering with the saved view.
fn store_config(web: &WebState, config: SparkConfig) -> Response {
    let mut config = config;
    config.discard_stale_context_tokens();
    if let Err((status, message)) = save_and_adopt(web, config.clone()) {
        return api_error(status, message);
    }
    Json(ConfigView {
        path: web.config_path.display().to_string(),
        exists: true,
        load_error: None,
        config,
    })
    .into_response()
}

/// The save-and-adopt half of [`store_config`], shared with the swap task, which has no HTTP
/// response to build. The error keeps the status `store_config` should answer with: a bad value
/// is the caller's, a failed write is ours.
fn save_and_adopt(web: &WebState, config: SparkConfig) -> Result<(), (StatusCode, String)> {
    let mut config = config;
    config.discard_stale_context_tokens();
    config
        .validate()
        .map_err(|error| (StatusCode::BAD_REQUEST, error.message))?;
    config
        .save(&web.config_path)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.message))?;
    {
        let mut state = web.config.write().expect("config lock");
        state.config = config.clone();
        state.load_error = None;
        state.exists = true;
    }
    *web.telemetry.write().expect("telemetry lock") =
        Arc::new(TelemetrySource::new(&config.web.serve_url));
    Ok(())
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

/// Checks every node (in parallel, read-only) and keeps the report for the state view.
async fn inspect_nodes(State(web): State<WebState>) -> Response {
    // The automatic check after a port pass covers this; a manual check in the middle would store
    // a half-applied cluster as the report the port fixes replay.
    if let Some(id) = web.jobs.running_in(&[
        "network-setup-r0",
        "network-setup-r1",
        "network-setup-r2",
        "network-setup-r3",
    ]) {
        return api_error(
            StatusCode::CONFLICT,
            format!("port setup job {id} is running; check the Sparks when it finishes"),
        );
    }
    match web.inspect_now().await {
        Ok(report) => Json(report).into_response(),
        Err(message) => api_error(StatusCode::CONFLICT, message),
    }
}

fn run_inspect(config: &SparkConfig, probe: &dyn SshProbe) -> SetupReport {
    let checkpoint = config.resolve_checkpoint().ok();
    let timeout = std::time::Duration::from_secs(config.web.ssh_timeout_secs + INSPECT_SECS);
    let probes: Vec<ProbeResult> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..config.cluster.nodes.len())
            .map(|rank| {
                let command = inspect_command(config, rank, checkpoint.as_deref());
                scope.spawn(move || {
                    probe
                        .capture(&command, timeout)
                        .map(|text| inspect::parse_facts(&text))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err("the check stopped unexpectedly".to_owned()))
            })
            .collect()
    });
    let fabric = run_fabric_probe(config, probe, &probes);
    inspect::analyze_fabric(
        config,
        &probes,
        fabric,
        lead_model_complete(config),
        now_ms(),
    )
}

/// Measures which fast port reaches which Spark: one read-only probe per node, run in parallel,
/// `None` when the ranks have no usable fast port to probe from.
fn run_fabric_probe(
    config: &SparkConfig,
    probe: &dyn SshProbe,
    probes: &[ProbeResult],
) -> Option<inspect::Fabric> {
    let addresses = inspect::probe_addresses(config, probes)?;
    let primaries = inspect::planned_peers(config, probes)?;
    let world = probes.len();
    let timeout = std::time::Duration::from_secs(config.web.ssh_timeout_secs + FABRIC_SECS);
    let results: Vec<Option<inspect::FabricProbe>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let peers: Vec<(usize, String)> = addresses
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != rank)
                    .flat_map(|(index, list)| {
                        list.iter().map(move |address| (index, address.clone()))
                    })
                    .collect();
                let pings: Vec<(usize, String)> = primaries
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != rank)
                    .map(|(index, address)| (index, address.clone()))
                    .collect();
                let rails = probes[rank]
                    .as_ref()
                    .map(|facts| facts.rails())
                    .unwrap_or_default();
                // One attempt per netdev: the two halves of a cabled port are distinct netdevs
                // with their own addresses, and both answer the same neighbour.
                let mut netdevs: Vec<String> =
                    rails.iter().map(|rail| rail.netdev.clone()).collect();
                netdevs.sort();
                netdevs.dedup();
                let command = fabric_probe_command(config, rank, &netdevs, &peers, &pings);
                scope.spawn(move || {
                    probe
                        .capture(&command, timeout)
                        .ok()
                        .map(|text| inspect::parse_fabric(rank, &text))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or(None))
            .collect()
    });
    Some(inspect::fabric_from_probes(results, world))
}

/// Whether the console believes the group is up: a launch or swap that succeeded and no stop since.
/// The telemetry sample is the other signal; this one survives a sampler gap, a console that has
/// just restarted, and a serve URL the sampler cannot read.
fn cluster_believed_running(jobs: &[JobSummary]) -> bool {
    let finished = |kinds: &[&str], succeeded_only: bool| {
        jobs.iter()
            .filter(|job| kinds.contains(&job.kind.as_str()))
            .filter(|job| !succeeded_only || job.status == JobStatus::Succeeded)
            .filter_map(|job| job.finished_ms)
            .max()
    };
    let launched = finished(&["launch", "swap"], true);
    let stopped = finished(&["stop"], false);
    launched.is_some_and(|launched| stopped.is_none_or(|stopped| launched > stopped))
}

/// Whether the lead holds the whole checkpoint: a download left no `.incomplete` blob behind, or
/// `model.checkpoint` names a directory that exists.
fn lead_model_complete(config: &SparkConfig) -> bool {
    if !config.model.checkpoint.trim().is_empty() {
        return config.node_path(&config.model.checkpoint).is_dir();
    }
    config
        .hub_cache_dir()
        .and_then(|cache| hf::lookup(&cache, &config.model.id))
        .is_some_and(|model| model.complete)
}

/// Writes the NCCL settings and peer addresses the last check derived into the configuration.
async fn apply_network(State(web): State<WebState>) -> Response {
    let plan = web
        .last_inspect
        .read()
        .expect("inspect lock")
        .as_ref()
        .and_then(|report| report.plan.clone());
    let Some(plan) = plan else {
        return api_error(
            StatusCode::CONFLICT,
            "there are no detected network settings yet; check the Sparks first",
        );
    };
    let mut config = match web.usable_config() {
        Ok(config) => config,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    // The plan is what the check measured; replaying it onto a configuration that has changed
    // since would write another cluster's addresses, exactly as the port setup refuses to do.
    let inspected_for = web
        .last_inspect_config
        .read()
        .expect("inspect lock")
        .clone();
    if inspected_for.as_ref() != Some(&config) {
        return api_error(
            StatusCode::CONFLICT,
            "the last node check was taken of a different configuration; check the nodes again",
        );
    }
    plan.apply(&mut config);
    store_config(&web, config)
}

/// Applies the fast-port configuration the last check computed on every node that needs it: one
/// SSH job per node running under `sudo -n`. Refuses when nothing is waiting to be applied.
async fn network_setup(State(web): State<WebState>) -> Response {
    let config = match web.usable_config() {
        Ok(config) => config,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    // The fixes are the ones the last check computed, including their netplan bodies. Replaying
    // them against a configuration that has since changed would write another machine's
    // addresses to this rank, so a report taken for a different configuration is refused.
    let report = web.last_inspect.read().expect("inspect lock").clone();
    let inspected_for = web
        .last_inspect_config
        .read()
        .expect("inspect lock")
        .clone();
    let Some(report) = report.filter(|_| inspected_for.as_ref() == Some(&config)) else {
        return api_error(
            StatusCode::CONFLICT,
            "the last node check was taken of a different configuration; check the nodes again",
        );
    };
    // The config was not the only thing that could change since the check: an address, a lease or
    // a cable could have. Ten minutes is long enough to read the report and click the button.
    if now_ms().saturating_sub(report.checked_ms) > NETWORK_SETUP_MAX_AGE_MS {
        return api_error(
            StatusCode::CONFLICT,
            "the last node check is more than ten minutes old; check the nodes again before \
             writing its fixes",
        );
    }
    // netplan apply drops the fast ports for a moment, so a serving cluster would lose the links
    // its collectives are mid-transfer on. Rank 0 answering is the positive signal; the job
    // history covers a console that has not sampled yet, or one whose telemetry cannot reach it.
    if web.stats.latest().is_some_and(|sample| sample.online) {
        return api_error(
            StatusCode::CONFLICT,
            "rank 0 is online; stop the cluster before reconfiguring the fast ports",
        );
    }
    // A launch that just finished is still booting: netplan apply now would drop the links the
    // group is coming up over. The telemetry gate above only fires once the sampler has seen the
    // server, which is later than the job ending. Only a launch the console still believes in
    // counts: a stop since (or a launch that failed) means nothing is coming up.
    let believed = cluster_believed_running(&web.jobs.list());
    let launched_ms = web
        .jobs
        .list()
        .iter()
        .filter(|job| {
            matches!(job.kind.as_str(), "launch" | "swap") && job.status == JobStatus::Succeeded
        })
        .filter_map(|job| job.finished_ms)
        .max();
    if believed
        && launched_ms.is_some_and(|finished| now_ms().saturating_sub(finished) < LAUNCH_SETTLE_MS)
    {
        return api_error(
            StatusCode::CONFLICT,
            "the cluster was launched less than five minutes ago and may still be coming up; wait until it answers or stop it before reconfiguring the fast ports",
        );
    }
    if believed {
        return api_error(
            StatusCode::CONFLICT,
            "the cluster was launched and has not been stopped; stop it before reconfiguring the fast ports",
        );
    }
    // A port job holds its rank's own kind, so it cannot start twice; the other ranks of the
    // same pass have different kinds, so a second click would start them again while the first
    // pass is still running. One pass at a time.
    if let Some(id) = web.jobs.running_in(&[
        "network-setup-r0",
        "network-setup-r1",
        "network-setup-r2",
        "network-setup-r3",
    ]) {
        return api_error(
            StatusCode::CONFLICT,
            format!("port setup job {id} is already running"),
        );
    }
    let fixes: Vec<(usize, String)> = report
        .nodes
        .iter()
        .filter_map(|node| {
            node.network_fix_remote
                .clone()
                .map(|remote| (node.rank, remote))
        })
        .collect();
    if fixes.is_empty() {
        // Nothing to apply. That is only "the ports already match" when the check compared every
        // node: one that did not answer, or one whose problem no port fix can write, leaves this
        // button with nothing to do and must be named instead of certified.
        let waiting: Vec<String> = report
            .nodes
            .iter()
            .filter(|node| {
                !node.reachable
                    || (node.network_fix.is_none()
                        && report.findings.iter().any(|finding| {
                            finding.rank == Some(node.rank) && finding.severity == Severity::Error
                        }))
            })
            .map(|node| node.hostname.clone().unwrap_or_else(|| node.ssh.clone()))
            .collect();
        // A cluster-wide error (two Sparks on one address, for instance) is a problem no port
        // fix can write either, so "already match" would be wrong for it too.
        let cluster_error = report
            .findings
            .iter()
            .find(|finding| finding.rank.is_none() && finding.severity == Severity::Error)
            .map(|finding| finding.message.clone());
        if !waiting.is_empty() {
            return api_error(
                StatusCode::CONFLICT,
                format!(
                    "the last check still reports a problem on {}; fix it and check the nodes again",
                    waiting.join(", ")
                ),
            );
        }
        if let Some(message) = cluster_error {
            return api_error(
                StatusCode::CONFLICT,
                format!("the last check still reports a problem: {message}"),
            );
        }
        return api_error(
            StatusCode::CONFLICT,
            "the fast ports already match the plan; check the nodes to recompute",
        );
    }
    let mut started = Vec::new();
    for (rank, remote) in fixes {
        let Some(command) = network_fix_command(&config, rank, &remote) else {
            return api_error(
                StatusCode::CONFLICT,
                format!("rank {rank} is not in the saved configuration any more"),
            );
        };
        match web
            .jobs
            .start_exclusive(command, NETWORK_SETUP_EXCLUSIVE)
            .await
        {
            Ok(id) => started.push(id),
            Err(error) => {
                // Earlier ranks are already running their jobs; say so rather than dropping the
                // ids silently.
                return api_error(
                    StatusCode::CONFLICT,
                    format!(
                        "{} ({} job(s) had already started and are still running)",
                        error.message,
                        started.len()
                    ),
                );
            }
        }
    }
    accepted_jobs(&web, &started)
}

/// Downloads the configured model into the lead's Hugging Face cache.
/// The download action's optional body.
#[derive(Debug, Default, Deserialize)]
struct DownloadRequest {
    /// A model the browser picked, downloaded without first saving it into the plan.
    #[serde(default)]
    id: Option<String>,
}

/// Downloads the configured model, or the browsed model a body's `id` names, into the lead's
/// Hugging Face cache.
///
/// A browsed id is downloaded only when a device engine serves it: the plan cannot name a model
/// that is not cached yet, so this is how a Hub pick becomes selectable, and a model this build
/// cannot serve is refused with what it is missing instead of costing a multi-hundred-gigabyte
/// transfer.
async fn download(State(web): State<WebState>, body: axum::body::Bytes) -> Response {
    let request: DownloadRequest = if body.iter().all(u8::is_ascii_whitespace) {
        DownloadRequest::default()
    } else {
        match serde_json::from_slice(&body) {
            Ok(request) => request,
            Err(error) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    format!("the download request: {error}"),
                );
            }
        }
    };
    let mut config = match web.usable_config() {
        Ok(config) => config,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    if let Some(id) = request
        .id
        .map(|id| id.trim().to_owned())
        .filter(|id| !id.is_empty())
    {
        let cache = config.hub_cache_dir();
        let cluster = web.cluster(&config);
        let wanted = id.clone();
        let inspected =
            tokio::task::spawn_blocking(move || hf::inspect(cache.as_deref(), &wanted, &cluster))
                .await
                .unwrap_or_else(|error| Err(format!("the model inspection failed: {error}")));
        match inspected {
            Ok(model) if model.cached && model.complete => {
                return api_error(
                    StatusCode::CONFLICT,
                    format!("{id} is already in the Hugging Face cache; nothing to download"),
                );
            }
            Ok(model) if !model.supported => {
                return api_error(
                    StatusCode::CONFLICT,
                    format!(
                        "{id} is not a model this build can serve; it is missing {}",
                        if model.missing_parts.is_empty() {
                            "a device engine".to_owned()
                        } else {
                            model.missing_parts.join("; ")
                        }
                    ),
                );
            }
            Ok(_) => {}
            Err(message) => return api_error(StatusCode::CONFLICT, message),
        }
        config.model.id = id;
        config.model.checkpoint.clear();
    } else if config.model.checkpoint.trim().is_empty() && lead_model_complete(&config) {
        // The cache is the source of truth before the download starts. A complete snapshot needs
        // no network transfer, and saying so is more useful than starting a job that no-ops.
        return api_error(
            StatusCode::CONFLICT,
            format!(
                "{} is already in the Hugging Face cache; nothing to download",
                config.model.id
            ),
        );
    }
    let command = match download_command(&config) {
        Ok(command) => command,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    match web.jobs.start_exclusive(command, MODEL_EXCLUSIVE).await {
        Ok(id) => accepted_jobs(&web, &[id]),
        Err(error) => api_error(StatusCode::CONFLICT, error.message),
    }
}

/// Retries the local build/fetch the automatic provisioner performs.
async fn prepare(State(web): State<WebState>) -> Response {
    match web.prepare_local().await {
        Ok(ids) => accepted_jobs(&web, &ids),
        Err(message) => api_error(StatusCode::CONFLICT, message),
    }
}

#[derive(Deserialize)]
struct SyncRequest {
    what: SyncWhat,
}

/// Copies the software or the model from the lead to the other nodes.
async fn sync(State(web): State<WebState>, Json(request): Json<SyncRequest>) -> Response {
    let config = match web.usable_config() {
        Ok(config) => config,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    match request.what {
        SyncWhat::Software if !config.binary_path().is_file() => {
            return api_error(
                StatusCode::CONFLICT,
                format!(
                    "{} is not built yet; restart the console from the checkout so it can adopt the running binary",
                    config.binary_path().display()
                ),
            );
        }
        SyncWhat::Model if !lead_model_complete(&config) => {
            return api_error(
                StatusCode::CONFLICT,
                "the model is not completely on this Spark yet; download it first",
            );
        }
        _ => {}
    }
    // The fast network carries the copy once a check has confirmed the configured addresses.
    let targets = sync_targets(&web, &config);
    let command = match sync_command(&config, request.what, &targets) {
        Ok(command) => command,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    let exclusive: &[&str] = match request.what {
        // A port job can cut the link the copy is running over.
        SyncWhat::Software => &[
            "launch",
            "swap",
            "network-setup-r0",
            "network-setup-r1",
            "network-setup-r2",
            "network-setup-r3",
        ],
        SyncWhat::Model => MODEL_EXCLUSIVE,
    };
    match web.jobs.start_exclusive(command, exclusive).await {
        Ok(id) => accepted_jobs(&web, &[id]),
        Err(error) => api_error(StatusCode::CONFLICT, error.message),
    }
}

/// The models in the Hugging Face cache, for the model selector.
#[derive(Serialize)]
struct ModelsView {
    cache: Option<String>,
    /// The cluster every model's fit was judged against.
    cluster: Cluster,
    models: Vec<CachedModel>,
    /// The cached models that can be prepared and swapped from the Overview page.
    shelf: Vec<ShelfModel>,
    /// The families the browser lists on its side.
    families: Vec<hf::Family>,
    error: Option<String>,
}

/// The model-browser search result.
#[derive(Serialize)]
struct ModelSearchView {
    query: String,
    family: Option<String>,
    /// The cluster every result's fit was judged against.
    cluster: Cluster,
    models: Vec<RemoteModel>,
    error: Option<String>,
}

/// The selected Hub model's inspected configuration.
#[derive(Serialize)]
struct ModelInspectView {
    /// The cluster the fit was judged against.
    cluster: Cluster,
    model: Option<RemoteModel>,
    error: Option<String>,
}

/// The model browser's search parameters.
#[derive(Deserialize)]
struct ModelSearchQuery {
    /// What to search for; empty lists cached models or a family's default term.
    q: Option<String>,
    /// The family label or weight selector to filter by.
    family: Option<String>,
    /// How many Hub hits to ask for.
    limit: Option<usize>,
}

/// The model browser's inspect parameter.
#[derive(Deserialize)]
struct ModelInspectQuery {
    /// The Hugging Face model id to inspect.
    id: String,
}

/// One cached model's readiness to be selected without editing the configuration.
#[derive(Serialize)]
struct ShelfModel {
    /// The Hugging Face model id.
    id: String,
    /// The control-plane weight selector.
    weights: Option<String>,
    /// The engine selector.
    engine: Option<String>,
    /// The checkpoint's size in bytes.
    bytes: u64,
    /// Whether the download is complete.
    complete: bool,
    /// Whether launch needs per-rank converted weights.
    requires_conversion: bool,
    /// Whether this is the model the configuration currently names.
    active: bool,
    /// Whether this cluster plan can run the model at all.
    compatible: bool,
    /// Whether every rank already has what a launch needs.
    prepared: bool,
    /// The ranks whose probe reported ready.
    prepared_ranks: Vec<usize>,
    /// The number of ranks in this plan.
    world_size: usize,
    /// A job is currently sharding this model.
    preparing: bool,
    /// Why this model cannot be swapped to, when it cannot.
    note: Option<String>,
}

async fn models(State(web): State<WebState>) -> Json<ModelsView> {
    let config = web.config();
    let cache = config.hub_cache_dir();
    let cluster = web.cluster(&config);
    let listed = match cache.clone() {
        Some(dir) => tokio::task::spawn_blocking(move || {
            hf::list(&dir).map(|mut models| {
                hf::judge_cached(&mut models, &cluster);
                models
            })
        })
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
    let shelf = model_shelf(&web, &config, &models).await;
    Json(ModelsView {
        cache: cache.map(|dir| dir.display().to_string()),
        cluster,
        models,
        shelf,
        families: hf::families(),
        error,
    })
}

/// Search the Hub and the local cache for the model browser.
async fn models_search(
    State(web): State<WebState>,
    Query(query): Query<ModelSearchQuery>,
) -> Json<ModelSearchView> {
    let config = web.config();
    let cache = config.hub_cache_dir();
    let cluster = web.cluster(&config);
    let text = query.q.unwrap_or_default();
    let family = query.family.as_deref().and_then(|name| {
        MODEL_PROFILES.iter().find(|profile| {
            profile.label.eq_ignore_ascii_case(name)
                || profile.weights.eq_ignore_ascii_case(name)
                || profile.model_type.eq_ignore_ascii_case(name)
        })
    });
    let limit = query.limit.unwrap_or(24);
    let query_text = text.clone();
    let searched = tokio::task::spawn_blocking(move || {
        hf::search(cache.as_deref(), &query_text, family, limit, &cluster)
    })
    .await
    .unwrap_or_else(|error| Err(format!("the Hub search failed: {error}")));
    let (models, error) = match searched {
        Ok(models) => (models, None),
        Err(error) => (Vec::new(), Some(error)),
    };
    Json(ModelSearchView {
        query: text,
        family: family.map(|profile| profile.label.to_owned()),
        cluster,
        models,
        error,
    })
}

/// Inspect one model id for the browser: a cached entry, or its Hub config.json.
async fn models_inspect(
    State(web): State<WebState>,
    Query(query): Query<ModelInspectQuery>,
) -> Json<ModelInspectView> {
    let config = web.config();
    let cache = config.hub_cache_dir();
    let cluster = web.cluster(&config);
    let id = query.id;
    let inspected =
        tokio::task::spawn_blocking(move || hf::inspect(cache.as_deref(), &id, &cluster))
            .await
            .unwrap_or_else(|error| Err(format!("the model inspection failed: {error}")));
    match inspected {
        Ok(model) => Json(ModelInspectView {
            cluster,
            model: Some(model),
            error: None,
        }),
        Err(error) => Json(ModelInspectView {
            cluster,
            model: None,
            error: Some(error),
        }),
    }
}

/// The shelf the Overview page draws: every cached model this build can run, with the ranks
/// that already hold everything a launch needs.
async fn model_shelf(
    web: &WebState,
    config: &SparkConfig,
    models: &[CachedModel],
) -> Vec<ShelfModel> {
    let jobs = web.jobs.list();
    // Building each target reads the cached config.json and resolves refs/main, so it belongs off
    // the async worker.
    let candidates: Vec<(CachedModel, Result<SparkConfig, String>)> = {
        let config = config.clone();
        let models = models.to_vec();
        tokio::task::spawn_blocking(move || {
            models
                .into_iter()
                .filter(|model| model.supported)
                .map(|model| {
                    let target = config
                        .for_cached_model(&model)
                        .map_err(|error| error.message);
                    (model, target)
                })
                .collect()
        })
        .await
        .unwrap_or_default()
    };
    let configs: Vec<SparkConfig> = candidates
        .iter()
        .filter_map(|(_, target)| target.as_ref().ok().cloned())
        .collect();
    // Before the first save the node list is still a placeholder, so probing it would open SSH
    // to names that are not this cluster yet. The shelf can still show the cache.
    let can_probe = web.config_exists();
    let matrix = if !can_probe || configs.is_empty() {
        Vec::new()
    } else {
        let probe = Arc::clone(&web.ssh);
        let timeout =
            std::time::Duration::from_secs(config.web.ssh_timeout_secs.saturating_add(10));
        tokio::task::spawn_blocking(move || prepared_matrix(&configs, probe.as_ref(), timeout))
            .await
            .unwrap_or_default()
    };

    let mut next = 0usize;
    candidates
        .into_iter()
        .map(|(model, target)| {
            let ranks = if target.is_ok() {
                let ranks = matrix.get(next).cloned().unwrap_or_default();
                next += 1;
                ranks
            } else {
                Vec::new()
            };
            let active = model.id == config.model.id;
            let preparing = target.as_ref().is_ok_and(|target| {
                let scope = target.converted_path().display().to_string();
                jobs.iter().any(|job| {
                    job.status == JobStatus::Running
                        && (job.kind.starts_with("prepare-") || job.kind.starts_with("convert-"))
                        && job.command.contains(&scope)
                })
            });
            let prepared = !ranks.is_empty() && ranks.iter().all(|ready| *ready);
            let note = match &target {
                Err(error) => Some(error.clone()),
                Ok(_) if !can_probe => {
                    Some("save the configuration to check every Spark".to_owned())
                }
                Ok(_) if !prepared => {
                    let missing: Vec<String> = ranks
                        .iter()
                        .enumerate()
                        .filter(|(_, ready)| !**ready)
                        .map(|(rank, _)| format!("rank {rank}"))
                        .collect();
                    Some(if missing.is_empty() {
                        "not prepared yet".to_owned()
                    } else {
                        format!("not prepared on {}", missing.join(", "))
                    })
                }
                Ok(_) => None,
            };
            ShelfModel {
                id: model.id,
                weights: model.weights,
                engine: model.engine,
                bytes: model.bytes,
                complete: model.complete,
                requires_conversion: target.as_ref().map_or_else(
                    |_| model.requires_conversion.unwrap_or(true),
                    SparkConfig::requires_conversion,
                ),
                active,
                compatible: target.is_ok(),
                prepared,
                prepared_ranks: ranks
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, ready)| ready.then_some(rank))
                    .collect(),
                world_size: config.world_size(),
                preparing,
                note,
            }
        })
        .collect()
}

/// Probes every (model, rank) pair in parallel. The probes are read-only existence checks, one
/// SSH stream per rank, and never touch a checkpoint.
fn prepared_matrix(
    configs: &[SparkConfig],
    probe: &dyn SshProbe,
    timeout: std::time::Duration,
) -> Vec<Vec<bool>> {
    let mut ready: Vec<Vec<bool>> = configs
        .iter()
        .map(|config| vec![false; config.world_size()])
        .collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (index, config) in configs.iter().enumerate() {
            for rank in 0..config.world_size() {
                let command = prepared_probe_command(config, rank);
                handles.push((
                    index,
                    rank,
                    scope.spawn(move || probe.capture(&command, timeout)),
                ));
            }
        }
        for (index, rank, handle) in handles {
            if let Ok(result) = handle.join() {
                ready[index][rank] = result.is_ok_and(|text| text.trim() == "ready");
            }
        }
    });
    ready
}

/// The ranks of one model that already hold everything a launch needs.
async fn prepared_ranks(web: &WebState, config: &SparkConfig) -> Result<Vec<bool>, String> {
    let config = config.clone();
    let probe = Arc::clone(&web.ssh);
    let timeout = std::time::Duration::from_secs(config.web.ssh_timeout_secs.saturating_add(10));
    tokio::task::spawn_blocking(move || {
        prepared_matrix(std::slice::from_ref(&config), probe.as_ref(), timeout)
            .pop()
            .unwrap_or_default()
    })
    .await
    .map_err(|error| format!("the preparation check failed: {error}"))
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
    if !config.requires_conversion() {
        return api_error(
            StatusCode::CONFLICT,
            format!("{} has no converted layout to write", config.model.weights),
        );
    }
    if web.stats.latest().is_some_and(|sample| sample.online) {
        return api_error(
            StatusCode::CONFLICT,
            "rank 0 is online; stop the cluster before converting its weights",
        );
    }
    let world_size = config.world_size();
    let ranks = request.ranks.unwrap_or_else(|| (0..world_size).collect());
    if ranks.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "ranks must not be empty");
    }
    for rank in &ranks {
        if *rank >= world_size {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("rank {rank} is outside 0..{world_size}"),
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
        // Refused while this rank converts, the group is being launched or stopped, or the
        // checkpoint is being downloaded or copied.
        match web.jobs.start_exclusive(command, CONVERT_EXCLUSIVE).await {
            Ok(id) => started.push(id),
            Err(error) => {
                return api_error(StatusCode::CONFLICT, error.message);
            }
        }
    }
    accepted_jobs(&web, &started)
}

#[derive(Deserialize)]
struct ModelRequest {
    /// The Hugging Face model id to prepare or swap to.
    id: String,
}

/// The cached model `id` as a resolved configuration, using this build's profile for it.
fn cached_target(web: &WebState, id: &str) -> Result<(SparkConfig, CachedModel), String> {
    let active = web.usable_config()?;
    let cache = active.hub_cache_dir().ok_or_else(|| {
        "no Hugging Face cache directory: set HF_HUB_CACHE or model.hub_cache".to_owned()
    })?;
    let model = hf::lookup(&cache, id).ok_or_else(|| {
        format!(
            "{id} is not in the Hugging Face cache at {}",
            cache.display()
        )
    })?;
    if !model.complete {
        return Err(format!("{id} is still downloading"));
    }
    if !model.supported {
        return Err(format!("{id} is not a model type this build can run"));
    }
    let target = active
        .for_cached_model(&model)
        .map_err(|error| error.message)?;
    Ok((target, model))
}

/// The running conversion or preparation for `config`, if one is writing its files.
fn conversion_job(web: &WebState, config: &SparkConfig) -> Option<u64> {
    let scope = config.converted_path().display().to_string();
    web.jobs
        .list()
        .iter()
        .find(|job| {
            job.status == JobStatus::Running
                && (job.kind.starts_with("convert-") || job.kind.starts_with("prepare-"))
                && job.command.contains(&scope)
        })
        .map(|job| job.id)
}

/// Shards a different model on every rank while the current server keeps running.
async fn prepare_model(State(web): State<WebState>, Json(request): Json<ModelRequest>) -> Response {
    if !web.config_exists() {
        return api_error(
            StatusCode::CONFLICT,
            "save the configuration before preparing a model",
        );
    }
    let lookup = {
        let web = web.clone();
        let id = request.id.clone();
        tokio::task::spawn_blocking(move || cached_target(&web, &id)).await
    };
    let (target, model) = match lookup {
        Ok(Ok(target)) => target,
        Ok(Err(message)) => return api_error(StatusCode::CONFLICT, message),
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("the model lookup failed: {error}"),
            );
        }
    };
    if !target.requires_conversion() {
        return api_error(
            StatusCode::CONFLICT,
            format!("{} has no converted layout to prepare", model.id),
        );
    }
    // The selected model is the one a launch reads. Rewriting its rank files while it serves,
    // loads or is about to start is the one overlap a background prepare must never have; the
    // foreground Convert action owns that case and refuses while the cluster is online.
    if target.model.id == web.config().model.id {
        return api_error(
            StatusCode::CONFLICT,
            format!(
                "{} is the selected model; stop the cluster and use Convert, or swap to another model first",
                model.id
            ),
        );
    }
    if let Some(id) = web.jobs.running_in(&["swap"]) {
        return api_error(
            StatusCode::CONFLICT,
            format!("job {id} is swapping models; wait for it to finish"),
        );
    }
    if let Some(id) = conversion_job(&web, &target) {
        return api_error(
            StatusCode::CONFLICT,
            format!(
                "conversion job {id} is already working on {}",
                target.model.id
            ),
        );
    }
    let mut started = Vec::new();
    for rank in 0..target.world_size() {
        let command = prepare_command(&target, rank);
        match web
            .jobs
            .start_exclusive(
                command,
                &[
                    "download",
                    "sync-model",
                    "swap",
                    "network-setup-r0",
                    "network-setup-r1",
                    "network-setup-r2",
                    "network-setup-r3",
                ],
            )
            .await
        {
            Ok(id) => started.push(id),
            Err(error) => {
                let note = if started.is_empty() {
                    String::new()
                } else {
                    format!(" (ranks already started: {started:?})")
                };
                return api_error(StatusCode::CONFLICT, format!("{}{note}", error.message));
            }
        }
    }
    accepted_jobs(&web, &started)
}

/// Stops the serving model, selects `request.id`, and launches it again as one in-process job.
async fn swap(State(web): State<WebState>, Json(request): Json<ModelRequest>) -> Response {
    if !web.config_exists() {
        return api_error(
            StatusCode::CONFLICT,
            "save the configuration before swapping models",
        );
    }
    let lookup = {
        let web = web.clone();
        let id = request.id.clone();
        tokio::task::spawn_blocking(move || cached_target(&web, &id)).await
    };
    let (target, model) = match lookup {
        Ok(Ok(target)) => target,
        Ok(Err(message)) => return api_error(StatusCode::CONFLICT, message),
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("the model lookup failed: {error}"),
            );
        }
    };
    let current = match web
        .usable_config()
        .map(|config| config.resolved().unwrap_or(config))
    {
        Ok(config) => config,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    if target.model.id == current.model.id {
        return api_error(
            StatusCode::CONFLICT,
            format!("{} is already the selected model", model.id),
        );
    }
    if let Some(id) = web.jobs.running_in(&[
        "swap",
        "launch",
        "stop",
        "download",
        "sync-model",
        "sync-software",
        "convert-r0",
        "convert-r1",
        "convert-r2",
        "convert-r3",
        "network-setup-r0",
        "network-setup-r1",
        "network-setup-r2",
        "network-setup-r3",
    ]) {
        return api_error(
            StatusCode::CONFLICT,
            format!("job {id} is changing the cluster; wait for it to finish"),
        );
    }
    if let Some(id) = conversion_job(&web, &target) {
        return api_error(
            StatusCode::CONFLICT,
            format!("{} is still being prepared by job {id}", model.id),
        );
    }
    let ready = match prepared_ranks(&web, &target).await {
        Ok(ready) => ready,
        Err(message) => return api_error(StatusCode::CONFLICT, message),
    };
    let missing: Vec<String> = ready
        .iter()
        .enumerate()
        .filter(|(_, ready)| !**ready)
        .map(|(rank, _)| format!("rank {rank}"))
        .collect();
    if !missing.is_empty() {
        return api_error(
            StatusCode::CONFLICT,
            format!(
                "{} is not prepared on {}; prepare it before swapping",
                model.id,
                missing.join(", ")
            ),
        );
    }
    // A background prepare that started while the ranks were being probed must not overlap the
    // swap. The job reservation below then blocks any new one.
    if let Some(id) = conversion_job(&web, &target) {
        return api_error(
            StatusCode::CONFLICT,
            format!("{} is still being prepared by job {id}", model.id),
        );
    }
    let label = format!("Swap to {}", model.id);
    // The scope makes the reservation atomic against a prepare of *this* checkpoint, without
    // waiting for a background prepare of some other model.
    let scope = target.converted_path().display().to_string();
    let task = match web.jobs.start_task_scoped(
        "swap",
        &label,
        format!(
            "in-process swap {} -> {}",
            current.model.id, target.model.id
        ),
        &[
            "launch",
            "stop",
            "swap",
            "download",
            "sync-model",
            "sync-software",
            "convert-r0",
            "convert-r1",
            "convert-r2",
            "convert-r3",
            "network-setup-r0",
            "network-setup-r1",
            "network-setup-r2",
            "network-setup-r3",
        ],
        &scope,
    ) {
        Ok(task) => task,
        Err(error) => return api_error(StatusCode::CONFLICT, error.message),
    };
    let job_id = task.id();
    let web_for_task = web.clone();
    tokio::spawn(run_swap(web_for_task, task, current, target));
    accepted_jobs(&web, &[job_id])
}

async fn run_swap(
    web: WebState,
    task: crate::jobs::JobTask,
    current: SparkConfig,
    target: SparkConfig,
) {
    match swap_steps(&web, &task, &current, &target).await {
        Ok(()) => task.finish(true),
        Err(message) => {
            task.log(message);
            task.finish(false);
        }
    }
}

async fn swap_steps(
    web: &WebState,
    task: &crate::jobs::JobTask,
    current: &SparkConfig,
    target: &SparkConfig,
) -> Result<(), String> {
    task.log(format!(
        "Swapping {} -> {}; the target is prepared on every rank",
        current.model.id, target.model.id
    ));
    let previous = web.config();
    // The stop is sent even when telemetry says the server is offline: a failed telemetry fetch
    // looks exactly like a stopped server, and launching over a live one is reported as success by
    // both launchers. The stop command is idempotent, so a stopped cluster is left alone.
    task.log(format!(
        "Stopping {} and letting in-flight requests drain",
        current.model.id
    ));
    let stop = web
        .jobs
        .start_exclusive(serve_action_command(current, ServeAction::Stop), &[])
        .await
        .map_err(|error| format!("the stop was not accepted: {}", error.message))?;
    task.log(format!("stop job {stop} started"));
    wait_job(web, stop, STOP_TIMEOUT).await?;
    task.log("Stop finished; confirming through the launcher that no rank is still running");
    wait_stopped(current).await?;
    task.log(format!("Saving {} as the selected model", target.model.id));
    // The saved file keeps `checkpoint` empty, as the model selector does: the cache's
    // `refs/main` stays authoritative, and the next launch resolves it again.
    let mut saved = target.clone();
    saved.model.checkpoint.clear();
    saved.model.tokenizer.clear();
    if let Err((_, message)) = save_and_adopt(web, saved) {
        return Err(format!(
            "the stop succeeded but the new selection could not be saved: {message}; the cluster is stopped and the previous selection was kept"
        ));
    }
    task.log(format!("Launching {}", target.model.id));
    let launch = match web
        .jobs
        .start_exclusive(serve_action_command(target, ServeAction::Launch), &[])
        .await
    {
        Ok(id) => id,
        Err(error) => {
            let restored = restore_selection(web, previous);
            return Err(format!(
                "the launch was not accepted: {}; {restored} and the cluster is stopped",
                error.message
            ));
        }
    };
    task.log(format!(
        "launch job {launch} started; waiting for the launcher to confirm the ranks survived start-up"
    ));
    if let Err(message) = wait_job(web, launch, LAUNCH_TIMEOUT).await {
        let restored = restore_selection(web, previous);
        return Err(format!(
            "the launch failed: {message}; {restored} and the cluster is stopped"
        ));
    }
    task.log(
        "Launcher confirmed the ranks are up; they answer /readyz when the weights finish loading",
    );
    Ok(())
}

/// Puts the pre-swap selection back after a failed launch, so the saved model is the one that was
/// running rather than one that never started. Returns what happened, for the job log.
fn restore_selection(web: &WebState, previous: SparkConfig) -> String {
    match save_and_adopt(web, previous) {
        Ok(()) => "the previous selection was restored".to_owned(),
        Err((_, message)) => format!("restoring the previous selection failed: {message}"),
    }
}

async fn wait_job(web: &WebState, id: u64, timeout: std::time::Duration) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let Some(detail) = web.jobs.get(id) else {
            return Err(format!("job {id} disappeared before it finished"));
        };
        if detail.summary.status.is_done() {
            if detail.summary.status == JobStatus::Succeeded {
                return Ok(());
            }
            let reason = detail
                .summary
                .last_line
                .unwrap_or_else(|| "no output".to_owned());
            return Err(format!("job {id} failed: {reason}"));
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!(
                "job {id} did not finish within {}s",
                timeout.as_secs()
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

/// Asks the launcher whether any rank process is still alive.
///
/// The command is run directly rather than as a job: the drain is polled, and a job per poll would
/// evict the activity history and race the job-list cap.
async fn launcher_running(config: &SparkConfig) -> Result<bool, String> {
    let spec = serve_action_command(config, ServeAction::Status);
    let mut command = tokio::process::Command::new(&spec.program);
    command
        .args(&spec.args)
        .envs(&spec.env)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(cwd) = &spec.cwd {
        command.current_dir(cwd);
    }
    let output = command
        .output()
        .await
        .map_err(|error| format!("the launcher status could not run: {error}"))?;
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(format!("the launcher status failed: {}", text.trim()));
    }
    // A status line says either "running" or "not running"; only the former is a live process.
    Ok(text
        .lines()
        .any(|line| line.contains("running") && !line.contains("not running")))
}

/// Confirms through the launcher's `status` command that every rank has stopped.
///
/// Telemetry is deliberately not used here: a failed fetch is indistinguishable from an offline
/// server, and both launchers report a launch over a live process as success. `status` checks the
/// processes themselves, so it is evidence that the old model is really gone.
async fn wait_stopped(config: &SparkConfig) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + STOP_TIMEOUT;
    loop {
        if !launcher_running(config).await? {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(
                "a rank is still running after the stop; the model was not switched".to_owned(),
            );
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

async fn launch(State(web): State<WebState>) -> Response {
    if web.stats.latest().is_some_and(|sample| sample.online) {
        return api_error(
            StatusCode::CONFLICT,
            "rank 0 is already reachable; stop the cluster before launching it",
        );
    }
    if web
        .jobs
        .list()
        .iter()
        .find(|job| job.kind == "sync-software")
        .is_some_and(|job| job.status == JobStatus::Failed)
    {
        return api_error(
            StatusCode::CONFLICT,
            "the last Deploy failed; retry it before launching the cluster",
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
    // Launch and stop exclude each other (and a second of themselves), and a launch waits for
    // every copy and conversion; status runs alongside.
    let exclusive: &[&str] = match action {
        ServeAction::Launch => LAUNCH_EXCLUSIVE,
        ServeAction::Stop => CLUSTER_STATE_KINDS,
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

    /// A healthy four-Spark cluster. When `fresh_rank` is set, that rank's fast ports are
    /// reported as cabled but unconfigured (no addresses, MTU 1500), which is the state the
    /// one-click port fix exists for.
    struct FakeSsh {
        fresh_rank: Option<usize>,
    }
    impl FakeSsh {
        fn fresh_node_text(rank: usize) -> String {
            format!(
                "host spark-{rank}\nport rocep1s0f1 ACTIVE 200\ngid rocep1s0f1 0 IB/RoCE_v1 fe80:0000:0000:0000:4ebb:47ff:fe00:0002 enp1s0f1np1\ngid rocep1s0f1 1 RoCE_v2 fe80:0000:0000:0000:4ebb:47ff:fe00:0002 enp1s0f1np1\nport roceP2p1s0f1 ACTIVE 200\ngid roceP2p1s0f1 0 IB/RoCE_v1 fe80:0000:0000:0000:4ebb:47ff:fe00:0006 enP2p1s0f1np1\narp all 0 0\nlink enp1s0f1np1 1500 up 0 0\nlink enP2p1s0f1np1 1500 up 0 0\n"
            )
        }
    }
    impl SshProbe for FakeSsh {
        fn probe(&self, _command: &CommandSpec) -> ProbeOutcome {
            ProbeOutcome {
                ok: true,
                message: String::new(),
            }
        }

        /// A healthy dual-rail node per rank: the lead runs locally, the rest over SSH to
        /// `spark1`..`spark3`, and `spark3` refuses.
        fn capture(
            &self,
            command: &CommandSpec,
            _timeout: std::time::Duration,
        ) -> Result<String, String> {
            if command.kind == "prepared" {
                return Ok("ready".to_owned());
            }
            // The local rank's commands carry no ssh destination; a link probe runs under
            // `sudo -n`, so the program is not the way to tell it either.
            let rank = (1..4)
                .find(|rank| {
                    command
                        .args
                        .iter()
                        .any(|arg| arg.ends_with(&format!("spark{rank}")))
                })
                .unwrap_or(0);
            if command.kind == "fabric-probe" {
                if self.fresh_rank == Some(rank) {
                    // Ports cabled but unconfigured: nothing answers.
                    return Ok((0..4)
                        .filter(|peer| *peer != rank)
                        .map(|peer| format!("ping {peer} dev{rank} no\n"))
                        .collect());
                }
                // A fully switched fabric: this rank's port reaches every other rank.
                if rank == 3 {
                    return Err("Permission denied (publickey).".to_owned());
                }
                let mut reply = String::new();
                for peer in 0..4 {
                    if peer != rank {
                        reply.push_str(&format!(
                            "link {peer} dev{rank} 198.51.100.{} ok\nping {peer} dev{rank} ok\n",
                            21 + peer
                        ));
                    }
                }
                return Ok(reply);
            }
            if rank == 3 && self.fresh_rank.is_none() {
                return Err("Permission denied (publickey).".to_owned());
            }
            if self.fresh_rank == Some(rank) {
                return Ok(Self::fresh_node_text(rank));
            }
            let rank = u8::try_from(rank).expect("rank");
            Ok(crate::inspect::tests::node_text(
                &format!("spark-{rank}"),
                11 + rank,
                21 + rank,
            ))
        }
    }

    /// A four-Spark cluster where every node answers and every fast port is already configured,
    /// so the node check finds nothing for the port fix to write.
    struct HealthySsh;
    impl SshProbe for HealthySsh {
        fn probe(&self, _command: &CommandSpec) -> ProbeOutcome {
            ProbeOutcome {
                ok: true,
                message: String::new(),
            }
        }

        fn capture(
            &self,
            command: &CommandSpec,
            _timeout: std::time::Duration,
        ) -> Result<String, String> {
            if command.kind == "prepared" {
                return Ok("ready".to_owned());
            }
            // The local rank's commands carry no ssh destination; a link probe runs under
            // `sudo -n`, so the program is not the way to tell it either.
            let rank = (1..4)
                .find(|rank| {
                    command
                        .args
                        .iter()
                        .any(|arg| arg.ends_with(&format!("spark{rank}")))
                })
                .unwrap_or(0);
            if command.kind == "fabric-probe" {
                let mut reply = String::new();
                for peer in 0..4 {
                    if peer != rank {
                        reply.push_str(&format!(
                            "link {peer} dev{rank} 198.51.100.{} ok\nping {peer} dev{rank} ok\n",
                            21 + peer
                        ));
                    }
                }
                return Ok(reply);
            }
            let rank = u8::try_from(rank).expect("rank");
            Ok(crate::inspect::tests::node_text(
                &format!("spark-{rank}"),
                11 + rank,
                21 + rank,
            ))
        }
    }

    fn test_state(dir: &std::path::Path) -> WebState {
        test_state_with(dir, false, None)
    }

    fn test_state_with_ssh(
        dir: &std::path::Path,
        exists: bool,
        load_error: Option<String>,
        ssh: Arc<dyn SshProbe>,
    ) -> WebState {
        test_state_inner(dir, exists, load_error, ssh)
    }

    fn test_state_with(
        dir: &std::path::Path,
        exists: bool,
        load_error: Option<String>,
    ) -> WebState {
        test_state_inner(
            dir,
            exists,
            load_error,
            Arc::new(FakeSsh { fresh_rank: None }),
        )
    }

    fn test_state_inner(
        dir: &std::path::Path,
        exists: bool,
        load_error: Option<String>,
        ssh: Arc<dyn SshProbe>,
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
            ssh,
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
        assert!(html.contains("model-shelf"));
        assert!(html.contains("id=\"model-panel\""));
        assert!(html.contains("id=\"model-search\""));
        assert!(html.contains("id=\"model-families\""));
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
        assert_eq!(
            body["supported"]["models"][0]["id"],
            crate::config::DEFAULT_MODEL_ID
        );
        assert_eq!(
            body["supported"]["models"][0]["weights"],
            crate::config::DEFAULT_WEIGHTS_ID
        );
        // The registry order is not part of the contract; find the family by id.
        let models = body["supported"]["models"]
            .as_array()
            .expect("the supported model list");
        let glm = models
            .iter()
            .find(|model| model["id"] == crate::config::GLM_MODEL_ID)
            .expect("the GLM model is listed");
        assert_eq!(glm["engine"], crate::config::GLM_ENGINE_ID);
        assert_eq!(body["config_exists"], false);
    }

    #[tokio::test]
    async fn a_solo_qwen_config_is_saved_and_reported_as_one_world() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        let mut config = SparkConfig::default();
        config.model.weights = crate::config::QWEN_WEIGHTS_ID.to_owned();
        config.model.id = crate::config::QWEN_MODEL_ID.to_owned();
        config.model.checkpoint = "/data/Qwen3.8-Flash-Next-NVFP4".to_owned();
        config.model.cubins.clear();
        config.engine.kind = crate::config::QWEN_ENGINE_ID.to_owned();
        config.runtime.max_context_tokens = crate::config::QWEN_CONTEXT_TOKENS;
        config.cluster.nodes = vec![crate::config::NodeConfig::new("this-spark.local", "", true)];
        let response = app
            .clone()
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
        let saved = SparkConfig::load(&dir.path().join("sparkflash.toml")).expect("saved");
        assert!(saved.is_solo());
        assert_eq!(saved.cluster.nodes[0].peer, "");

        let state = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/state")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response"),
        )
        .await;
        assert_eq!(state["limits"]["world_size"], 1);
        assert_eq!(
            state["limits"]["max_context_tokens"],
            crate::config::QWEN_CONTEXT_TOKENS
        );
        assert_eq!(
            state["supported"]["solo"]["weights"],
            crate::config::QWEN_WEIGHTS_ID
        );

        // The solo pack converts too: one rank file, which its launch reads.
        let convert = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .header(ACTION_HEADER, "1")
                    .uri("/api/actions/convert")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(convert.status(), StatusCode::ACCEPTED);
        let body = body_json(convert).await;
        assert_eq!(
            body["jobs"][0]["kind"].as_str().unwrap_or_default(),
            "convert-r0"
        );
    }

    #[tokio::test]
    async fn automatic_provisioning_waits_for_a_saved_configuration() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state(dir.path());
        assert!(!state.config_exists());
        state.provision_once().await.expect("a no-op");
        assert!(state.jobs().list().is_empty());
    }

    #[tokio::test]
    async fn automatic_provisioning_reuses_the_boot_node_check() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state_with(dir.path(), true, None);
        {
            let mut config = state.config();
            config.model.weights = crate::config::GLM_WEIGHTS_ID.to_owned();
            config.engine.kind = crate::config::GLM_ENGINE_ID.to_owned();
            config.paths.binary = "/bin/true".to_owned();
            config.paths.nccl_lib.clear();
            state.config.write().expect("config lock").config = config;
        }
        state.provision_once().await.expect("the first check");
        let checked = state
            .last_inspect
            .read()
            .expect("inspect lock")
            .as_ref()
            .expect("report")
            .checked_ms;
        // Age the report: the next pass must still reuse it, because no software job changed
        // anything. The hash pass is a startup cost, not a poll.
        state
            .last_inspect
            .write()
            .expect("inspect lock")
            .as_mut()
            .expect("report")
            .checked_ms = 1;
        state.provision_once().await.expect("the cached check");
        assert_eq!(
            state
                .last_inspect
                .read()
                .expect("inspect lock")
                .as_ref()
                .expect("report")
                .checked_ms,
            1,
            "the second pass re-ran the node check"
        );
        assert!(checked > 1);
    }

    #[test]
    fn a_finished_software_job_invalidates_the_cached_node_check() {
        let job = JobSummary {
            id: 1,
            kind: "sync-software".to_owned(),
            label: String::new(),
            command: String::new(),
            status: JobStatus::Succeeded,
            started_ms: 10,
            finished_ms: Some(20),
            exit_code: Some(0),
            last_line: None,
            progress: None,
        };
        assert!(software_changed_since(std::slice::from_ref(&job), 10));
        assert!(!software_changed_since(&[job], 20));
        let mut other = JobSummary {
            id: 2,
            kind: "download".to_owned(),
            label: String::new(),
            command: String::new(),
            status: JobStatus::Succeeded,
            started_ms: 10,
            finished_ms: Some(20),
            exit_code: Some(0),
            last_line: None,
            progress: None,
        };
        other.finished_ms = Some(30);
        assert!(!software_changed_since(&[other], 10));
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
        assert!(nodes.iter().any(|node| node["ssh"] == "spark7.local"));
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
    async fn launch_refuses_after_a_failed_deploy() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state(dir.path());
        let command = CommandSpec {
            program: "sh".to_owned(),
            args: vec!["-c".to_owned(), "exit 1".to_owned()],
            env: std::collections::BTreeMap::new(),
            cwd: None,
            label: "Deploy the server".to_owned(),
            kind: "sync-software".to_owned(),
        };
        let id = state.jobs.start(command).await.expect("job accepted");
        for _ in 0..100 {
            if state
                .jobs
                .get(id)
                .is_some_and(|job| job.summary.status.is_done())
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
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
                .contains("last Deploy failed"),
            "{body}"
        );
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
        let families = models["families"].as_array().expect("families");
        assert_eq!(families.len(), crate::config::MODEL_PROFILES.len());
        assert!(families.iter().any(|family| family["label"] == "DeepSeek"));
        assert!(families.iter().any(|family| family["label"] == "Qwen"));
        let state = body_json(app.clone().oneshot(get("/api/state")).await.expect("state")).await;
        assert_eq!(state["checkpoint"]["from_cache"], true);
        assert_eq!(
            state["checkpoint"]["path"],
            snapshot.display().to_string().as_str()
        );
        assert_eq!(state["checkpoint"]["exists"], true);
        assert_eq!(state["checkpoint"]["cached"]["complete"], true);
        assert_eq!(state["checkpoint"]["cached"]["has_tokenizer"], true);

        // A complete cache snapshot needs no download job.
        let response = app
            .oneshot(post_json("/api/actions/download", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let error = body_json(response).await["error"].to_string();
        assert!(
            error.contains("already in the Hugging Face cache"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn the_model_browser_searches_the_cache_and_inspects_before_selecting() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        let app = test_router(test_state(dir.path()));
        let get = |uri: &str| {
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("request")
        };
        // An empty query lists the cache and does not depend on the network.
        let search = body_json(
            app.clone()
                .oneshot(get("/api/models/search?q="))
                .await
                .expect("search"),
        )
        .await;
        assert_eq!(search["error"], serde_json::Value::Null);
        assert_eq!(search["models"][0]["id"], crate::config::DEFAULT_MODEL_ID);
        assert_eq!(search["models"][0]["cached"], true);
        assert_eq!(search["models"][0]["family"], "DeepSeek");

        // Selecting a cached model resolves its profile without a network read.
        let inspected = body_json(
            app.oneshot(get(&format!(
                "/api/models/inspect?id={}",
                crate::config::DEFAULT_MODEL_ID
            )))
            .await
            .expect("inspect"),
        )
        .await;
        assert_eq!(inspected["error"], serde_json::Value::Null);
        assert_eq!(
            inspected["model"]["weights"],
            crate::config::DEFAULT_WEIGHTS_ID
        );
        assert_eq!(inspected["model"]["family"], "DeepSeek");
        assert_eq!(inspected["model"]["cached"], true);
    }

    #[tokio::test]
    async fn the_browser_family_side_filters_the_cache_without_the_network() {
        let dir = tempfile::tempdir().expect("tempdir");
        for (id, revision, kind) in [
            (crate::config::DEFAULT_MODEL_ID, "a", "deepseek_v41"),
            (crate::config::QWEN_MODEL_ID, "b", "qwen4_exp"),
            (crate::config::GLM_MODEL_ID, "c", "glm5_next"),
        ] {
            crate::hf::tests::cache_model(&dir.path().join("hub"), id, revision, kind);
        }
        let app = test_router(test_state(dir.path()));
        let get = |uri: &str| {
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("request")
        };
        let qwen = body_json(
            app.clone()
                .oneshot(get(&format!(
                    "/api/models/search?q=&family={}",
                    crate::config::QWEN_WEIGHTS_ID
                )))
                .await
                .expect("search"),
        )
        .await;
        assert_eq!(qwen["error"], serde_json::Value::Null);
        assert_eq!(qwen["family"], "Qwen");
        let ids: Vec<&str> = qwen["models"]
            .as_array()
            .expect("models")
            .iter()
            .filter_map(|model| model["id"].as_str())
            .collect();
        assert!(ids.contains(&crate::config::QWEN_MODEL_ID), "{ids:?}");
        assert!(
            ids.iter()
                .all(|id| id.to_ascii_lowercase().contains("qwen")),
            "{ids:?}"
        );
    }

    #[tokio::test]
    async fn the_model_shelf_names_the_active_model_and_its_prepared_ranks() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::QWEN_MODEL_ID,
            "abc123",
            "qwen4_exp",
        );
        let app = test_router(test_state_with(dir.path(), true, None));
        let models = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/models")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("models"),
        )
        .await;
        let shelf = models["shelf"].as_array().expect("shelf");
        let active = shelf
            .iter()
            .find(|model| model["id"] == crate::config::DEFAULT_MODEL_ID)
            .expect("the active model");
        assert_eq!(active["active"], true);
        assert_eq!(active["prepared"], true);
        assert_eq!(active["world_size"], 4);
        assert_eq!(active["prepared_ranks"], serde_json::json!([0, 1, 2, 3]));
        let qwen = shelf
            .iter()
            .find(|model| model["id"] == crate::config::QWEN_MODEL_ID)
            .expect("the cached Qwen model");
        assert_eq!(qwen["compatible"], false);
        assert!(
            qwen["note"]
                .as_str()
                .unwrap_or_default()
                .contains("cluster.nodes"),
            "{qwen}"
        );

        // A TP=1 model cannot be swapped into a four-Spark plan; the refusal names the topology
        // before anything is stopped.
        let response = app
            .oneshot(post_json(
                "/api/actions/swap",
                &serde_json::json!({"id": crate::config::QWEN_MODEL_ID}).to_string(),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = body_json(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("cluster.nodes"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_background_prepare_starts_one_job_per_rank_without_touching_the_active_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::GLM_MODEL_ID,
            "glm123",
            "glm5_next",
        );
        let state = test_state_with(dir.path(), true, None);
        {
            let mut config = state.config.write().expect("config lock");
            config.config.paths.binary = "/bin/true".to_owned();
            for node in &mut config.config.cluster.nodes {
                node.local = true;
            }
        }
        let app = test_router(state);
        let response = app
            .oneshot(post_json(
                "/api/actions/prepare-model",
                &serde_json::json!({"id": crate::config::GLM_MODEL_ID}).to_string(),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = body_json(response).await;
        let jobs = body["jobs"].as_array().expect("jobs");
        assert_eq!(jobs.len(), 4);
        let mut kinds: Vec<&str> = jobs
            .iter()
            .map(|job| job["kind"].as_str().expect("kind"))
            .collect();
        kinds.sort_unstable();
        assert_eq!(
            kinds,
            ["prepare-r0", "prepare-r1", "prepare-r2", "prepare-r3"]
        );
        for job in jobs {
            assert!(
                job["label"]
                    .as_str()
                    .unwrap_or_default()
                    .contains(crate::config::GLM_MODEL_ID),
                "{job}"
            );
        }
    }

    #[tokio::test]
    async fn a_swap_saves_the_prepared_model_and_launches_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::GLM_MODEL_ID,
            "glm123",
            "glm5_next",
        );
        let launch_script = dir.path().join("serve.sh");
        std::fs::write(&launch_script, "#!/bin/sh\nexit 0\n").expect("launch script");
        let state = test_state_with(dir.path(), true, None);
        {
            let mut config = state.config.write().expect("config lock");
            config.config.paths.binary = "/bin/true".to_owned();
            config.config.paths.serve_script = launch_script.display().to_string();
            for node in &mut config.config.cluster.nodes {
                node.local = true;
            }
        }
        let jobs = Arc::clone(&state.jobs);
        let app = test_router(state);
        let response = app
            .clone()
            .oneshot(post_json(
                "/api/actions/swap",
                &serde_json::json!({"id": crate::config::GLM_MODEL_ID}).to_string(),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = body_json(response).await;
        let id = body["jobs"][0]["id"].as_u64().expect("swap job id");
        let detail = loop {
            let detail = jobs.get(id).expect("swap job");
            if detail.summary.status.is_done() {
                break detail;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        assert_eq!(detail.summary.status, JobStatus::Succeeded, "{detail:?}");

        let config = body_json(
            app.oneshot(
                Request::builder()
                    .uri("/api/config")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("config"),
        )
        .await;
        assert_eq!(config["config"]["model"]["id"], crate::config::GLM_MODEL_ID);
        assert_eq!(config["config"]["model"]["checkpoint"], "");
    }

    #[tokio::test]
    async fn a_cached_model_this_build_cannot_run_is_refused_even_if_posted_directly() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(&dir.path().join("hub"), "org/llama", "abc", "llama");
        let state = test_state(dir.path());
        let mut config: serde_json::Value =
            serde_json::to_value(state.config()).expect("config JSON");
        let app = test_router(state);
        config["model"]["id"] = serde_json::json!("org/llama");
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
        let body = body_json(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("model_type"),
            "{body}"
        );
        assert!(!dir.path().join("sparkflash.toml").exists());
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

    #[tokio::test]
    async fn network_setup_requires_a_check_and_only_touches_nodes_that_need_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state_with_ssh(
            dir.path(),
            false,
            None,
            Arc::new(FakeSsh {
                fresh_rank: Some(3),
            }),
        );
        let app = test_router(state);

        // Nothing checked yet: the endpoint refuses rather than guess.
        let response = app
            .clone()
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);

        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);

        // Rank 3 is the only node whose ports are unconfigured; exactly one job starts.
        let response = app
            .clone()
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = body_json(response).await;
        let jobs = body["jobs"].as_array().expect("jobs array");
        assert_eq!(jobs.len(), 1, "{body}");
        assert_eq!(jobs[0]["kind"], "network-setup-r3");

        // The job that started carries the port fix for that rank: the SSH destination and the
        // `sudo -n` wrapper, not some other node's netplan.
        let listed = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/jobs")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response"),
        )
        .await;
        let job = listed
            .as_array()
            .and_then(|jobs| jobs.iter().find(|job| job["kind"] == "network-setup-r3"))
            .expect("the port job is listed");
        let command = job["command"].as_str().expect("command");
        assert!(command.starts_with("ssh "), "{command}");
        assert!(command.contains("spark3"), "{command}");
        assert!(command.contains("sudo -n bash -c"), "{command}");
        assert!(command.contains("netplan apply"), "{command}");
    }

    #[tokio::test]
    async fn network_setup_refuses_a_report_taken_of_another_configuration() {
        // The fixes replay the netplan body the last check computed, including its addresses.
        // Saving a different configuration between the check and the button must not write the
        // old machine's fast-network address onto whatever now runs at that rank.
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state_with_ssh(
            dir.path(),
            false,
            None,
            Arc::new(FakeSsh {
                fresh_rank: Some(3),
            }),
        );
        let app = test_router(state);
        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);

        let state = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/state")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response"),
        )
        .await;
        let mut config: SparkConfig =
            serde_json::from_value(state["config"].clone()).expect("config");
        config.web.ssh_timeout_secs += 1;
        let response = app
            .clone()
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

        let response = app
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body"),
        )
        .expect("json");
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|text| text.contains("different configuration")),
            "{body}"
        );
    }

    #[tokio::test]
    async fn network_setup_conflicts_when_the_ports_are_already_configured() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state_with_ssh(
            dir.path(),
            false,
            None,
            Arc::new(HealthySsh),
        ));
        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        // The API-level check really runs the classifier: the fake's four ports each answer every
        // peer, which is a switched fabric.
        let report = body_json(response).await;
        assert_eq!(report["fabric"]["shape"], "switched", "{report}");
        assert_eq!(report["fabric"]["links_measured"], true, "{report}");
        let response = app
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let error = body_json(response).await["error"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(error.contains("already match the plan"), "{error}");
    }

    #[tokio::test]
    async fn network_setup_does_not_certify_a_node_the_check_could_not_compare() {
        // Rank 3 does not answer the node check, so its ports were never compared with the plan.
        // Nothing is left for the button to apply, and the answer must name the node instead of
        // claiming its ports already match.
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let response = app
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let error = body_json(response).await["error"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(error.contains("still reports a problem on"), "{error}");
        assert!(!error.contains("already match"), "{error}");
    }

    #[tokio::test]
    async fn network_setup_refuses_a_report_older_than_ten_minutes() {
        // The fixes replay the addresses and routes measured at check time under root. The
        // configuration check cannot see a host that changed since, so the age is bounded too.
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state_with_ssh(
            dir.path(),
            false,
            None,
            Arc::new(FakeSsh {
                fresh_rank: Some(3),
            }),
        );
        let app = test_router(state.clone());
        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        {
            let mut lock = state.last_inspect.write().expect("inspect lock");
            let report = lock.as_mut().expect("a report");
            report.checked_ms = 1;
        }
        let response = app
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let error = body_json(response).await["error"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(error.contains("more than ten minutes old"), "{error}");
    }

    #[tokio::test]
    async fn network_setup_does_not_certify_a_cluster_wide_error() {
        // Two Sparks on one address is a finding with no rank, and no port fix can write it. The
        // "already match the plan" answer would be wrong while it stands.
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state_with_ssh(dir.path(), false, None, Arc::new(HealthySsh));
        let app = test_router(state.clone());
        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        {
            let mut lock = state.last_inspect.write().expect("inspect lock");
            let report = lock.as_mut().expect("a report");
            report.findings.push(inspect::Finding {
                rank: None,
                severity: Severity::Error,
                message: "Two Sparks have the same fast-network address".to_owned(),
            });
        }
        let response = app
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let error = body_json(response).await["error"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(error.contains("Two Sparks have the same"), "{error}");
        assert!(!error.contains("already match"), "{error}");
    }

    #[tokio::test]
    async fn the_node_check_is_refused_while_a_port_pass_runs() {
        // A check in the middle of a port pass records a half-applied cluster, and its fixes are
        // what the port setup later replays.
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state(dir.path());
        let app = test_router(state.clone());
        let spec = CommandSpec {
            program: "bash".to_owned(),
            args: vec!["-c".to_owned(), "sleep 1".to_owned()],
            env: BTreeMap::new(),
            cwd: None,
            label: "Port setup on spark0".to_owned(),
            kind: "network-setup-r0".to_owned(),
        };
        let _ = state
            .jobs
            .start_exclusive(spec, &[])
            .await
            .expect("port job");
        let response = app
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let error = body_json(response).await["error"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(error.contains("port setup job"), "{error}");
    }

    #[tokio::test]
    async fn network_setup_is_allowed_again_after_the_cluster_is_stopped() {
        // The settle window is about a launch that is still coming up. A stop after it means
        // nothing is, and refusing the port fix for five minutes would be wrong.
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state_with_ssh(
            dir.path(),
            false,
            None,
            Arc::new(FakeSsh {
                fresh_rank: Some(3),
            }),
        );
        let app = test_router(state.clone());
        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        for kind in ["launch", "stop"] {
            let spec = CommandSpec {
                program: "bash".to_owned(),
                args: vec!["-c".to_owned(), "true".to_owned()],
                env: BTreeMap::new(),
                cwd: None,
                label: kind.to_owned(),
                kind: kind.to_owned(),
            };
            let id = state.jobs.start_exclusive(spec, &[]).await.expect("job");
            loop {
                let detail = state.jobs.get(id).expect("job");
                if detail.summary.status.is_done() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
        let response = app
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }

    #[test]
    fn a_launch_that_was_not_stopped_keeps_the_ports_alone() {
        let job = |kind: &str, status: JobStatus, finished_ms: u64| JobSummary {
            id: finished_ms,
            kind: kind.to_owned(),
            label: String::new(),
            command: String::new(),
            status,
            started_ms: 0,
            finished_ms: Some(finished_ms),
            exit_code: None,
            last_line: None,
            progress: None,
        };
        assert!(!cluster_believed_running(&[]));
        assert!(cluster_believed_running(&[job(
            "launch",
            JobStatus::Succeeded,
            10
        )]));
        assert!(cluster_believed_running(&[job(
            "swap",
            JobStatus::Succeeded,
            10
        )]));
        assert!(!cluster_believed_running(&[
            job("launch", JobStatus::Succeeded, 10),
            job("stop", JobStatus::Succeeded, 20),
        ]));
        // A stop that ran before the launch does not count, and a stop attempt after it does.
        assert!(cluster_believed_running(&[
            job("launch", JobStatus::Succeeded, 10),
            job("stop", JobStatus::Succeeded, 5),
        ]));
        assert!(!cluster_believed_running(&[
            job("launch", JobStatus::Succeeded, 10),
            job("stop", JobStatus::Failed, 20),
        ]));
        // A launch that failed started nothing to protect.
        assert!(!cluster_believed_running(&[job(
            "launch",
            JobStatus::Failed,
            10
        )]));
    }

    #[tokio::test]
    async fn network_setup_waits_for_a_launch_that_is_still_coming_up() {
        // The launcher job ends when the script returns; rank 0 binds later and the telemetry
        // samples later still, so a port setup in that window would drop the links the group is
        // booting over.
        let dir = tempfile::tempdir().expect("tempdir");
        let state = test_state_with_ssh(
            dir.path(),
            false,
            None,
            Arc::new(FakeSsh {
                fresh_rank: Some(3),
            }),
        );
        let app = test_router(state.clone());
        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let spec = CommandSpec {
            program: "bash".to_owned(),
            args: vec!["-c".to_owned(), "true".to_owned()],
            env: BTreeMap::new(),
            cwd: None,
            label: "Launch".to_owned(),
            kind: "launch".to_owned(),
        };
        let id = state
            .jobs
            .start_exclusive(spec, &[])
            .await
            .expect("launch job");
        loop {
            let detail = state.jobs.get(id).expect("job");
            if detail.summary.status.is_done() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let response = app
            .oneshot(post_json("/api/network/setup", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let error = body_json(response).await["error"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(error.contains("may still be coming up"), "{error}");
    }

    #[test]
    fn a_port_setup_and_the_cluster_state_actions_exclude_each_other() {
        // NETWORK_SETUP_EXCLUSIVE names what a port job must wait for; CLUSTER_STATE_KINDS is
        // what a stop waits for. The two directions have to cover each other or one of them can
        // start while the other is running.
        for kind in [
            "network-setup-r0",
            "network-setup-r1",
            "network-setup-r2",
            "network-setup-r3",
        ] {
            assert!(
                CLUSTER_STATE_KINDS.contains(&kind),
                "a stop must wait for {kind}"
            );
        }
        // The port jobs stay out of their own list on purpose: one pass runs four of them at
        // once, so only the cluster-state actions have to be in both directions.
        for kind in ["launch", "stop", "swap"] {
            assert!(
                NETWORK_SETUP_EXCLUSIVE.contains(&kind),
                "a port job must wait for {kind}"
            );
        }
        assert!(
            !NETWORK_SETUP_EXCLUSIVE
                .iter()
                .any(|kind| kind.starts_with("network-setup-")),
            "one port-setup pass starts the four ranks together"
        );
        for kind in ["download", "sync-model", "sync-software"] {
            assert!(
                NETWORK_SETUP_EXCLUSIVE.contains(&kind),
                "a port job must wait for {kind}"
            );
        }
    }

    fn post_json(uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .header(ACTION_HEADER, "1")
            .header(header::CONTENT_TYPE, "application/json")
            .uri(uri)
            .body(Body::from(body.to_owned()))
            .expect("request")
    }

    #[tokio::test]
    async fn the_node_check_reports_each_spark_and_its_network_plan_can_be_applied() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        let response = app
            .clone()
            .oneshot(post_json("/api/network/apply", "{}"))
            .await
            .expect("response");
        assert_eq!(
            response.status(),
            StatusCode::CONFLICT,
            "nothing checked yet"
        );

        let response = app
            .clone()
            .oneshot(post_json("/api/inspect", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let report = body_json(response).await;
        assert_eq!(report["nodes"].as_array().map(Vec::len), Some(4));
        assert_eq!(report["nodes"][0]["hostname"], "spark-0");
        assert_eq!(
            report["nodes"][0]["rails"].as_array().map(Vec::len),
            Some(2)
        );
        assert_eq!(report["nodes"][3]["reachable"], false);
        assert!(
            report["plan"].is_null(),
            "an unreachable node blocks the plan"
        );
        assert!(
            report["findings"][0]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("Can't reach spark3"),
            "{report}"
        );

        let state = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/state")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let state = body_json(state).await;
        assert_eq!(state["setup"]["nodes"][1]["hostname"], "spark-1");
    }

    #[tokio::test]
    async fn the_detected_network_settings_are_saved_into_the_configuration() {
        let dir = tempfile::tempdir().expect("tempdir");
        let web = test_state(dir.path());
        let config = web.config();
        let probes: Vec<ProbeResult> = (0..4u8)
            .map(|rank| {
                Ok(inspect::parse_facts(&crate::inspect::tests::node_text(
                    &format!("spark-{rank}"),
                    11 + rank,
                    21 + rank,
                )))
            })
            .collect();
        *web.last_inspect.write().expect("inspect lock") =
            Some(inspect::analyze(&config, &probes, true, 1));
        // A real check records the configuration it measured; the apply refuses a report of
        // another one.
        *web.last_inspect_config.write().expect("inspect lock") = Some(config.clone());
        let app = test_router(web.clone());
        let response = app
            .oneshot(post_json("/api/network/apply", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let saved = SparkConfig::load(&dir.path().join("sparkflash.toml")).expect("saved");
        assert_eq!(saved.network.nccl_ib_hca, "rocep1s0f1,roceP2p1s0f1");
        assert_eq!(saved.network.nccl_socket_ifname, "enP2p1s0f1np1");
        assert_eq!(saved.network.nccl_ib_gid_index, Some(3));
        assert_eq!(saved.cluster.nodes[3].peer, "198.51.100.24");
        assert_eq!(saved.cluster.nodes[3].ssh, "spark3", "the SSH names stay");
    }

    #[tokio::test]
    async fn downloads_and_copies_are_refused_until_their_inputs_exist() {
        let dir = tempfile::tempdir().expect("tempdir");
        let web = test_state(dir.path());
        let app = test_router(web.clone());

        // The binary is not built in the test checkout, and nothing is downloaded.
        for (body, expected) in [
            (r#"{"what":"software"}"#, "is not built yet"),
            (r#"{"what":"model"}"#, "download it first"),
        ] {
            let response = app
                .clone()
                .oneshot(post_json("/api/actions/sync", body))
                .await
                .expect("response");
            assert_eq!(response.status(), StatusCode::CONFLICT, "{body}");
            let error = body_json(response).await["error"].to_string();
            assert!(error.contains(expected), "{body}: {error}");
        }
        let response = app
            .clone()
            .oneshot(post_json("/api/actions/sync", r#"{"what":"everything"}"#))
            .await
            .expect("response");
        assert!(response.status().is_client_error());

        let mut config = web.config();
        config.model.checkpoint = dir.path().display().to_string();
        config
            .save(&dir.path().join("sparkflash.toml"))
            .expect("save");
        let response = app
            .clone()
            .oneshot(post_json("/api/config/reload", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let response = app
            .oneshot(post_json("/api/actions/download", "{}"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let error = body_json(response).await["error"].to_string();
        assert!(
            error.contains("model.checkpoint names a directory"),
            "{error}"
        );
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
    /// A probe that answers every readiness check with a missing piece, for the refusal path.
    struct UnpreparedSsh;
    impl SshProbe for UnpreparedSsh {
        fn probe(&self, _command: &CommandSpec) -> ProbeOutcome {
            ProbeOutcome {
                ok: true,
                message: String::new(),
            }
        }
        fn capture(
            &self,
            command: &CommandSpec,
            _timeout: std::time::Duration,
        ) -> Result<String, String> {
            if command.kind == "prepared" {
                Ok("missing rank-file".to_owned())
            } else {
                Ok(String::new())
            }
        }
    }

    fn local_plan(state: &WebState, serve_script: Option<&std::path::Path>) {
        let mut config = state.config.write().expect("config lock");
        config.config.paths.binary = "/bin/true".to_owned();
        if let Some(script) = serve_script {
            config.config.paths.serve_script = script.display().to_string();
        }
        for node in &mut config.config.cluster.nodes {
            node.local = true;
        }
    }

    #[tokio::test]
    async fn a_model_is_not_prepared_before_the_configuration_is_saved() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = test_router(test_state(dir.path()));
        let response = app
            .oneshot(post_json(
                "/api/actions/prepare-model",
                &serde_json::json!({"id": crate::config::DEFAULT_MODEL_ID}).to_string(),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = body_json(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("save the configuration"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_background_prepare_never_rewrites_the_selected_models_shards() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        let state = test_state_with(dir.path(), true, None);
        local_plan(&state, None);
        let app = test_router(state);
        let response = app
            .oneshot(post_json(
                "/api/actions/prepare-model",
                &serde_json::json!({"id": crate::config::DEFAULT_MODEL_ID}).to_string(),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = body_json(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("selected model"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_swap_refuses_a_target_that_is_not_prepared_on_every_rank() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::GLM_MODEL_ID,
            "glm123",
            "glm5_next",
        );
        let mut state = test_state_with(dir.path(), true, None);
        state.ssh = Arc::new(UnpreparedSsh);
        local_plan(&state, None);
        let app = test_router(state);
        let response = app
            .oneshot(post_json(
                "/api/actions/swap",
                &serde_json::json!({"id": crate::config::GLM_MODEL_ID}).to_string(),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = body_json(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("not prepared on rank 0"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_failed_launch_fails_the_swap_and_restores_the_selection() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::GLM_MODEL_ID,
            "glm123",
            "glm5_next",
        );
        let launch_script = dir.path().join("serve.sh");
        std::fs::write(
            &launch_script,
            "#!/bin/sh\ncase \"$1\" in\n  status) echo 'serve-four-node: rank 0 not running on spark0'; exit 0;;\n  launch) echo 'launch exploded' >&2; exit 1;;\n  *) exit 0;;\nesac\n",
        )
        .expect("launch script");
        let state = test_state_with(dir.path(), true, None);
        local_plan(&state, Some(&launch_script));
        let jobs = Arc::clone(&state.jobs);
        let app = test_router(state);
        let response = app
            .clone()
            .oneshot(post_json(
                "/api/actions/swap",
                &serde_json::json!({"id": crate::config::GLM_MODEL_ID}).to_string(),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = body_json(response).await;
        let id = body["jobs"][0]["id"].as_u64().expect("swap job id");
        let detail = loop {
            let detail = jobs.get(id).expect("swap job");
            if detail.summary.status.is_done() {
                break detail;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        assert_eq!(detail.summary.status, JobStatus::Failed, "{detail:?}");
        assert!(
            detail.log.iter().any(|line| line.contains("launch failed")),
            "{detail:?}"
        );
        assert!(
            detail
                .log
                .iter()
                .any(|line| line.contains("previous selection was restored")),
            "{detail:?}"
        );
        let config = body_json(
            app.oneshot(
                Request::builder()
                    .uri("/api/config")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("config"),
        )
        .await;
        assert_eq!(
            config["config"]["model"]["id"],
            crate::config::DEFAULT_MODEL_ID
        );
    }

    #[tokio::test]
    async fn a_swap_is_refused_while_the_target_is_being_prepared() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::DEFAULT_MODEL_ID,
            "1bae0e1",
            "deepseek_v41",
        );
        crate::hf::tests::cache_model(
            &dir.path().join("hub"),
            crate::config::GLM_MODEL_ID,
            "glm123",
            "glm5_next",
        );
        let state = test_state_with(dir.path(), true, None);
        local_plan(&state, None);
        // A prepare job for the swap target, running long enough to overlap the request. The
        // command carries the target's converted scope, exactly as the real prepare does.
        let scope = {
            let config = state.usable_config().expect("config");
            let cache = config.hub_cache_dir().expect("hub cache");
            let model = hf::lookup(&cache, crate::config::GLM_MODEL_ID).expect("cached GLM");
            config
                .for_cached_model(&model)
                .expect("compatible target")
                .converted_path()
                .display()
                .to_string()
        };
        let spec = CommandSpec {
            program: "bash".to_owned(),
            args: vec!["-c".to_owned(), format!("sleep 5; # {scope}")],
            env: BTreeMap::new(),
            cwd: None,
            label: format!("Prepare {} on rank 0", crate::config::GLM_MODEL_ID),
            kind: "prepare-r0".to_owned(),
        };
        state
            .jobs
            .start_exclusive(spec, &[])
            .await
            .expect("prepare job");
        let app = test_router(state);
        let response = app
            .oneshot(post_json(
                "/api/actions/swap",
                &serde_json::json!({"id": crate::config::GLM_MODEL_ID}).to_string(),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = body_json(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("still being prepared"),
            "{body}"
        );
    }
}
