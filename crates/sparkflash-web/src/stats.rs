//! Cheap serving telemetry and its bounded history.
//!
//! The sampler runs outside the inference path: once every configured interval it reads the local
//! rank-0 `/metrics` and `/v1/cluster/stats` endpoints and folds them into an in-memory ring. The
//! ring is downsampled when a graph asks for a window, so a 24-hour view is a few hundred points,
//! not 43,200 line segments.
//!
//! Persistence is deliberately narrow. A JSON snapshot is written only while the cluster is idle
//! (the running and waiting gauges are both zero) and at most once per flush interval. Inference
//! never waits on this module's disk I/O.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::StatsConfig;

/// The maximum points returned to a chart; larger windows are averaged into buckets.
pub const MAX_SERIES_POINTS: usize = 600;

/// The on-disk schema version.
const PERSIST_VERSION: u32 = 1;

/// One node from `/v1/cluster/stats`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeInfo {
    /// The tensor-parallel rank.
    pub rank: u32,
    /// The node's hostname.
    pub hostname: String,
    /// Free coherent memory in GiB.
    pub uma_free_gib: f64,
    /// Total coherent memory in GiB.
    pub uma_total_gib: f64,
    /// The locked SM clock.
    pub gpu_clock_mhz: u32,
    /// The NVIDIA driver version.
    pub driver_version: String,
    /// Negotiated rail 0 speed.
    pub rail0_speed_mbps: u32,
    /// Negotiated rail 1 speed.
    pub rail1_speed_mbps: u32,
}

/// The cluster-level fields the UI reads, normalized from the server's JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterTelemetry {
    /// Served model id.
    pub model: String,
    /// Fabric description.
    pub fabric_mode: String,
    /// Advertised maximum sequence length.
    pub max_seq_len: u64,
    /// Per-rank state.
    pub nodes: Vec<NodeInfo>,
    /// Cumulative KV prefix-cache hit rate in [0, 1].
    pub cache_hit_rate: f64,
    /// Requests the KV prefix cache has looked up. Zero means the engine does not report it.
    #[serde(default)]
    pub cache_requests: u64,
    /// Cumulative DSpark draft acceptance rate in [0, 1].
    pub acceptance_rate: f64,
    /// Accepted draft tokens.
    pub draft_tokens_accepted: u64,
    /// Proposed draft tokens.
    pub draft_tokens_proposed: u64,
    /// Last request's decode speed.
    pub decode_tok_per_s: f64,
    /// Last request's prefill speed.
    pub prefill_tok_per_s: f64,
    /// Last request's time to first token.
    pub ttft_ms: f64,
}

impl Default for ClusterTelemetry {
    fn default() -> Self {
        Self {
            model: String::new(),
            fabric_mode: String::new(),
            max_seq_len: 0,
            nodes: Vec::new(),
            cache_hit_rate: 0.0,
            cache_requests: 0,
            acceptance_rate: 0.0,
            draft_tokens_accepted: 0,
            draft_tokens_proposed: 0,
            decode_tok_per_s: 0.0,
            prefill_tok_per_s: 0.0,
            ttft_ms: 0.0,
        }
    }
}

/// The Prometheus values the control plane consumes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsSnapshot {
    /// Requests currently executing.
    pub running: Option<u64>,
    /// Accepted requests not yet executing.
    pub waiting: Option<u64>,
    /// Cumulative prompt tokens.
    pub prompt_tokens: Option<u64>,
    /// Cumulative generated tokens.
    pub generation_tokens: Option<u64>,
    /// Cumulative completed requests.
    pub requests: Option<u64>,
    /// Cumulative refused requests.
    pub rejected: Option<u64>,
    /// Cumulative prompt tokens served from any prefix cache.
    pub cache_hit_tokens: Option<u64>,
    /// Cumulative accepted DSpark drafts.
    pub draft_tokens_accepted: Option<u64>,
    /// Cumulative proposed DSpark drafts.
    pub draft_tokens_proposed: Option<u64>,
    /// 1 while accepting requests, 0 while draining.
    pub serving: Option<u64>,
    /// The last request's decode tokens per second.
    pub last_decode_tok_per_s: Option<f64>,
    /// The last request's prefill tokens per second.
    pub last_prefill_tok_per_s: Option<f64>,
    /// The last request's acceptance rate.
    pub last_acceptance_rate: Option<f64>,
}

/// One scrape, before it is folded into history.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawSnapshot {
    /// Whether either endpoint answered.
    pub online: bool,
    /// Whether the serving gauge says requests are accepted.
    pub ready: bool,
    /// Prometheus counters and gauges.
    pub metrics: MetricsSnapshot,
    /// Cluster JSON, when it answered.
    pub cluster: Option<ClusterTelemetry>,
    /// A short scrape failure, retained for the UI.
    pub error: Option<String>,
    /// Why rank 0's `/health` refused (its device loop failed or stalled); `None` when it
    /// answered 200 or did not answer at all.
    pub unhealthy: Option<String>,
}

/// One point in the graph history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// Unix milliseconds.
    pub t_ms: u64,
    /// Whether rank 0 answered this scrape.
    pub online: bool,
    /// Whether rank 0 is accepting requests.
    pub ready: bool,
    /// Decode tokens per second of the last completed request; `None` before one completes.
    pub decode_tok_per_s: Option<f64>,
    /// Prefill tokens per second of the last completed request; `None` before one completes.
    pub prefill_tok_per_s: Option<f64>,
    /// DSpark draft acceptance in [0, 1]; `None` while no draft token has been proposed.
    pub acceptance_rate: Option<f64>,
    /// KV prefix-cache hit rate in [0, 1]; `None` when the engine reports no cache lookups.
    pub cache_hit_rate: Option<f64>,
    /// Requests executing.
    pub running: u64,
    /// Requests waiting.
    pub waiting: u64,
    /// Tokens generated per second across all requests over the sample gap; `None` without a
    /// previous scrape to compare with.
    ///
    /// Prompt tokens are left out: the server counts a prompt only when its request completes,
    /// so a long prompt would land in a single interval as a burst of hundreds of thousands of
    /// tokens per second.
    pub generated_tok_per_s: Option<f64>,
    /// Cumulative generated tokens reported by the serving process.
    pub generation_tokens: u64,
    /// Cumulative prompt tokens reported by the serving process.
    pub prompt_tokens: u64,
    /// Cumulative completed requests reported by the serving process.
    pub requests: u64,
    /// Cumulative accepted draft tokens.
    pub draft_tokens_accepted: u64,
    /// Cumulative proposed draft tokens.
    pub draft_tokens_proposed: u64,
    /// Tokens served since this control plane's history began (survives restarts).
    pub tokens_served_total: u64,
    /// Tokens served on the sample's UTC day.
    pub tokens_today: u64,
    /// Why rank 0's `/health` refused at this sample, when it did.
    #[serde(default)]
    pub unhealthy: Option<String>,
}

/// Cumulative totals that survive a control-plane restart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Totals {
    /// Prompt tokens seen across serving restarts.
    pub prompt_tokens: u64,
    /// Generated tokens seen across serving restarts.
    pub generation_tokens: u64,
    /// Completed requests seen across serving restarts.
    pub requests: u64,
    /// Total tokens (prompt plus generated) keyed by UTC date.
    pub tokens_by_day: BTreeMap<String, u64>,
}

impl Totals {
    /// Prompt plus generated tokens.
    #[must_use]
    pub const fn tokens_served(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.generation_tokens)
    }
}

#[derive(Debug, Clone, Default)]
struct Counters {
    prompt_tokens: u64,
    generation_tokens: u64,
    requests: u64,
    draft_tokens_accepted: u64,
    draft_tokens_proposed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedStats {
    version: u32,
    updated_ms: u64,
    totals: Totals,
    counters: Option<PersistedCounters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedCounters {
    prompt_tokens: u64,
    generation_tokens: u64,
    requests: u64,
    draft_tokens_accepted: u64,
    draft_tokens_proposed: u64,
}

struct Inner {
    samples: VecDeque<Sample>,
    totals: Totals,
    counters: Option<Counters>,
    last_sample_ms: Option<u64>,
    last_persist_ms: u64,
    last_idle: bool,
    cluster: Option<ClusterTelemetry>,
    last_error: Option<String>,
}

/// The graph window and the store state.
#[derive(Debug, Clone, Serialize)]
pub struct SeriesResponse {
    /// The requested window in seconds.
    pub window_secs: u64,
    /// The maximum number of points the server will return.
    pub max_points: usize,
    /// Downsampled samples, oldest first.
    pub points: Vec<Sample>,
    /// Cumulative totals.
    pub totals: Totals,
    /// Latest cluster JSON, when the last scrape had it.
    pub cluster: Option<ClusterTelemetry>,
    /// The latest scrape error, if any.
    pub last_error: Option<String>,
}

/// The in-memory telemetry store.
pub struct StatsStore {
    inner: Mutex<Inner>,
    retention_ms: u64,
    flush_ms: u64,
    persist_path: PathBuf,
}

impl StatsStore {
    /// Builds a store and loads any persisted totals beside `path`.
    #[must_use]
    pub fn new(config: &StatsConfig, path: PathBuf) -> Self {
        let mut totals = Totals::default();
        let mut counters = None;
        let mut last_persist_ms = 0;
        if let Ok(text) = std::fs::read_to_string(&path)
            && let Ok(persisted) = serde_json::from_str::<PersistedStats>(&text)
            && persisted.version == PERSIST_VERSION
        {
            totals = persisted.totals;
            counters = persisted.counters.map(|counters| Counters {
                prompt_tokens: counters.prompt_tokens,
                generation_tokens: counters.generation_tokens,
                requests: counters.requests,
                draft_tokens_accepted: counters.draft_tokens_accepted,
                draft_tokens_proposed: counters.draft_tokens_proposed,
            });
            last_persist_ms = persisted.updated_ms;
        }
        Self {
            inner: Mutex::new(Inner {
                samples: VecDeque::new(),
                totals,
                counters,
                last_sample_ms: None,
                last_persist_ms,
                last_idle: false,
                cluster: None,
                last_error: None,
            }),
            retention_ms: config.retention_secs.saturating_mul(1_000),
            flush_ms: config.flush_interval_secs.saturating_mul(1_000),
            persist_path: path,
        }
    }

    /// The path this store writes when the cluster is idle.
    #[must_use]
    pub fn persist_path(&self) -> &Path {
        &self.persist_path
    }

    /// Folds one scrape into the history and returns the resulting point.
    pub fn ingest(&self, raw: &RawSnapshot, now_ms: u64) -> Sample {
        let mut inner = self.inner.lock().expect("stats lock");
        if raw.online {
            inner.cluster = raw.cluster.clone();
            inner.last_error = raw.error.clone();
        } else if raw.error.is_some() {
            inner.last_error = raw.error.clone();
        }
        let sample = self.build_sample(&mut inner, raw, now_ms);
        inner.samples.push_back(sample.clone());
        let cutoff = now_ms.saturating_sub(self.retention_ms);
        while inner
            .samples
            .front()
            .is_some_and(|oldest| oldest.t_ms < cutoff)
        {
            inner.samples.pop_front();
        }
        let idle =
            raw.online && raw.metrics.running == Some(0) && raw.metrics.waiting.unwrap_or(0) == 0;
        inner.last_idle = idle;
        if idle && now_ms.saturating_sub(inner.last_persist_ms) >= self.flush_ms {
            let _ = self.persist_locked(&mut inner, now_ms);
        }
        sample
    }

    /// The newest point, if any.
    #[must_use]
    pub fn latest(&self) -> Option<Sample> {
        let inner = self.inner.lock().expect("stats lock");
        inner.samples.back().cloned()
    }

    /// The latest scrape error.
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        self.inner.lock().expect("stats lock").last_error.clone()
    }

    /// The latest cluster snapshot.
    #[must_use]
    pub fn cluster(&self) -> Option<ClusterTelemetry> {
        self.inner.lock().expect("stats lock").cluster.clone()
    }

    /// Cumulative totals.
    #[must_use]
    pub fn totals(&self) -> Totals {
        self.inner.lock().expect("stats lock").totals.clone()
    }

    /// A downsampled window ending at `now_ms`.
    #[must_use]
    pub fn series(&self, window_secs: u64, now_ms: u64, max_points: usize) -> SeriesResponse {
        let inner = self.inner.lock().expect("stats lock");
        let cutoff = now_ms.saturating_sub(window_secs.saturating_mul(1_000));
        let points: Vec<Sample> = inner
            .samples
            .iter()
            .filter(|sample| sample.t_ms >= cutoff)
            .cloned()
            .collect();
        SeriesResponse {
            window_secs,
            max_points,
            points: downsample(points, max_points),
            totals: inner.totals.clone(),
            cluster: inner.cluster.clone(),
            last_error: inner.last_error.clone(),
        }
    }

    /// Writes the current totals if the most recent scrape said the cluster was idle.
    ///
    /// # Errors
    /// Returns a message when the JSON cannot be encoded or the file cannot be replaced.
    pub fn persist_if_idle(&self) -> Result<bool, String> {
        let mut inner = self.inner.lock().expect("stats lock");
        if !inner.last_idle {
            return Ok(false);
        }
        let now = now_ms();
        self.persist_locked(&mut inner, now)?;
        Ok(true)
    }

    fn build_sample(&self, inner: &mut Inner, raw: &RawSnapshot, now_ms: u64) -> Sample {
        let metrics = &raw.metrics;
        let cluster = raw.cluster.as_ref();
        // Without a baseline the first counters still count toward the totals, but they are
        // everything the server has served so far, not a rate over the last interval.
        let had_baseline = inner.counters.is_some();
        let previous_counters = inner.counters.clone().unwrap_or_default();
        let current_counters = if raw.online && metrics.prompt_tokens.is_some() {
            Counters {
                prompt_tokens: metrics
                    .prompt_tokens
                    .unwrap_or(previous_counters.prompt_tokens),
                generation_tokens: metrics
                    .generation_tokens
                    .unwrap_or(previous_counters.generation_tokens),
                requests: metrics.requests.unwrap_or(previous_counters.requests),
                draft_tokens_accepted: metrics
                    .draft_tokens_accepted
                    .unwrap_or(previous_counters.draft_tokens_accepted),
                draft_tokens_proposed: metrics
                    .draft_tokens_proposed
                    .unwrap_or(previous_counters.draft_tokens_proposed),
            }
        } else {
            previous_counters.clone()
        };
        let prompt_delta = delta(
            previous_counters.prompt_tokens,
            current_counters.prompt_tokens,
        );
        let generation_delta = delta(
            previous_counters.generation_tokens,
            current_counters.generation_tokens,
        );
        let request_delta = delta(previous_counters.requests, current_counters.requests);
        if raw.online {
            inner.totals.prompt_tokens = inner.totals.prompt_tokens.saturating_add(prompt_delta);
            inner.totals.generation_tokens = inner
                .totals
                .generation_tokens
                .saturating_add(generation_delta);
            inner.totals.requests = inner.totals.requests.saturating_add(request_delta);
            let day = utc_day_key(now_ms);
            let day_total = inner.totals.tokens_by_day.entry(day).or_insert(0);
            *day_total = day_total.saturating_add(prompt_delta + generation_delta);
            inner.counters = Some(current_counters.clone());
        }
        let seconds = inner
            .last_sample_ms
            .filter(|previous| now_ms > *previous)
            .map(|previous| (now_ms - previous) as f64 / 1_000.0)
            .unwrap_or(0.0);
        let generated_tok_per_s = (seconds > 0.0 && raw.online && had_baseline)
            .then(|| finite(generation_delta as f64 / seconds));
        inner.last_sample_ms = Some(now_ms);

        // A zero speed gauge means no request has completed yet, not a stalled engine.
        let decode_tok_per_s = positive(metrics.last_decode_tok_per_s)
            .or_else(|| positive(cluster.map(|cluster| cluster.decode_tok_per_s)));
        let prefill_tok_per_s = positive(metrics.last_prefill_tok_per_s)
            .or_else(|| positive(cluster.map(|cluster| cluster.prefill_tok_per_s)));
        // Before the first draft (or with the drafter off, or on a build without the draft
        // counters) nothing has been proposed, and a 0% acceptance would be invented.
        let drafts_proposed = metrics
            .draft_tokens_proposed
            .or_else(|| cluster.map(|cluster| cluster.draft_tokens_proposed))
            .unwrap_or(0);
        let acceptance_rate = (drafts_proposed > 0)
            .then(|| {
                rate(metrics.draft_tokens_accepted, metrics.draft_tokens_proposed)
                    .or(metrics.last_acceptance_rate)
                    .or_else(|| cluster.map(|cluster| cluster.acceptance_rate))
            })
            .flatten();
        // An engine without a prefix cache reports zero hits and zero lookups; that is no data,
        // not a 0% hit rate.
        let cache_reported = metrics.cache_hit_tokens.is_some_and(|hits| hits > 0)
            || cluster.is_some_and(|cluster| cluster.cache_requests > 0);
        let cache_hit_rate = cache_reported
            .then(|| {
                rate(metrics.cache_hit_tokens, metrics.prompt_tokens)
                    .or_else(|| cluster.map(|cluster| cluster.cache_hit_rate))
            })
            .flatten();
        let day = utc_day_key(now_ms);
        Sample {
            t_ms: now_ms,
            online: raw.online,
            ready: raw.ready,
            decode_tok_per_s,
            prefill_tok_per_s,
            acceptance_rate: acceptance_rate.map(finite),
            cache_hit_rate: cache_hit_rate.map(finite),
            running: if raw.online {
                metrics.running.unwrap_or(0)
            } else {
                0
            },
            waiting: if raw.online {
                metrics.waiting.unwrap_or(0)
            } else {
                0
            },
            generated_tok_per_s,
            generation_tokens: current_counters.generation_tokens,
            prompt_tokens: current_counters.prompt_tokens,
            requests: current_counters.requests,
            draft_tokens_accepted: current_counters.draft_tokens_accepted,
            draft_tokens_proposed: current_counters.draft_tokens_proposed,
            tokens_served_total: inner.totals.tokens_served(),
            tokens_today: inner.totals.tokens_by_day.get(&day).copied().unwrap_or(0),
            unhealthy: if raw.online {
                raw.unhealthy.clone()
            } else {
                None
            },
        }
    }

    fn persist_locked(&self, inner: &mut Inner, now_ms: u64) -> Result<(), String> {
        let persisted = PersistedStats {
            version: PERSIST_VERSION,
            updated_ms: now_ms,
            totals: inner.totals.clone(),
            counters: inner.counters.as_ref().map(|counters| PersistedCounters {
                prompt_tokens: counters.prompt_tokens,
                generation_tokens: counters.generation_tokens,
                requests: counters.requests,
                draft_tokens_accepted: counters.draft_tokens_accepted,
                draft_tokens_proposed: counters.draft_tokens_proposed,
            }),
        };
        if let Some(parent) = self.persist_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(&persisted)
            .map_err(|error| format!("cannot encode stats: {error}"))?;
        let temp = self
            .persist_path
            .with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&temp, text)
            .map_err(|error| format!("cannot write {}: {error}", temp.display()))?;
        std::fs::rename(&temp, &self.persist_path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            format!("cannot replace {}: {error}", self.persist_path.display())
        })?;
        inner.last_persist_ms = now_ms;
        Ok(())
    }
}

/// Downloads one scrape from rank 0.
pub struct TelemetrySource {
    base_url: String,
    agent: ureq::Agent,
}

impl TelemetrySource {
    /// Builds a source for `serve_url`.
    #[must_use]
    pub fn new(serve_url: &str) -> Self {
        Self {
            base_url: serve_url.trim_end_matches('/').to_owned(),
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(3))
                .build(),
        }
    }

    /// Fetches both endpoints, tolerating one being unavailable.
    #[must_use]
    pub fn fetch(&self) -> RawSnapshot {
        let metrics_result = self.get("/metrics");
        let cluster_result = self.get("/v1/cluster/stats");
        let metrics = metrics_result
            .as_ref()
            .ok()
            .map(|text| metrics_snapshot(&parse_prometheus(text)))
            .unwrap_or_default();
        let cluster = cluster_result
            .as_ref()
            .ok()
            .and_then(|text| parse_cluster_stats(text));
        let online = metrics_result.is_ok() || cluster_result.is_ok();
        let ready = metrics.serving == Some(1);
        let unhealthy = if online { self.health() } else { None };
        let mut errors = Vec::new();
        if let Err(error) = &metrics_result {
            errors.push(format!("metrics: {error}"));
        }
        if let Err(error) = &cluster_result {
            errors.push(format!("cluster stats: {error}"));
        }
        RawSnapshot {
            online,
            ready,
            metrics,
            cluster,
            error: if errors.is_empty() {
                None
            } else {
                Some(errors.join("; "))
            },
            unhealthy,
        }
    }

    /// Rank 0's `/health` refusal: the reason a failed or stalled device loop gives with its
    /// 503. `None` when it answered 200, or did not answer (the other endpoints say so).
    fn health(&self) -> Option<String> {
        let url = format!("{}/health", self.base_url);
        match self.agent.get(&url).call() {
            Err(ureq::Error::Status(status, response)) => {
                let reason = response.into_string().unwrap_or_default();
                let reason = reason.trim();
                Some(if reason.is_empty() {
                    format!("/health answered {status}")
                } else {
                    reason.to_owned()
                })
            }
            Ok(_) | Err(ureq::Error::Transport(_)) => None,
        }
    }

    fn get(&self, path: &str) -> Result<String, String> {
        let url = format!("{}{path}", self.base_url);
        match self.agent.get(&url).call() {
            Ok(response) => response
                .into_string()
                .map_err(|error| format!("{url}: {error}")),
            Err(error) => Err(format!("{url}: {error}")),
        }
    }
}

/// Parses Prometheus text exposition into the last value of each family.
///
/// Labelled families keep the bare metric name; the control plane only consumes unlabelled
/// process-wide counters. Comments, malformed lines and timestamps are ignored.
#[must_use]
pub fn parse_prometheus(text: &str) -> BTreeMap<String, f64> {
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next() else {
            continue;
        };
        let name = name.split('{').next().unwrap_or(name);
        if let Ok(value) = value.parse::<f64>() {
            values.insert(name.to_owned(), value);
        }
    }
    values
}

/// Turns a Prometheus map into the fields the control plane consumes.
#[must_use]
pub fn metrics_snapshot(values: &BTreeMap<String, f64>) -> MetricsSnapshot {
    let counter = |name: &str| values.get(name).map(|value| as_u64(*value));
    MetricsSnapshot {
        running: counter("sparkflash_num_requests_running"),
        waiting: counter("sparkflash_num_requests_waiting"),
        prompt_tokens: counter("sparkflash_prompt_tokens_total"),
        generation_tokens: counter("sparkflash_generation_tokens_total"),
        requests: counter("sparkflash_requests_total"),
        rejected: counter("sparkflash_requests_rejected_total"),
        cache_hit_tokens: counter("sparkflash_global_kv_hit_tokens_total"),
        draft_tokens_accepted: counter("sparkflash_draft_tokens_accepted_total"),
        draft_tokens_proposed: counter("sparkflash_draft_tokens_proposed_total"),
        serving: counter("sparkflash_serving"),
        last_decode_tok_per_s: values
            .get("sparkflash_last_decode_tokens_per_second")
            .copied(),
        last_prefill_tok_per_s: values
            .get("sparkflash_last_prefill_tokens_per_second")
            .copied(),
        last_acceptance_rate: values.get("sparkflash_last_acceptance_rate").copied(),
    }
}

/// Parses the subset of `/v1/cluster/stats` the UI shows.
#[must_use]
pub fn parse_cluster_stats(text: &str) -> Option<ClusterTelemetry> {
    let value: Value = serde_json::from_str(text).ok()?;
    let nodes = value
        .get("nodes")
        .and_then(Value::as_array)
        .map(|nodes| {
            nodes
                .iter()
                .filter_map(|node| {
                    Some(NodeInfo {
                        rank: as_u64_value(node.get("rank")?) as u32,
                        hostname: node
                            .get("hostname")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        uma_free_gib: as_f64_value(node.get("uma_free_gib")?),
                        uma_total_gib: as_f64_value(node.get("uma_total_gib")?),
                        gpu_clock_mhz: as_u64_value(node.get("gpu_clock_mhz")?) as u32,
                        driver_version: node
                            .get("driver_version")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        rail0_speed_mbps: as_u64_value(node.get("rail0_speed_mbps")?) as u32,
                        rail1_speed_mbps: as_u64_value(node.get("rail1_speed_mbps")?) as u32,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let get_f64 = |pointer: &str| value.pointer(pointer).map(as_f64_value).unwrap_or_default();
    let get_u64 = |pointer: &str| value.pointer(pointer).map(as_u64_value).unwrap_or_default();
    Some(ClusterTelemetry {
        model: value
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        fabric_mode: value
            .get("fabric_mode")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        max_seq_len: as_u64_value(value.get("max_seq_len").unwrap_or(&Value::Null)),
        nodes,
        cache_hit_rate: get_f64("/kv/global_kv_hit_rate"),
        cache_requests: get_u64("/kv/requests_total"),
        acceptance_rate: get_f64("/dspark/acceptance_rate"),
        draft_tokens_accepted: get_u64("/dspark/draft_tokens_accepted"),
        draft_tokens_proposed: get_u64("/dspark/draft_tokens_proposed"),
        decode_tok_per_s: get_f64("/timing/decode_tok_per_s"),
        prefill_tok_per_s: get_f64("/timing/prefill_tok_per_s"),
        ttft_ms: get_f64("/timing/ttft_ms"),
    })
}

/// The UTC date key (`YYYY-MM-DD`) containing `unix_ms`.
#[must_use]
pub fn utc_day_key(unix_ms: u64) -> String {
    let days = i64::try_from(unix_ms / 86_400_000).unwrap_or(i64::MAX);
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Howard Hinnant's `civil_from_days`, valid for the full proleptic Gregorian range used here.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (
        year,
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
}

/// Averages a window into at most `max_points` buckets.
#[must_use]
pub fn downsample(points: Vec<Sample>, max_points: usize) -> Vec<Sample> {
    if max_points == 0 || points.len() <= max_points {
        return points;
    }
    let bucket = points.len().div_ceil(max_points);
    let mut output = Vec::with_capacity(points.len().div_ceil(bucket));
    let mut chunk = Vec::with_capacity(bucket);
    for point in points {
        chunk.push(point);
        if chunk.len() == bucket {
            output.push(average_bucket(&chunk));
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        output.push(average_bucket(&chunk));
    }
    output
}

fn average_bucket(bucket: &[Sample]) -> Sample {
    let last = bucket.last().expect("bucket is non-empty");
    Sample {
        t_ms: last.t_ms,
        online: bucket.iter().any(|sample| sample.online),
        ready: last.ready,
        decode_tok_per_s: mean(bucket.iter().map(|s| s.decode_tok_per_s)),
        prefill_tok_per_s: mean(bucket.iter().map(|s| s.prefill_tok_per_s)),
        acceptance_rate: mean(bucket.iter().map(|s| s.acceptance_rate)),
        cache_hit_rate: mean(bucket.iter().map(|s| s.cache_hit_rate)),
        running: bucket.iter().map(|s| s.running).max().unwrap_or(0),
        waiting: bucket.iter().map(|s| s.waiting).max().unwrap_or(0),
        generated_tok_per_s: mean(bucket.iter().map(|s| s.generated_tok_per_s)),
        generation_tokens: last.generation_tokens,
        prompt_tokens: last.prompt_tokens,
        requests: last.requests,
        draft_tokens_accepted: last.draft_tokens_accepted,
        draft_tokens_proposed: last.draft_tokens_proposed,
        tokens_served_total: last.tokens_served_total,
        tokens_today: last.tokens_today,
        unhealthy: last.unhealthy.clone(),
    }
}

/// The mean of the values present, or `None` when no sample in the bucket had one.
fn mean(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let (sum, count) = values
        .flatten()
        .fold((0.0, 0_u32), |(sum, count), value| (sum + value, count + 1));
    (count > 0).then(|| sum / f64::from(count))
}

fn delta(previous: u64, current: u64) -> u64 {
    if current >= previous {
        current - previous
    } else {
        current
    }
}

fn rate(numerator: Option<u64>, denominator: Option<u64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator > 0 => {
            Some(finite(numerator as f64 / denominator as f64))
        }
        _ => None,
    }
}

fn positive(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite() && *value > 0.0)
}

fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn as_u64(value: f64) -> u64 {
    if value.is_finite() && value >= 0.0 {
        value as u64
    } else {
        0
    }
}

fn as_u64_value(value: &Value) -> u64 {
    value.as_u64().unwrap_or_else(|| {
        value.as_f64().map_or(0, |value| {
            if value.is_finite() && value >= 0.0 {
                value as u64
            } else {
                0
            }
        })
    })
}

fn as_f64_value(value: &Value) -> f64 {
    value.as_f64().unwrap_or(0.0)
}

/// The current Unix time in milliseconds.
#[must_use]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

impl fmt::Debug for StatsStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StatsStore")
            .field("persist_path", &self.persist_path)
            .field("retention_ms", &self.retention_ms)
            .field("flush_ms", &self.flush_ms)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(prompt: u64, generation: u64, requests: u64) -> MetricsSnapshot {
        MetricsSnapshot {
            running: Some(0),
            waiting: Some(0),
            prompt_tokens: Some(prompt),
            generation_tokens: Some(generation),
            requests: Some(requests),
            serving: Some(1),
            ..MetricsSnapshot::default()
        }
    }

    fn raw(metrics: MetricsSnapshot) -> RawSnapshot {
        RawSnapshot {
            online: true,
            ready: true,
            metrics,
            cluster: None,
            error: None,
            unhealthy: None,
        }
    }

    fn config(dir: &Path) -> StatsConfig {
        StatsConfig {
            sample_interval_secs: 2,
            retention_secs: 3_600,
            persist_path: dir.join("stats.json").display().to_string(),
            flush_interval_secs: 1,
        }
    }

    #[test]
    fn prometheus_parser_reads_labelled_and_plain_families() {
        let text = "\
# HELP sparkflash_prompt_tokens_total prompt
# TYPE sparkflash_prompt_tokens_total counter
sparkflash_prompt_tokens_total 1200
sparkflash_uma_free_gib{rank=\"0\"} 9.5
sparkflash_num_requests_running 3
broken line
";
        let values = parse_prometheus(text);
        assert_eq!(values["sparkflash_prompt_tokens_total"], 1200.0);
        assert_eq!(values["sparkflash_uma_free_gib"], 9.5);
        assert_eq!(values["sparkflash_num_requests_running"], 3.0);
        assert!(!values.contains_key("broken"));
        let snapshot = metrics_snapshot(&values);
        assert_eq!(snapshot.prompt_tokens, Some(1200));
        assert_eq!(snapshot.running, Some(3));
        assert_eq!(snapshot.serving, None);
    }

    #[test]
    fn cluster_json_is_normalized() {
        let text = r#"{
          "model": "deepseek-ai/DeepSeek-V4.1-Flash",
          "fabric_mode": "tp4",
          "max_seq_len": 1048576,
          "nodes": [{"rank":0,"hostname":"spark0","uma_free_gib":9.5,"uma_total_gib":119.0,"gpu_clock_mhz":1800,"driver_version":"580.1","rail0_speed_mbps":200000,"rail1_speed_mbps":200000}],
          "kv": {"global_kv_hit_rate": 0.75},
          "dspark": {"acceptance_rate": 0.8, "draft_tokens_accepted": 400, "draft_tokens_proposed": 500},
          "timing": {"ttft_ms": 123.0, "prefill_tok_per_s": 5000.0, "decode_tok_per_s": 120.0}
        }"#;
        let cluster = parse_cluster_stats(text).expect("cluster");
        assert_eq!(cluster.nodes.len(), 1);
        assert_eq!(cluster.nodes[0].hostname, "spark0");
        assert_eq!(cluster.max_seq_len, 1_048_576);
        assert_eq!(cluster.acceptance_rate, 0.8);
        assert_eq!(cluster.decode_tok_per_s, 120.0);
        assert_eq!(cluster.cache_requests, 0);
        assert_eq!(parse_cluster_stats("not json"), None);
    }

    #[test]
    fn utc_days_are_calendar_days() {
        assert_eq!(utc_day_key(0), "1970-01-01");
        assert_eq!(utc_day_key(86_400_000), "1970-01-02");
        // 2024-02-29 is day 19,782 since the epoch.
        assert_eq!(utc_day_key(19_782 * 86_400_000), "2024-02-29");
    }

    #[test]
    fn counters_accumulate_deltas_and_survive_a_serving_restart() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = StatsStore::new(&config(dir.path()), dir.path().join("stats.json"));
        let offline = RawSnapshot {
            online: false,
            ready: false,
            metrics: MetricsSnapshot::default(),
            cluster: None,
            error: Some("connection refused".to_owned()),
            unhealthy: None,
        };
        store.ingest(&offline, 500);
        let first = store.ingest(&raw(metrics(100, 10, 1)), 1_000);
        assert_eq!(first.tokens_served_total, 110);
        // Tokens served before the first scrape are not a burst in the last interval.
        assert_eq!(first.generated_tok_per_s, None);
        let second = store.ingest(&raw(metrics(150, 40, 2)), 3_000);
        assert_eq!(second.tokens_served_total, 190);
        // 30 generated tokens over 2 s. The 50 prompt tokens were counted when their request
        // completed, so they count toward the totals but not the rate.
        assert_eq!(second.generated_tok_per_s, Some(15.0));
        assert_eq!(second.tokens_today, 190);
        assert_eq!(store.totals().requests, 2);
        // The serving process restarted: its counters are smaller, so the whole value is new work.
        let restarted = store.ingest(&raw(metrics(20, 5, 1)), 5_000);
        assert_eq!(restarted.tokens_served_total, 215);
        assert_eq!(store.totals().requests, 3);
    }

    #[test]
    fn a_long_prompt_completing_is_not_a_throughput_spike() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = StatsStore::new(&config(dir.path()), dir.path().join("stats.json"));
        store.ingest(&raw(metrics(0, 0, 0)), 1_000);
        // A 374,000-token prompt finishes with 40 generated tokens in this 2 s interval.
        let sample = store.ingest(&raw(metrics(374_000, 40, 1)), 3_000);
        assert_eq!(sample.generated_tok_per_s, Some(20.0));
        assert_eq!(sample.tokens_today, 374_040);
    }

    #[test]
    fn gauges_with_nothing_to_report_are_absent_rather_than_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = StatsStore::new(&config(dir.path()), dir.path().join("stats.json"));
        // The drafter is off, the engine has no prefix cache and no request has completed.
        let mut quiet = metrics(1_000, 10, 1);
        quiet.draft_tokens_proposed = Some(0);
        quiet.draft_tokens_accepted = Some(0);
        quiet.cache_hit_tokens = Some(0);
        quiet.last_decode_tok_per_s = Some(0.0);
        quiet.last_acceptance_rate = Some(0.0);
        let mut snapshot = raw(quiet);
        snapshot.cluster = Some(ClusterTelemetry::default());
        let sample = store.ingest(&snapshot, 1_000);
        assert_eq!(sample.decode_tok_per_s, None);
        assert_eq!(sample.prefill_tok_per_s, None);
        assert_eq!(sample.acceptance_rate, None);
        assert_eq!(sample.cache_hit_rate, None);

        let mut active = metrics(1_000, 10, 1);
        active.draft_tokens_proposed = Some(500);
        active.draft_tokens_accepted = Some(400);
        active.cache_hit_tokens = Some(0);
        active.last_decode_tok_per_s = Some(42.0);
        let mut snapshot = raw(active);
        // The cache exists and has looked up requests, so zero hits is a real 0%.
        snapshot.cluster = Some(ClusterTelemetry {
            cache_requests: 3,
            ..ClusterTelemetry::default()
        });
        let sample = store.ingest(&snapshot, 3_000);
        assert_eq!(sample.decode_tok_per_s, Some(42.0));
        assert_eq!(sample.acceptance_rate, Some(0.8));
        assert_eq!(sample.cache_hit_rate, Some(0.0));
    }

    #[test]
    fn bucket_means_skip_missing_values() {
        assert_eq!(mean([None, Some(2.0), Some(4.0)].into_iter()), Some(3.0));
        assert_eq!(mean([None, None].into_iter()), None);
    }

    #[test]
    fn persistence_happens_only_while_idle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stats_config = config(dir.path());
        let path = dir.path().join("stats.json");
        let store = StatsStore::new(&stats_config, path.clone());
        let mut busy = metrics(10, 1, 1);
        busy.running = Some(2);
        store.ingest(&raw(busy.clone()), 1_000);
        assert!(!path.exists(), "busy sample must not write");

        store.ingest(&raw(metrics(10, 1, 1)), 2_000);
        assert!(path.exists(), "idle sample writes");
        let written = std::fs::read_to_string(&path).expect("read");
        assert!(written.contains("\"version\": 1"));
        assert!(
            store.persist_if_idle().expect("idle persist"),
            "an explicit idle snapshot is always allowed"
        );

        busy.running = Some(1);
        store.ingest(&raw(busy), 5_000);
        assert!(!store.persist_if_idle().expect("busy persist"));
    }

    #[test]
    fn totals_reload_from_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stats_config = config(dir.path());
        let path = dir.path().join("stats.json");
        let store = StatsStore::new(&stats_config, path.clone());
        store.ingest(&raw(metrics(100, 50, 3)), 2_000);
        drop(store);
        let reloaded = StatsStore::new(&stats_config, path);
        let totals = reloaded.totals();
        assert_eq!(totals.prompt_tokens, 100);
        assert_eq!(totals.generation_tokens, 50);
        assert_eq!(totals.requests, 3);
        // The persisted counter baseline continues the delta across a control-plane restart.
        let next = reloaded.ingest(&raw(metrics(120, 55, 4)), 4_000);
        assert_eq!(next.tokens_served_total, 175);
    }

    #[test]
    fn series_windows_and_downsampling_keep_endpoints() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = StatsStore::new(&config(dir.path()), dir.path().join("stats.json"));
        for index in 0..100u64 {
            let mut snapshot = raw(metrics(index * 10, index, index));
            snapshot.metrics.last_decode_tok_per_s = Some(index as f64);
            store.ingest(&snapshot, (index + 1) * 1_000);
        }
        let series = store.series(3_600, 100_000, 10);
        assert!(series.points.len() <= 10);
        assert_eq!(series.points.last().expect("last").t_ms, 100_000);
        assert!(series.points.first().expect("first").t_ms <= 10_000);
        // The first sample establishes the baseline; the remaining 99 add one request each.
        assert_eq!(series.totals.requests, 99);
    }

    #[test]
    fn offline_samples_do_not_invent_tokens() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = StatsStore::new(&config(dir.path()), dir.path().join("stats.json"));
        let offline = RawSnapshot {
            online: false,
            ready: false,
            metrics: MetricsSnapshot::default(),
            cluster: None,
            error: Some("connection refused".to_owned()),
            unhealthy: None,
        };
        let sample = store.ingest(&offline, 1_000);
        assert!(!sample.online);
        assert_eq!(sample.tokens_served_total, 0);
        assert_eq!(store.last_error().as_deref(), Some("connection refused"));
    }

    /// Serves each request with the reply its path names, one connection per request.
    fn fake_rank0(replies: &'static [(&'static str, &'static str, &'static str)]) -> String {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut line = String::new();
                if BufReader::new(&stream).read_line(&mut line).is_err() {
                    continue;
                }
                let path = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_owned();
                let (status, body) = replies
                    .iter()
                    .find(|(route, _, _)| *route == path)
                    .map_or(("404 Not Found", ""), |(_, status, body)| (*status, *body));
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        format!("http://{address}")
    }

    #[test]
    fn a_stalled_rank_zero_reports_its_health_reason() {
        let url = fake_rank0(&[
            (
                "/metrics",
                "200 OK",
                "sparkflash_serving 1\nsparkflash_num_requests_running 2\n",
            ),
            (
                "/health",
                "503 Service Unavailable",
                "device loop stalled: no progress for 130 s",
            ),
        ]);
        let raw = TelemetrySource::new(&url).fetch();
        assert!(raw.online && raw.ready);
        assert_eq!(
            raw.unhealthy.as_deref(),
            Some("device loop stalled: no progress for 130 s")
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let store = StatsStore::new(&config(dir.path()), dir.path().join("stats.json"));
        let sample = store.ingest(&raw, 1_000);
        assert_eq!(sample.unhealthy, raw.unhealthy);

        let healthy = fake_rank0(&[
            ("/metrics", "200 OK", "sparkflash_serving 1\n"),
            ("/health", "200 OK", ""),
        ]);
        assert_eq!(TelemetrySource::new(&healthy).fetch().unhealthy, None);
    }
}
