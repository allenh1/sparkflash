//! Prometheus text exposition for Grafana, sparkDash and Ramjet tier-aware routers.
//!
//! The exporter owns a private Registry (never the process-global default), so tests can
//! build independent AppStates without duplicate-registration conflicts. Aggregate counters
//! are bumped as requests complete; the scheduler-state gauges are bumped by the engine driver
//! as requests are admitted and finish, the generation-token counter and the time-to-first-token
//! histogram are fed from the same driver-side events, and the per-layer and per-rank gauges are
//! refreshed from the engine's cluster snapshot at scrape time.
//!
//! The four ops families added for S4 parity keep the project's `sparkflash_` prefix and mirror
//! the vLLM metric they correspond to in name and definition: running/waiting are
//! `vllm:num_requests_running`/`vllm:num_requests_waiting`, the counter is
//! `vllm:generation_tokens_total`, and the histogram is
//! `vllm:time_to_first_token_seconds` with vLLM's default bucket layout
//! (vllm/v1/metrics/buckets.py). Only facts the host path can measure are exported: there is no
//! prefix-cache counter because this process has no prefix cache, and no GPU-cache gauge on a
//! CPU reference path. See docs/archive/build/serve-ops-metrics.md.

use prometheus::{
    Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, IntCounter, IntGauge, IntGaugeVec, Opts,
    Registry, TextEncoder,
};
use sparkflash_kv::PrefixMatchPlan;

use crate::engine::ClusterStatsSnapshot;
use crate::telemetry_types::{DeepSeekSparkRequestStats, OpenAiUsageWithCache, finite_or_zero};

/// vLLM's default bucket layout for the time-to-first-token histogram.
///
/// Copied deliberately from vLLM's single source of truth
/// (vllm/v1/metrics/buckets.py, TIME_TO_FIRST_TOKEN_BUCKETS: millisecond-scale prefill for tiny
/// prompts up to the multi-minute worst case, densest around interactive sub-second latencies).
/// Keeping the boundaries identical is what lets a dashboard or an SLO compare the two servers
/// bucket by bucket instead of approximating.
pub const TIME_TO_FIRST_TOKEN_BUCKETS: [f64; 22] = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];

/// The SparkFlash Prometheus exporter.
pub struct Metrics {
    registry: Registry,
    requests_total: IntCounter,
    prompt_tokens_total: IntCounter,
    /// Requests currently executing (admitted to a batch slot or the serial run loop).
    num_requests_running: IntGauge,
    /// Requests accepted by the driver and queued, not yet admitted for execution.
    num_requests_waiting: IntGauge,
    /// Generated token ids, counted as each token is generated.
    generation_tokens_total: IntCounter,
    /// Seconds from engine-queue admission to the first generated token.
    time_to_first_token_seconds: Histogram,
    /// 1 while the server is accepting requests, 0 after the drain signal.
    serving: IntGauge,
    /// Requests refused before engine admission because the bounded job queue was full.
    requests_rejected_total: IntCounter,
    global_kv_hit_tokens_total: IntCounter,
    swa_tail_cache_hits_total: IntCounter,
    /// DSpark draft tokens accepted by the target model, cumulative.
    draft_tokens_accepted_total: IntCounter,
    /// DSpark draft tokens proposed by the drafter, cumulative.
    draft_tokens_proposed_total: IntCounter,
    /// The last completed request's decode speed.
    last_decode_tokens_per_second: Gauge,
    /// The last completed request's prefill speed.
    last_prefill_tokens_per_second: Gauge,
    /// The last completed request's DSpark acceptance rate in [0, 1].
    last_acceptance_rate: Gauge,
    /// The last completed request's time to first token, in seconds.
    last_time_to_first_token_seconds: Gauge,
    engram_lookups_total: IntGaugeVec,
    engram_hit_rate: GaugeVec,
    trace_active_scope_mask: IntGauge,
    uma_free_gib: GaugeVec,
}

impl Metrics {
    /// Builds and registers every collector on a fresh Registry.
    ///
    /// # Panics
    /// Panics only if a static metric name or help string is malformed, which is a
    /// programming error rather than a request failure.
    #[must_use]
    pub fn new() -> Self {
        let registry = Registry::new();

        let requests_total = IntCounter::new(
            "sparkflash_requests_total",
            "Total inference requests served",
        )
        .expect("static metric descriptor");
        let prompt_tokens_total = IntCounter::new(
            "sparkflash_prompt_tokens_total",
            "Total prompt tokens across all requests",
        )
        .expect("static metric descriptor");
        let global_kv_hit_tokens_total = IntCounter::new(
            "sparkflash_global_kv_hit_tokens_total",
            "Total prompt tokens served from any Global KV cache tier",
        )
        .expect("static metric descriptor");
        let swa_tail_cache_hits_total = IntCounter::new(
            "sparkflash_swa_tail_cache_hits_total",
            "Total requests whose SWA tail was found in the ring",
        )
        .expect("static metric descriptor");
        let draft_tokens_accepted_total = IntCounter::new(
            "sparkflash_draft_tokens_accepted_total",
            "Total DSpark draft tokens accepted by the target model",
        )
        .expect("static metric descriptor");
        let draft_tokens_proposed_total = IntCounter::new(
            "sparkflash_draft_tokens_proposed_total",
            "Total DSpark draft tokens proposed by the drafter",
        )
        .expect("static metric descriptor");
        let last_decode_tokens_per_second = Gauge::new(
            "sparkflash_last_decode_tokens_per_second",
            "Decode tokens per second measured for the most recently completed request",
        )
        .expect("static metric descriptor");
        let last_prefill_tokens_per_second = Gauge::new(
            "sparkflash_last_prefill_tokens_per_second",
            "Prefill tokens per second measured for the most recently completed request",
        )
        .expect("static metric descriptor");
        let last_acceptance_rate = Gauge::new(
            "sparkflash_last_acceptance_rate",
            "DSpark draft acceptance rate for the most recently completed request",
        )
        .expect("static metric descriptor");
        let last_time_to_first_token_seconds = Gauge::new(
            "sparkflash_last_time_to_first_token_seconds",
            "Time to first token for the most recently completed request, in seconds",
        )
        .expect("static metric descriptor");
        let engram_lookups_total = IntGaugeVec::new(
            Opts::new(
                "sparkflash_engram_lookups_total",
                "Engram lookups per layer",
            ),
            &["layer"],
        )
        .expect("static metric descriptor");
        let engram_hit_rate = GaugeVec::new(
            Opts::new(
                "sparkflash_engram_hit_rate",
                "Engram tier1+tier2 hit rate per layer",
            ),
            &["layer"],
        )
        .expect("static metric descriptor");
        let trace_active_scope_mask = IntGauge::new(
            "sparkflash_trace_active_scope_mask",
            "Currently armed trace scope bitmask",
        )
        .expect("static metric descriptor");
        let uma_free_gib = GaugeVec::new(
            Opts::new(
                "sparkflash_uma_free_gib",
                "Free coherent UMA memory per rank in GiB",
            ),
            &["rank"],
        )
        .expect("static metric descriptor");
        let num_requests_running = IntGauge::new(
            "sparkflash_num_requests_running",
            "Number of requests currently executing (vLLM vllm:num_requests_running semantics)",
        )
        .expect("static metric descriptor");
        let num_requests_waiting = IntGauge::new(
            "sparkflash_num_requests_waiting",
            "Number of requests accepted by the driver and queued, not yet executing (vLLM vllm:num_requests_waiting semantics)",
        )
        .expect("static metric descriptor");
        let generation_tokens_total = IntCounter::new(
            "sparkflash_generation_tokens_total",
            "Total generated token ids, counted as each token is generated (vLLM vllm:generation_tokens_total semantics)",
        )
        .expect("static metric descriptor");
        let serving = IntGauge::new(
            "sparkflash_serving",
            "1 while the server is accepting requests, 0 after the graceful-drain signal (vLLM's ServerStatus does not export this; added for drain-aware probes and dashboards)",
        )
        .expect("static metric descriptor");
        let requests_rejected_total = IntCounter::new(
            "sparkflash_requests_rejected_total",
            "Total requests refused before engine admission because the bounded driver job queue was full (backpressure; the refusal is a typed 503 and the request never entered the queue)",
        )
        .expect("static metric descriptor");
        let time_to_first_token_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "sparkflash_time_to_first_token_seconds",
                "Seconds from engine-queue admission to the first generated token, including scheduler queue wait and prefill (vLLM vllm:time_to_first_token_seconds semantics; vLLM default buckets)",
            )
            .buckets(TIME_TO_FIRST_TOKEN_BUCKETS.to_vec()),
        )
        .expect("static metric descriptor");

        for collector in [
            Box::new(requests_total.clone()) as Box<dyn prometheus::core::Collector>,
            Box::new(prompt_tokens_total.clone()),
            Box::new(global_kv_hit_tokens_total.clone()),
            Box::new(swa_tail_cache_hits_total.clone()),
            Box::new(engram_lookups_total.clone()),
            Box::new(engram_hit_rate.clone()),
            Box::new(trace_active_scope_mask.clone()),
            Box::new(uma_free_gib.clone()),
            Box::new(num_requests_running.clone()),
            Box::new(num_requests_waiting.clone()),
            Box::new(generation_tokens_total.clone()),
            Box::new(serving.clone()),
            Box::new(requests_rejected_total.clone()),
            Box::new(time_to_first_token_seconds.clone()),
            Box::new(draft_tokens_accepted_total.clone()),
            Box::new(draft_tokens_proposed_total.clone()),
            Box::new(last_decode_tokens_per_second.clone()),
            Box::new(last_prefill_tokens_per_second.clone()),
            Box::new(last_acceptance_rate.clone()),
            Box::new(last_time_to_first_token_seconds.clone()),
        ] {
            registry
                .register(collector)
                .expect("static metric name is unique");
        }

        serving.set(1);
        Self {
            registry,
            requests_total,
            prompt_tokens_total,
            global_kv_hit_tokens_total,
            swa_tail_cache_hits_total,
            engram_lookups_total,
            engram_hit_rate,
            trace_active_scope_mask,
            uma_free_gib,
            num_requests_running,
            num_requests_waiting,
            generation_tokens_total,
            time_to_first_token_seconds,
            serving,
            requests_rejected_total,
            draft_tokens_accepted_total,
            draft_tokens_proposed_total,
            last_decode_tokens_per_second,
            last_prefill_tokens_per_second,
            last_acceptance_rate,
            last_time_to_first_token_seconds,
        }
    }

    /// Records the speculative and timing facts of one completed request.
    ///
    /// The counters are cumulative so a dashboard can derive a rate over any window; the gauges
    /// are the last completed request's values, which is what the control plane's per-request
    /// graph shows. Both are cheap increments/sets in the existing request-completion path.
    pub fn record_request_stats(&self, stats: &DeepSeekSparkRequestStats) {
        self.draft_tokens_accepted_total
            .inc_by(stats.dspark_speculative.draft_tokens_accepted as u64);
        self.draft_tokens_proposed_total
            .inc_by(stats.dspark_speculative.draft_tokens_proposed as u64);
        self.last_decode_tokens_per_second
            .set(finite_or_zero(stats.timing.decode_tok_per_s));
        self.last_prefill_tokens_per_second
            .set(finite_or_zero(stats.timing.prefill_tok_per_s));
        self.last_acceptance_rate
            .set(finite_or_zero(stats.dspark_speculative.acceptance_rate));
        self.last_time_to_first_token_seconds
            .set(finite_or_zero(stats.timing.ttft_ms / 1_000.0));
    }

    /// Records the aggregate counters of one completed request.
    pub fn record_request(&self, usage: &OpenAiUsageWithCache, plan: &PrefixMatchPlan) {
        self.requests_total.inc();
        self.prompt_tokens_total.inc_by(usage.prompt_tokens as u64);
        self.global_kv_hit_tokens_total
            .inc_by((plan.uma_hit_tokens + plan.nvme_restored_tokens) as u64);
        if plan.swa_tail_hit {
            self.swa_tail_cache_hits_total.inc();
        }
    }

    /// Records that one request entered the engine queue, before it is enqueued.
    ///
    /// The driver calls this immediately before the channel send, so a scrape never sees a
    /// request that is queued in the channel but missing from the gauge. The engine thread moves
    /// the same request out of the gauge when it admits it.
    pub fn request_enqueued(&self) {
        self.num_requests_waiting.inc();
    }

    /// Records that the enqueue was refused because the queue was full.
    ///
    /// Undoes Self::request_enqueued for the request that never entered the queue; a refusal is
    /// not a queued request and must not show up as one. The refusal also increments
    /// `sparkflash_requests_rejected_total`, so a dashboard can alert on backpressure without
    /// watching the 503 bodies.
    pub fn request_enqueue_refused(&self) {
        self.decrement_gauge(&self.num_requests_waiting);
        self.requests_rejected_total.inc();
    }

    /// Sets the drain state gauge: 1 while accepting requests, 0 once the drain signal is in.
    pub fn set_serving(&self, accepting: bool) {
        self.serving.set(i64::from(accepting));
    }

    /// Records admission of one queued request: it stops waiting and starts executing.
    ///
    /// One call per admitted request, from the engine thread, immediately before the request's
    /// first engine work.
    pub fn request_admitted(&self) {
        self.decrement_gauge(&self.num_requests_waiting);
        self.num_requests_running.inc();
    }

    /// Records that one admitted request stopped executing, for any terminal outcome.
    ///
    /// One call per admitted request after the terminal reply (or the failed/panicked path); the
    /// driver calls it from the same thread that admitted the request.
    pub fn request_finished(&self) {
        self.decrement_gauge(&self.num_requests_running);
    }

    /// Counts generated token ids.
    ///
    /// Counted as the tokens are generated, at the same point the wire's completion_tokens
    /// counter is bumped, so the counter and the usage field agree for a request that completes.
    /// A request that fails after generating some tokens has those tokens counted too: they were
    /// real engine work even though no usage was ever reported for them.
    pub fn record_generation_tokens(&self, count: u64) {
        if count > 0 {
            self.generation_tokens_total.inc_by(count);
        }
    }

    /// Observes one time-to-first-token sample, in seconds.
    ///
    /// The definition and the bucket layout mirror vLLM. A value that is not a finite,
    /// non-negative duration is not a measurement and is dropped rather than exported as one.
    pub fn observe_time_to_first_token(&self, seconds: f64) {
        if seconds.is_finite() && seconds >= 0.0 {
            self.time_to_first_token_seconds.observe(seconds);
        }
    }

    /// Decrements a gauge without ever letting it go negative.
    ///
    /// The engine driver is a single request-lifecycle thread, so the read-modify-write is not
    /// racing another owner; refusing to go below zero keeps an unbalanced caller from exporting
    /// a request that does not exist.
    fn decrement_gauge(&self, gauge: &IntGauge) {
        let current = gauge.get();
        if current > 0 {
            gauge.set(current - 1);
        }
    }

    /// Refreshes the scrape-time gauges from a cluster snapshot.
    pub fn observe_cluster(&self, snapshot: &ClusterStatsSnapshot, trace_mask: u64) {
        self.trace_active_scope_mask.set(trace_mask as i64);
        for layer in &snapshot.engram {
            let label = layer.layer_id.to_string();
            self.engram_lookups_total
                .with_label_values(&[&label])
                .set(layer.lookups_total as i64);
            self.engram_hit_rate
                .with_label_values(&[&label])
                .set(finite_or_zero(layer.hit_rate));
        }
        for node in &snapshot.nodes {
            let label = node.rank.to_string();
            self.uma_free_gib
                .with_label_values(&[&label])
                .set(node.uma_free_gib);
        }
    }

    /// Renders the registry in Prometheus text exposition format.
    #[must_use]
    pub fn render(&self) -> String {
        let mut buffer = Vec::new();
        let encoder = TextEncoder::new();
        if encoder
            .encode(&self.registry.gather(), &mut buffer)
            .is_err()
        {
            return String::new();
        }
        String::from_utf8(buffer).unwrap_or_default()
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ClusterStatsSnapshot;

    #[test]
    fn every_documented_metric_family_is_exported() {
        let metrics = Metrics::new();
        metrics.record_request(
            &OpenAiUsageWithCache {
                prompt_tokens: 100,
                completion_tokens: 10,
                total_tokens: 110,
                prompt_tokens_details: crate::telemetry_types::PromptTokensDetails {
                    cached_tokens: 64,
                    uma_cached_tokens: 64,
                    nvme_cached_tokens: 0,
                    image_tokens: 0,
                },
                completion_tokens_details: Default::default(),
                prompt_cache_hit_tokens: 64,
                prompt_cache_miss_tokens: 36,
            },
            &crate::engine::plan_for(100, 64, 0, true),
        );
        metrics.record_request_stats(&DeepSeekSparkRequestStats {
            kv_cache: crate::engine::plan_for(100, 64, 0, true),
            kv_cache_hit_rate: 0.64,
            engram_layers: crate::engine::default_engram_layers(),
            dspark_speculative: crate::telemetry_types::DSparkStepStats::from_counts(10, 50, 0, 40),
            transport: crate::telemetry_types::RoceTransportStats::default(),
            timing: crate::telemetry_types::RequestTimingStats {
                ttft_ms: 250.0,
                prefill_tok_per_s: 4_000.0,
                decode_tok_per_s: 120.0,
                min_rank_uma_free_gib: 8.0,
            },
        });
        let snapshot = ClusterStatsSnapshot::sample();
        metrics.observe_cluster(&snapshot, 0b1010_0101);
        let text = metrics.render();
        for name in [
            "sparkflash_requests_total",
            "sparkflash_prompt_tokens_total",
            "sparkflash_global_kv_hit_tokens_total",
            "sparkflash_swa_tail_cache_hits_total",
            "sparkflash_engram_lookups_total",
            "sparkflash_engram_hit_rate",
            "sparkflash_trace_active_scope_mask",
            "sparkflash_uma_free_gib",
            "sparkflash_draft_tokens_accepted_total",
            "sparkflash_draft_tokens_proposed_total",
            "sparkflash_last_decode_tokens_per_second",
            "sparkflash_last_prefill_tokens_per_second",
            "sparkflash_last_acceptance_rate",
            "sparkflash_last_time_to_first_token_seconds",
        ] {
            assert!(text.contains(name), "missing metric {name}\n{text}");
        }
        assert!(text.contains("layer=\"1\""));
        assert!(text.contains("rank=\"0\""));
        assert!(text.contains("sparkflash_draft_tokens_accepted_total 40"));
        assert!(text.contains("sparkflash_last_decode_tokens_per_second 120"));
    }

    #[test]
    fn ops_families_export_vllm_shaped_state() {
        let metrics = Metrics::new();
        metrics.request_enqueued();
        metrics.request_enqueued();
        metrics.request_enqueue_refused();
        assert!(
            metrics
                .render()
                .contains("sparkflash_num_requests_waiting 1")
        );
        // A refused enqueue never enters the queue but is counted as backpressure, and the
        // serving gauge is 1 until the drain signal asks it to turn 0.
        assert!(
            metrics
                .render()
                .contains("sparkflash_requests_rejected_total 1")
        );
        assert!(metrics.render().contains("sparkflash_serving 1"));
        metrics.set_serving(false);
        let drained = metrics.render();
        assert!(drained.contains("sparkflash_serving 0"), "{drained}");
        metrics.set_serving(true);
        assert!(metrics.render().contains("sparkflash_serving 1"));
        assert!(
            metrics
                .render()
                .contains("sparkflash_num_requests_running 0")
        );

        metrics.request_admitted();
        let during = metrics.render();
        assert!(
            during.contains("sparkflash_num_requests_waiting 0"),
            "{during}"
        );
        assert!(
            during.contains("sparkflash_num_requests_running 1"),
            "{during}"
        );

        metrics.record_generation_tokens(5);
        metrics.observe_time_to_first_token(0.02);
        metrics.observe_time_to_first_token(3.0);
        metrics.observe_time_to_first_token(f64::NAN);
        metrics.observe_time_to_first_token(-1.0);
        metrics.request_finished();
        let text = metrics.render();
        assert!(text.contains("sparkflash_generation_tokens_total 5"));
        assert!(
            text.contains(
                "# HELP sparkflash_time_to_first_token_seconds Seconds from engine-queue admission"
            ),
            "{text}"
        );
        assert!(
            text.contains("sparkflash_time_to_first_token_seconds_bucket{le=\"0.02\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("sparkflash_time_to_first_token_seconds_bucket{le=\"+Inf\"} 2"),
            "{text}"
        );
        assert!(
            text.contains("sparkflash_time_to_first_token_seconds_count 2"),
            "{text}"
        );
        assert!(text.contains("sparkflash_num_requests_running 0"), "{text}");
        assert!(text.contains("sparkflash_num_requests_waiting 0"), "{text}");

        // Balanced accounting survives a re-scrape: every family is still exported and the
        // generator counter is monotone.
        assert!(
            metrics
                .render()
                .contains("sparkflash_generation_tokens_total 5")
        );
        assert!(metrics.render().contains("sparkflash_requests_total 0"));
    }

    #[test]
    fn registries_are_private_per_instance() {
        let first = Metrics::new();
        let second = Metrics::new();
        first.record_request(
            &OpenAiUsageWithCache::default(),
            &crate::engine::plan_for(0, 0, 0, false),
        );
        assert!(first.render().contains("sparkflash_requests_total 1"));
        assert!(second.render().contains("sparkflash_requests_total 0"));
    }
}
