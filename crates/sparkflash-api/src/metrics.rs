//! Prometheus text exposition for Grafana, sparkDash and Ramjet tier-aware routers.
//!
//! The exporter owns a private Registry (never the process-global default), so tests can
//! build independent AppStates without duplicate-registration conflicts. Aggregate counters
//! are bumped as requests complete; the per-layer and per-rank gauges are refreshed from
//! the engine's cluster snapshot at scrape time.

use prometheus::{
    Encoder, GaugeVec, IntCounter, IntGauge, IntGaugeVec, Opts, Registry, TextEncoder,
};
use sparkflash_kv::PrefixMatchPlan;

use crate::engine::ClusterStatsSnapshot;
use crate::telemetry_types::{OpenAiUsageWithCache, finite_or_zero};

/// The SparkFlash Prometheus exporter.
pub struct Metrics {
    registry: Registry,
    requests_total: IntCounter,
    prompt_tokens_total: IntCounter,
    global_kv_hit_tokens_total: IntCounter,
    swa_tail_cache_hits_total: IntCounter,
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

        for collector in [
            Box::new(requests_total.clone()) as Box<dyn prometheus::core::Collector>,
            Box::new(prompt_tokens_total.clone()),
            Box::new(global_kv_hit_tokens_total.clone()),
            Box::new(swa_tail_cache_hits_total.clone()),
            Box::new(engram_lookups_total.clone()),
            Box::new(engram_hit_rate.clone()),
            Box::new(trace_active_scope_mask.clone()),
            Box::new(uma_free_gib.clone()),
        ] {
            registry
                .register(collector)
                .expect("static metric name is unique");
        }

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
        }
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
            },
            &crate::engine::plan_for(100, 64, 0, true),
        );
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
        ] {
            assert!(text.contains(name), "missing metric {name}\n{text}");
        }
        assert!(text.contains("layer=\"1\""));
        assert!(text.contains("rank=\"0\""));
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
