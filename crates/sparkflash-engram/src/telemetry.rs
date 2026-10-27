//! Per-layer Engram telemetry exported through the OpenAI usage block, /v1/cluster/stats
//! and Prometheus.

use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed-point scale used for the mean gate alpha accumulator (micro-units).
pub const GATE_ALPHA_SCALE: u64 = 1_000_000;

/// Per-layer, per-tier counters. Cheap enough to update on every lookup: relaxed atomics only.
#[derive(Debug, Default)]
pub struct EngramLayerTelemetry {
    /// 24-column hash lookups issued by this layer (one per n-gram head per token).
    pub lookups_total: AtomicU64,
    /// Lookups served from the tier-1 tier of the cache (the DRM host buffer).
    pub tier1_drm_hits: AtomicU64,
    /// Lookups served from the tier-2 UMA cache.
    pub tier2_uma_hits: AtomicU64,
    /// Lookups that had to reach the NVMe store.
    pub tier3_nvme_misses: AtomicU64,
    /// Tokens excluded from n-gram hashing because they fall inside an image span.
    pub image_tokens_masked: AtomicU64,
    /// Total time tokens waited on Engram I/O, in nanoseconds.
    pub prefetch_wait_ns_total: AtomicU64,
    /// Sum of mean gate alpha, scaled by GATE_ALPHA_SCALE.
    pub gate_alpha_sum_micro: AtomicU64,
    /// Number of token batches whose gate alpha was accumulated.
    pub gate_samples_total: AtomicU64,
}

pub use sparkflash_model::telemetry::EngramTelemetrySnapshot;

impl EngramLayerTelemetry {
    /// Records one batch of lookups with the tier they resolved in.
    pub fn record_lookups(&self, total: u64, tier1: u64, tier2: u64, tier3: u64, wait_ns: u64) {
        self.lookups_total.fetch_add(total, Ordering::Relaxed);
        self.tier1_drm_hits.fetch_add(tier1, Ordering::Relaxed);
        self.tier2_uma_hits.fetch_add(tier2, Ordering::Relaxed);
        self.tier3_nvme_misses.fetch_add(tier3, Ordering::Relaxed);
        self.prefetch_wait_ns_total
            .fetch_add(wait_ns, Ordering::Relaxed);
    }

    /// Records the mean gate alpha of one token batch (in [0, 1]) and the batch size.
    pub fn record_gate_alpha(&self, alpha: f64, samples: u64) {
        let scaled = (alpha.clamp(0.0, 1.0) * GATE_ALPHA_SCALE as f64).round() as u64;
        self.gate_alpha_sum_micro
            .fetch_add(scaled * samples, Ordering::Relaxed);
        self.gate_samples_total
            .fetch_add(samples, Ordering::Relaxed);
    }

    /// Records image tokens that were masked out of Engram lookup.
    pub fn record_image_tokens_masked(&self, count: u64) {
        self.image_tokens_masked.fetch_add(count, Ordering::Relaxed);
    }

    /// Snapshots the counters for one layer id.
    #[must_use]
    pub fn snapshot(&self, layer_id: usize) -> EngramTelemetrySnapshot {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let lookups = load(&self.lookups_total);
        let hits = load(&self.tier1_drm_hits) + load(&self.tier2_uma_hits);
        let gate_samples = load(&self.gate_samples_total);
        EngramTelemetrySnapshot {
            layer_id,
            lookups_total: lookups,
            tier1_drm_hits: load(&self.tier1_drm_hits),
            tier2_uma_hits: load(&self.tier2_uma_hits),
            tier3_nvme_misses: load(&self.tier3_nvme_misses),
            hit_rate: if lookups == 0 {
                0.0
            } else {
                hits as f64 / lookups as f64
            },
            mean_prefetch_stall_us: if lookups == 0 {
                0.0
            } else {
                load(&self.prefetch_wait_ns_total) as f64 / lookups as f64 / 1000.0
            },
            mean_gate_alpha: if gate_samples == 0 {
                0.0
            } else {
                load(&self.gate_alpha_sum_micro) as f64
                    / gate_samples as f64
                    / GATE_ALPHA_SCALE as f64
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_telemetry_has_zero_rates() {
        let telemetry = EngramLayerTelemetry::default();
        let snapshot = telemetry.snapshot(1);
        assert_eq!(snapshot.layer_id, 1);
        assert_eq!(snapshot.hit_rate, 0.0);
        assert_eq!(snapshot.mean_gate_alpha, 0.0);
        assert_eq!(snapshot.mean_prefetch_stall_us, 0.0);
    }

    #[test]
    fn hit_rate_and_stall_average_are_exact() {
        let telemetry = EngramLayerTelemetry::default();
        telemetry.record_lookups(26880, 7120, 12890, 6870, 26_880_000);
        telemetry.record_gate_alpha(0.312, 1000);
        telemetry.record_gate_alpha(0.288, 1000);
        telemetry.record_image_tokens_masked(64);
        let snapshot = telemetry.snapshot(14);
        assert_eq!(snapshot.lookups_total, 26880);
        assert_eq!(snapshot.tier3_nvme_misses, 6870);
        assert!(
            (snapshot.hit_rate - 0.74442).abs() < 1e-4,
            "{}",
            snapshot.hit_rate
        );
        // 26_880_000 ns of total stall over 26_880 lookups = 1000 ns = 1 us per lookup.
        assert!((snapshot.mean_prefetch_stall_us - 1.0).abs() < 1e-9);
        assert!((snapshot.mean_gate_alpha - 0.300).abs() < 1e-6);
        assert_eq!(telemetry.image_tokens_masked.load(Ordering::Relaxed), 64);
    }

    #[test]
    fn snapshot_is_serializable_in_the_documented_shape() {
        let telemetry = EngramLayerTelemetry::default();
        telemetry.record_lookups(100, 40, 30, 30, 0);
        let json = serde_json::to_value(telemetry.snapshot(1)).unwrap();
        for key in [
            "layer_id",
            "lookups_total",
            "tier1_drm_hits",
            "tier2_uma_hits",
            "tier3_nvme_misses",
            "hit_rate",
            "mean_prefetch_stall_us",
            "mean_gate_alpha",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(json["hit_rate"], 0.7);
    }
}
