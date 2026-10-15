//! Prefix-match planning and the SWA Bounded Replay execution plan.
//!
//! This is the weight-free core of the KV manager: given how many prompt tokens were found
//! in each cache tier, it decides exactly how many tokens must be replayed through the
//! encoder (to rebuild sliding-window KV) and through the decoder (Decoder SWA Bounded
//! Replay, see arXiv:2609.19969 section 3.2.2), and records the decision in telemetry.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sparkflash_trace::{TraceScope, spark_trace_span};

use crate::config::{KV_PAGE_TOKENS, SWA_WINDOW_SIZE};

/// Cumulative KV cache telemetry exported through /v1/cluster/stats and Prometheus.
#[derive(Debug, Default)]
pub struct KvCacheTelemetry {
    /// Prompts planned since process start.
    pub requests_total: AtomicU64,
    /// Prompt tokens those plans covered.
    pub prompt_tokens_total: AtomicU64,
    /// Matched tokens served from coherent UMA.
    pub global_kv_uma_hit_tokens: AtomicU64,
    /// Matched tokens restored from the NVMe persistent tier.
    pub global_kv_nvme_hit_tokens: AtomicU64,
    /// Prompt tokens found in no global KV tier and therefore encoded in full.
    pub global_kv_miss_tokens: AtomicU64,
    /// Plans whose 128-token SWA tail was still resident in the ring.
    pub swa_tail_cache_hits: AtomicU64,
    /// Plans whose cached prefix needed bounded replay to rebuild that tail.
    pub swa_tail_cache_misses: AtomicU64,
    /// Tokens replayed through the encoder layers to rebuild sliding-window KV.
    pub encoder_swa_replay_tokens_total: AtomicU64,
    /// Tokens replayed through the decoder layers for the prompt tail.
    pub decoder_swa_replay_tokens_total: AtomicU64,
    /// Pages currently allocated from the pool.
    pub active_kv_pages: AtomicU64,
    /// Size of the paged KV pool, as last reported by the allocator.
    pub total_kv_pages: AtomicU64,
}

/// Point-in-time snapshot of the KV cache telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvCacheStatsSnapshot {
    /// Prompts planned since process start.
    pub requests_total: u64,
    /// Prompt tokens those plans covered.
    pub prompt_tokens_total: u64,
    /// Matched tokens served from coherent UMA.
    pub global_kv_uma_hit_tokens: u64,
    /// Matched tokens restored from the NVMe persistent tier.
    pub global_kv_nvme_hit_tokens: u64,
    /// Prompt tokens found in no global KV tier and therefore encoded in full.
    pub global_kv_miss_tokens: u64,
    /// Plans whose 128-token SWA tail was still resident in the ring.
    pub swa_tail_cache_hits: u64,
    /// Plans whose cached prefix needed bounded replay to rebuild that tail.
    pub swa_tail_cache_misses: u64,
    /// Tokens replayed through the encoder layers to rebuild sliding-window KV.
    pub encoder_swa_replay_tokens_total: u64,
    /// Tokens replayed through the decoder layers for the prompt tail.
    pub decoder_swa_replay_tokens_total: u64,
    /// Pages currently allocated from the pool.
    pub active_kv_pages: u64,
    /// Size of the paged KV pool, as last reported by the allocator.
    pub total_kv_pages: u64,
    /// Fraction of prompt tokens served from any global KV tier.
    pub global_kv_hit_rate: f64,
}

/// Result of matching an incoming prompt against the radix prefix tree and the SWA tail ring.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefixMatchPlan {
    /// Length of the prompt this plan was computed for.
    pub total_prompt_tokens: usize,
    /// Prefix tokens matched in global KV, aligned down to a 64-token page boundary.
    pub global_kv_cached_tokens: usize,
    /// How much of that came from coherent UMA.
    pub uma_hit_tokens: usize,
    /// How much had to be restored from the NVMe persistent tier.
    pub nvme_restored_tokens: usize,
    /// Whether the exact 128-token SWA tail at `global_kv_cached_tokens` was in the ring.
    pub swa_tail_hit: bool,
    /// Tokens replayed through encoder layers 0..19 to rebuild sliding-window KV.
    pub encoder_swa_replay_tokens: usize,
    /// Uncached suffix that must run the full encoder.
    pub uncached_suffix_tokens: usize,
    /// Tokens at the end of the prompt that must run decoder layers 20..39.
    pub decoder_swa_replay_tokens: usize,
}

impl PrefixMatchPlan {
    /// Total tokens that execute encoder layers (replay + uncached suffix).
    #[must_use]
    pub fn encoder_total_tokens(&self) -> usize {
        self.encoder_swa_replay_tokens + self.uncached_suffix_tokens
    }

    /// Cache hit rate over the prompt, in [0, 1].
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        if self.total_prompt_tokens == 0 {
            0.0
        } else {
            self.global_kv_cached_tokens as f64 / self.total_prompt_tokens as f64
        }
    }
}

impl KvCacheTelemetry {
    /// Computes the exact CED + SWA Bounded Replay execution plan for a prompt.
    ///
    /// `uma_matched_tokens` and `nvme_matched_tokens` are the raw radix-tree matches per
    /// tier; the plan aligns the usable global prefix down to a 64-token page boundary so
    /// paged FlashMLA execution stays bit-identical.
    pub fn plan_request(
        &self,
        total_prompt_tokens: usize,
        uma_matched_tokens: usize,
        nvme_matched_tokens: usize,
        swa_tail_hit: bool,
    ) -> PrefixMatchPlan {
        spark_trace_span!(
            TraceScope::KV_RADIX,
            "kv_radix_plan_request",
            total_prompt_tokens as u32
        );
        // Saturating: a caller that reports more matched tokens than the prompt (or two large
        // tier counts) must not wrap into a tiny cache hit.
        let raw_cached = uma_matched_tokens
            .saturating_add(nvme_matched_tokens)
            .min(total_prompt_tokens.saturating_sub(1));
        let global_kv_cached_tokens = (raw_cached / KV_PAGE_TOKENS) * KV_PAGE_TOKENS;
        let uma_hit = uma_matched_tokens.min(global_kv_cached_tokens);
        let nvme_hit = global_kv_cached_tokens.saturating_sub(uma_hit);
        let uncached_suffix_tokens = total_prompt_tokens - global_kv_cached_tokens;

        // Encoder SWA Bounded Replay: the global KV hit is usable, but the 128-token
        // sliding window tail that lived at global_kv_cached_tokens was evicted from the
        // ring, so replay at most n_win tokens of the cached prefix through the encoder.
        let encoder_swa_replay_tokens = if global_kv_cached_tokens == 0 || swa_tail_hit {
            0
        } else {
            global_kv_cached_tokens.min(SWA_WINDOW_SIZE)
        };

        // Decoder SWA Bounded Replay: under CED the decoder never sees the whole prompt
        // during prefill, so only the last min(N, n_win) tokens run decoder layers.
        let decoder_swa_replay_tokens = if swa_tail_hit && uncached_suffix_tokens == 0 {
            0
        } else {
            total_prompt_tokens.min(SWA_WINDOW_SIZE)
        };

        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_total
            .fetch_add(total_prompt_tokens as u64, Ordering::Relaxed);
        self.global_kv_uma_hit_tokens
            .fetch_add(uma_hit as u64, Ordering::Relaxed);
        self.global_kv_nvme_hit_tokens
            .fetch_add(nvme_hit as u64, Ordering::Relaxed);
        self.global_kv_miss_tokens
            .fetch_add(uncached_suffix_tokens as u64, Ordering::Relaxed);
        if global_kv_cached_tokens > 0 {
            if swa_tail_hit {
                self.swa_tail_cache_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.swa_tail_cache_misses.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.encoder_swa_replay_tokens_total
            .fetch_add(encoder_swa_replay_tokens as u64, Ordering::Relaxed);
        self.decoder_swa_replay_tokens_total
            .fetch_add(decoder_swa_replay_tokens as u64, Ordering::Relaxed);

        PrefixMatchPlan {
            total_prompt_tokens,
            global_kv_cached_tokens,
            uma_hit_tokens: uma_hit,
            nvme_restored_tokens: nvme_hit,
            swa_tail_hit,
            encoder_swa_replay_tokens,
            uncached_suffix_tokens,
            decoder_swa_replay_tokens,
        }
    }

    /// Registers a page allocation (called by the paged cache allocator).
    pub fn note_page_allocated(&self) {
        self.active_kv_pages.fetch_add(1, Ordering::Relaxed);
    }

    /// Registers a page free.
    pub fn note_page_freed(&self) {
        self.active_kv_pages.fetch_sub(1, Ordering::Relaxed);
    }

    /// Registers the total page count of the pool.
    pub fn set_total_pages(&self, pages: u64) {
        self.total_kv_pages.store(pages, Ordering::Relaxed);
    }

    /// Snapshots the telemetry counters.
    #[must_use]
    pub fn snapshot(&self) -> KvCacheStatsSnapshot {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let prompt = load(&self.prompt_tokens_total);
        let hits = load(&self.global_kv_uma_hit_tokens) + load(&self.global_kv_nvme_hit_tokens);
        KvCacheStatsSnapshot {
            requests_total: load(&self.requests_total),
            prompt_tokens_total: prompt,
            global_kv_uma_hit_tokens: load(&self.global_kv_uma_hit_tokens),
            global_kv_nvme_hit_tokens: load(&self.global_kv_nvme_hit_tokens),
            global_kv_miss_tokens: load(&self.global_kv_miss_tokens),
            swa_tail_cache_hits: load(&self.swa_tail_cache_hits),
            swa_tail_cache_misses: load(&self.swa_tail_cache_misses),
            encoder_swa_replay_tokens_total: load(&self.encoder_swa_replay_tokens_total),
            decoder_swa_replay_tokens_total: load(&self.decoder_swa_replay_tokens_total),
            active_kv_pages: load(&self.active_kv_pages),
            total_kv_pages: load(&self.total_kv_pages),
            global_kv_hit_rate: if prompt == 0 {
                0.0
            } else {
                hits as f64 / prompt as f64
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_prompt_has_no_cache_hits_and_full_encoder_run() {
        let telemetry = KvCacheTelemetry::default();
        let plan = telemetry.plan_request(4096, 0, 0, false);
        assert_eq!(plan.global_kv_cached_tokens, 0);
        assert_eq!(plan.uncached_suffix_tokens, 4096);
        assert_eq!(plan.encoder_swa_replay_tokens, 0);
        assert_eq!(plan.decoder_swa_replay_tokens, SWA_WINDOW_SIZE);
        assert_eq!(plan.encoder_total_tokens(), 4096);
    }

    #[test]
    fn cached_prefix_is_page_aligned_down() {
        let telemetry = KvCacheTelemetry::default();
        let plan = telemetry.plan_request(1000, 900, 0, true);
        assert_eq!(plan.global_kv_cached_tokens, 896);
        assert_eq!(plan.uma_hit_tokens, 896);
        assert_eq!(plan.nvme_restored_tokens, 0);
        assert_eq!(plan.uncached_suffix_tokens, 104);
        assert_eq!(plan.hit_rate(), 0.896);
    }

    #[test]
    fn swa_tail_miss_triggers_encoder_bounded_replay() {
        let telemetry = KvCacheTelemetry::default();
        let plan = telemetry.plan_request(8192, 4096, 0, false);
        assert_eq!(plan.global_kv_cached_tokens, 4096);
        assert_eq!(plan.encoder_swa_replay_tokens, SWA_WINDOW_SIZE);
        assert_eq!(plan.decoder_swa_replay_tokens, SWA_WINDOW_SIZE);
        assert_eq!(plan.encoder_total_tokens(), 128 + 4096);
    }

    #[test]
    fn full_prefix_hit_with_swa_tail_skips_all_replay() {
        let telemetry = KvCacheTelemetry::default();
        let plan = telemetry.plan_request(4096, 4096, 0, true);
        // Never cache away the final token: the next token still has to be produced.
        assert_eq!(plan.global_kv_cached_tokens, 4032);
        assert_eq!(plan.encoder_swa_replay_tokens, 0);
        assert_eq!(plan.decoder_swa_replay_tokens, SWA_WINDOW_SIZE);
        // With a fully cached prefix the decoder only needs the final window.
        let plan2 = telemetry.plan_request(128, 127, 0, true);
        assert_eq!(plan2.global_kv_cached_tokens, 64);
        assert_eq!(plan2.encoder_swa_replay_tokens, 0);
    }

    #[test]
    fn nvme_hits_are_reported_separately_from_uma_hits() {
        let telemetry = KvCacheTelemetry::default();
        let plan = telemetry.plan_request(100_000, 1024, 65_536, false);
        assert_eq!(plan.global_kv_cached_tokens, 66_560);
        assert_eq!(plan.uma_hit_tokens, 1024);
        assert_eq!(plan.nvme_restored_tokens, 66_560 - 1024);
        assert_eq!(plan.uncached_suffix_tokens, 100_000 - 66_560);
    }

    #[test]
    fn telemetry_counters_track_requests() {
        let telemetry = KvCacheTelemetry::default();
        telemetry.set_total_pages(1024);
        telemetry.note_page_allocated();
        telemetry.note_page_allocated();
        telemetry.note_page_freed();
        let _ = telemetry.plan_request(2048, 2048, 0, true);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.requests_total, 1);
        assert_eq!(snapshot.prompt_tokens_total, 2048);
        assert_eq!(snapshot.active_kv_pages, 1);
        assert_eq!(snapshot.total_kv_pages, 1024);
        assert!(snapshot.global_kv_hit_rate > 0.9);
    }
}
