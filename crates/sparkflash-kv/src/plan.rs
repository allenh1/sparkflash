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

//! Prefix-match planning and the SWA Bounded Replay execution plan.
//!
//! This is the weight-free core of the KV manager: given how many prompt tokens were found
//! in each cache tier, it decides exactly how many tokens must be replayed through the
//! encoder (to rebuild sliding-window KV) and through the decoder (Decoder SWA Bounded
//! Replay, see arXiv:2609.19969 section 3.2.2), and records the decision in telemetry.
//!
//! D7 (docs/DECISIONS.md) splits the 40-layer backbone after layer 20: the encoder layers
//! 0..=20 run over the whole prompt and the decoder layers 21..=39 only over the last
//! min(prompt, SWA_WINDOW_SIZE) tokens. A plan counts tokens, not layers, so the layer
//! boundary itself is owned by crate::config (ENCODER_LAST_LAYER, DECODER_FIRST_LAYER) and
//! shared with the CED schedule and swa::ReplayKind; the cross-check test pins all three
//! against each other.

pub use sparkflash_model::telemetry::{KvCacheStatsSnapshot, KvCacheTelemetry, PrefixMatchPlan};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SWA_WINDOW_SIZE;

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
    fn full_prefix_hit_with_swa_tail_skips_encoder_replay_but_not_the_decoder_window() {
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
