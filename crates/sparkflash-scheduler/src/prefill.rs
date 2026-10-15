//! Milestone 5 companions: the CED prefill bypass, sequence-parallel chunked prefill and
//! the L2 weight-prefetch plan (spec 6.2 line 1340, decisions D7 and D5).
//!
//! These are planners: they decide how many tokens traverse which layers, which collective
//! carries a sequence-parallel split, and how much of the next sub-layer's weights the
//! side-stream warner can touch inside a measured RoCEnante wait window. The GPU execution is
//! Milestone 4/6 work, so what is testable here is exactly the decision.

use crate::config::{CED_ENCODER_LAYERS, CED_SPLIT_LAYER, L2_CACHE_LINE_BYTES, SchedulerConfig};
use crate::error::SchedulerError;
use crate::transport::GatherShape;

/// Which collective path serves a payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveBackend {
    /// The RoCEnante one-shot path, for payloads that fit one arena slot.
    RocEnante,
    /// The patched NCCL path, for payloads above one arena slot.
    Nccl,
}

impl CollectiveBackend {
    /// The trace scope this backend's work belongs to.
    #[must_use]
    pub const fn trace_scope(self) -> u64 {
        match self {
            Self::RocEnante => sparkflash_trace::TraceScope::ROCENANTE_RDMA,
            Self::Nccl => sparkflash_trace::TraceScope::NCCL_COLLECTIVE,
        }
    }

    /// Chooses the backend for a payload against the one-shot slot size.
    #[must_use]
    pub const fn for_payload(bytes: usize, slot_bytes: usize) -> Self {
        if bytes > slot_bytes {
            Self::Nccl
        } else {
            Self::RocEnante
        }
    }
}

/// The 20-layer causal-encoder prefill bypass (spec 6.2, C_20 = H_19 W_20^KV; decision D7).
///
/// Layers 0..=20 run over the whole prompt; layers 21..=39 run only over the last sliding
/// window. Layer 20 is the only layer that projects a KV for the decoder, and D7 records
/// that it needs no tensor the checkpoint does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CedPrefillPlan {
    /// Prompt tokens the request brought.
    pub prompt_tokens: usize,
    /// Layers that run over the whole prompt.
    pub encoder_layers: usize,
    /// Layers that run over the tail only.
    pub decoder_layers: usize,
    /// Tail tokens the decoder layers see.
    pub decoder_tokens: usize,
    /// Tokens layer 20 projects the decoder KV from.
    pub kv_projection_tokens: usize,
    /// Layer that owns the projection.
    pub projection_layer: usize,
    /// Layer-token slots the full-depth plan would have run.
    pub full_depth_layer_tokens: u64,
    /// Layer-token slots the bypassed plan runs.
    pub bypass_layer_tokens: u64,
}

impl CedPrefillPlan {
    /// Plans the encoder/decoder split for one prompt.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for an empty prompt.
    pub fn plan(prompt_tokens: usize) -> Result<Self, SchedulerError> {
        if prompt_tokens == 0 {
            return Err(SchedulerError::InvalidArgument(
                "the CED split needs at least one prompt token".to_owned(),
            ));
        }
        let window = sparkflash_kv::SWA_WINDOW_SIZE;
        let decoder_tokens = prompt_tokens.min(window);
        let encoder_layer_tokens = (CED_ENCODER_LAYERS as u64) * (prompt_tokens as u64);
        let full_depth_layer_tokens =
            (sparkflash_kv::NUM_HIDDEN_LAYERS as u64) * (prompt_tokens as u64);
        Ok(Self {
            prompt_tokens,
            encoder_layers: CED_ENCODER_LAYERS,
            decoder_layers: sparkflash_kv::NUM_HIDDEN_LAYERS - CED_ENCODER_LAYERS,
            decoder_tokens,
            kv_projection_tokens: prompt_tokens,
            projection_layer: CED_SPLIT_LAYER,
            full_depth_layer_tokens,
            bypass_layer_tokens: encoder_layer_tokens
                + ((sparkflash_kv::NUM_HIDDEN_LAYERS - CED_ENCODER_LAYERS) as u64)
                    * (decoder_tokens as u64),
        })
    }

    /// Layer-token slots the bypass saves.
    #[must_use]
    pub const fn saved_layer_tokens(&self) -> u64 {
        self.full_depth_layer_tokens - self.bypass_layer_tokens
    }

    /// Fraction of the full-depth work the bypass removes.
    #[must_use]
    pub fn saved_fraction(&self) -> f64 {
        self.saved_layer_tokens() as f64 / self.full_depth_layer_tokens as f64
    }

    /// True when the prompt is longer than the sliding window, i.e. when there is anything to
    /// bypass at all.
    #[must_use]
    pub const fn bypasses_the_decoder(&self) -> bool {
        self.prompt_tokens > self.decoder_tokens
    }

    /// The trace scope this plan's work belongs to.
    #[must_use]
    pub const fn trace_scope(&self) -> u64 {
        sparkflash_trace::TraceScope::CSA2_ATTN
    }
}

/// The sequence-parallel chunked prefill plan (`DSV41_PREFILL_SP`, spec 6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqParallelPlan {
    /// Rows this rank holds before the reduce-scatter.
    pub rows: usize,
    /// Bytes one row of partial attention output carries.
    pub bytes_per_row: usize,
    /// Payload bytes the collective moves.
    pub partial_bytes: usize,
    /// Backend the payload size selects.
    pub backend: CollectiveBackend,
    /// Shape of the reduce-scatter.
    pub reduce_scatter: GatherShape,
    /// Shape of the all-gather.
    pub all_gather: GatherShape,
}

impl SeqParallelPlan {
    /// Plans a sequence-parallel split.
    ///
    /// The backend is a decision, not a constant: spec 6.2 routes payloads above the 2 MiB
    /// one-shot arena slot to the patched NCCL path, and this plan applies exactly that rule.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for zero rows or zero row width.
    pub fn plan(
        config: &SchedulerConfig,
        rows: usize,
        bytes_per_row: usize,
    ) -> Result<Self, SchedulerError> {
        config.validate()?;
        if rows == 0 || bytes_per_row == 0 {
            return Err(SchedulerError::InvalidArgument(
                "a sequence-parallel split needs rows and a row width".to_owned(),
            ));
        }
        let partial_bytes = rows.checked_mul(bytes_per_row).ok_or_else(|| {
            SchedulerError::InvalidArgument("the sequence-parallel payload overflows".to_owned())
        })?;
        let shape = GatherShape::new(rows, bytes_per_row);
        Ok(Self {
            rows,
            bytes_per_row,
            partial_bytes,
            backend: CollectiveBackend::for_payload(partial_bytes, config.slot_bytes),
            reduce_scatter: shape,
            all_gather: shape,
        })
    }

    /// The trace scope this plan's collective belongs to.
    #[must_use]
    pub const fn trace_scope(&self) -> u64 {
        self.backend.trace_scope()
    }
}

/// The simultaneous L2 weight-prefetch plan (`DSV41_L2_PREFETCH`, spec 6.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L2PrefetchPlan {
    /// Bytes of the next sub-layer's weights the warner touches.
    pub window_bytes: usize,
    /// Measured wait window the plan was derived from.
    pub wait_window_us: f64,
    /// Bandwidth the derivation used, in bytes per microsecond.
    pub bandwidth_bytes_per_us: u64,
    /// Backend whose wait window this plan overlaps.
    pub backend: CollectiveBackend,
    /// Engram rows the plan recommends issuing ahead.
    pub engram_lead_depth_rows: usize,
    /// The configured L2 prefetch window this plan was capped against
    /// (`SchedulerConfig::l2_prefetch_bytes`).
    pub configured_window_bytes: usize,
}

impl L2PrefetchPlan {
    /// Plans the side-stream prefetch for a collective.
    ///
    /// Returns `None` for the NCCL backend: the spec's warner overlaps the RoCEnante
    /// wait window, and a payload that large already has its own overlap schedule.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for a non-positive or non-finite wait
    /// window, or a non-positive bandwidth.
    pub fn plan(
        config: &SchedulerConfig,
        backend: CollectiveBackend,
        wait_window_us: f64,
        bandwidth_bytes_per_us: u64,
    ) -> Result<Option<Self>, SchedulerError> {
        config.validate()?;
        if !wait_window_us.is_finite() || wait_window_us <= 0.0 {
            return Err(SchedulerError::InvalidArgument(format!(
                "the wait window must be a positive number of microseconds, got {wait_window_us}"
            )));
        }
        if bandwidth_bytes_per_us == 0 {
            return Err(SchedulerError::InvalidArgument(
                "the prefetch bandwidth must be positive".to_owned(),
            ));
        }
        if backend == CollectiveBackend::Nccl {
            return Ok(None);
        }
        let budget = (wait_window_us * bandwidth_bytes_per_us as f64) as usize;
        let window_bytes = config
            .l2_prefetch_bytes
            .min(budget)
            .div_euclid(L2_CACHE_LINE_BYTES)
            * L2_CACHE_LINE_BYTES;
        if window_bytes == 0 {
            return Err(SchedulerError::InvalidArgument(format!(
                "a {wait_window_us} us wait window cannot cover one {L2_CACHE_LINE_BYTES}-byte cache line"
            )));
        }
        let lead_depth_rows =
            ((wait_window_us / config.assumed_engram_row_latency_us).ceil() as usize).max(1);
        Ok(Some(Self {
            window_bytes,
            wait_window_us,
            bandwidth_bytes_per_us,
            backend,
            engram_lead_depth_rows: lead_depth_rows,
            configured_window_bytes: config.l2_prefetch_bytes,
        }))
    }

    /// True when the plan covers the whole configured L2 window.
    ///
    /// The planned window is `min(configured, wait_window * bandwidth)` rounded down to a
    /// cache line, so the comparison is against the *coverable* configuration - the configured
    /// value rounded down to a cache line, at least one line - rather than the raw value.
    /// Comparing against the raw value made the answer permanently false for any configuration
    /// that is not a cache-line multiple (review M5 A2).
    #[must_use]
    pub const fn saturates_the_window(&self) -> bool {
        self.window_bytes >= coverable_window(self.configured_window_bytes)
    }

    /// The trace scope this plan's work belongs to.
    #[must_use]
    pub const fn trace_scope(&self) -> u64 {
        sparkflash_trace::TraceScope::L2_PREFETCH
    }
}

/// The most a plan can cover of `configured`: rounded down to a cache line, at least one line,
/// because the planned window is always cache-line aligned and a plan is never empty.
const fn coverable_window(configured: usize) -> usize {
    let lines = configured / L2_CACHE_LINE_BYTES;
    if lines == 0 {
        L2_CACHE_LINE_BYTES
    } else {
        lines * L2_CACHE_LINE_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LPDDR5X_BYTES_PER_US, PREFILL_REFERENCE_PROMPT_TOKENS, SchedulerConfig};
    use sparkflash_kv::{Csa2Mode, csa2_mode};

    #[test]
    fn the_ced_split_derives_its_layer_counts_from_the_backbone() {
        let plan = CedPrefillPlan::plan(PREFILL_REFERENCE_PROMPT_TOKENS).expect("plan");
        assert_eq!(
            plan.encoder_layers + plan.decoder_layers,
            sparkflash_kv::NUM_HIDDEN_LAYERS
        );
        assert_eq!(plan.encoder_layers, CED_SPLIT_LAYER + 1);
        assert_eq!(plan.decoder_tokens, sparkflash_kv::SWA_WINDOW_SIZE);
        assert_eq!(plan.kv_projection_tokens, plan.prompt_tokens);
        assert_eq!(plan.projection_layer, CED_SPLIT_LAYER);
        assert!(plan.bypasses_the_decoder());
        assert_eq!(plan.full_depth_layer_tokens, 40 * 400_000);
        assert_eq!(plan.bypass_layer_tokens, 21 * 400_000 + 19 * 128);
        assert_eq!(plan.saved_layer_tokens(), 7_597_568);
        let fraction = plan.saved_fraction();
        assert!(
            (0.45..=0.50).contains(&fraction),
            "the bypass should be about the spec's O(N L/2) saving, got {fraction}"
        );
    }

    #[test]
    fn only_layer_twenty_owns_the_decoder_kv_and_the_decoder_reuses_it() {
        assert_eq!(csa2_mode(CED_SPLIT_LAYER), Csa2Mode::Full);
        assert!(csa2_mode(CED_SPLIT_LAYER).owns_main_kv());
        for layer in (CED_SPLIT_LAYER + 1)..sparkflash_kv::NUM_HIDDEN_LAYERS {
            assert!(
                !csa2_mode(layer).owns_main_kv(),
                "layer {layer} must reuse the layer-20 KV, not project its own"
            );
        }
    }

    #[test]
    fn a_prompt_inside_the_window_has_nothing_to_bypass() {
        let plan = CedPrefillPlan::plan(64).expect("plan");
        assert_eq!(plan.decoder_tokens, 64);
        assert!(!plan.bypasses_the_decoder());
        assert_eq!(plan.saved_layer_tokens(), 0);
        assert_eq!(plan.saved_fraction(), 0.0);
        assert!(CedPrefillPlan::plan(0).is_err());
    }

    #[test]
    fn the_sequence_parallel_backend_is_chosen_by_payload_size() {
        let config = SchedulerConfig::default();
        let exact = SeqParallelPlan::plan(&config, 2048, 1024).expect("plan");
        assert_eq!(exact.partial_bytes, config.slot_bytes);
        assert_eq!(exact.backend, CollectiveBackend::RocEnante);
        assert_eq!(
            exact.trace_scope(),
            sparkflash_trace::TraceScope::ROCENANTE_RDMA
        );
        assert_eq!(exact.reduce_scatter, GatherShape::new(2048, 1024));
        assert_eq!(exact.all_gather, exact.reduce_scatter);

        let big = SeqParallelPlan::plan(&config, 8192, 1024).expect("plan");
        assert!(big.partial_bytes > config.slot_bytes);
        assert_eq!(big.backend, CollectiveBackend::Nccl);
        assert_eq!(
            big.trace_scope(),
            sparkflash_trace::TraceScope::NCCL_COLLECTIVE
        );
        assert!(SeqParallelPlan::plan(&config, 0, 1024).is_err());
        assert!(SeqParallelPlan::plan(&config, 8, 0).is_err());
        assert_eq!(
            CollectiveBackend::for_payload(config.slot_bytes, config.slot_bytes),
            CollectiveBackend::RocEnante
        );
        assert_eq!(
            CollectiveBackend::for_payload(config.slot_bytes + 1, config.slot_bytes),
            CollectiveBackend::Nccl
        );
    }

    #[test]
    fn the_l2_window_is_bounded_by_the_measured_wait_window() {
        let config = SchedulerConfig::default();
        let short = L2PrefetchPlan::plan(
            &config,
            CollectiveBackend::RocEnante,
            15.0,
            LPDDR5X_BYTES_PER_US,
        )
        .expect("plan")
        .expect("the RoCEnante path prefetches");
        assert!(short.window_bytes < config.l2_prefetch_bytes);
        assert!(
            !short.saturates_the_window(),
            "a 15 us window cannot cover the configured 6 MiB"
        );
        assert_eq!(short.window_bytes % L2_CACHE_LINE_BYTES, 0);
        let budget = (15.0 * LPDDR5X_BYTES_PER_US as f64) as usize;
        assert_eq!(short.window_bytes, budget.div_euclid(64) * 64);
        assert_eq!(short.engram_lead_depth_rows, 17);
        assert_eq!(
            short.trace_scope(),
            sparkflash_trace::TraceScope::L2_PREFETCH
        );
        assert_eq!(short.backend, CollectiveBackend::RocEnante);

        let long = L2PrefetchPlan::plan(
            &config,
            CollectiveBackend::RocEnante,
            28.0,
            LPDDR5X_BYTES_PER_US,
        )
        .expect("plan")
        .expect("the RoCEnante path prefetches");
        assert_eq!(long.window_bytes, config.l2_prefetch_bytes);
        assert!(
            long.saturates_the_window(),
            "the plan that was capped by the configuration saturates it"
        );
        assert_eq!(long.configured_window_bytes, config.l2_prefetch_bytes);
        assert!(long.window_bytes > short.window_bytes);

        assert!(
            L2PrefetchPlan::plan(&config, CollectiveBackend::Nccl, 20.0, LPDDR5X_BYTES_PER_US)
                .expect("no error")
                .is_none(),
            "the NCCL path does not use the RoCEnante wait window"
        );
        assert!(L2PrefetchPlan::plan(&config, CollectiveBackend::RocEnante, 0.0, 1).is_err());
        assert!(L2PrefetchPlan::plan(&config, CollectiveBackend::RocEnante, 20.0, 0).is_err());
        assert!(
            L2PrefetchPlan::plan(&config, CollectiveBackend::RocEnante, 0.0001, 1).is_err(),
            "a wait window under one cache line cannot prefetch anything"
        );
    }

    /// A configured window that is not a cache-line multiple must still be able to saturate: the
    /// planned window is rounded down to a cache line, so comparing it against the raw
    /// configuration made the answer permanently false (review M5 A2).
    #[test]
    fn an_unaligned_configured_window_can_still_saturate() {
        let config = SchedulerConfig {
            l2_prefetch_bytes: 100,
            ..SchedulerConfig::default()
        };
        let plan = L2PrefetchPlan::plan(
            &config,
            CollectiveBackend::RocEnante,
            28.0,
            LPDDR5X_BYTES_PER_US,
        )
        .expect("plan")
        .expect("the RoCEnante path prefetches");
        assert_eq!(
            plan.window_bytes, 64,
            "the window rounds down to one cache line"
        );
        assert_eq!(
            plan.configured_window_bytes, 100,
            "the configured value is reported unchanged"
        );
        assert!(
            plan.saturates_the_window(),
            "28 us covers the whole coverable 64 bytes of a 100-byte configuration"
        );
        // A configuration below one cache line has no coverable window and is refused rather
        // than reported as a plan that covers nothing.
        let tiny = SchedulerConfig {
            l2_prefetch_bytes: 1,
            ..SchedulerConfig::default()
        };
        assert!(
            L2PrefetchPlan::plan(
                &tiny,
                CollectiveBackend::RocEnante,
                28.0,
                LPDDR5X_BYTES_PER_US
            )
            .is_err(),
            "a configuration under one cache line cannot prefetch"
        );
    }
}
