//! Scheduling configuration: the specification's numbers as named defaults.
//!
//! Every constant here is either taken from the specification (marked with its section) or
//! re-exported from the crate that owns it, so no plan in this crate restates a model number.
//! Plans derive their decisions from `SchedulerConfig` rather than from these constants
//! directly, which is what lets a test change the chunk size and see every downstream number
//! move.

use crate::error::SchedulerError;

/// The two prefill chunk sizes the specification allows (spec 5.6.1).
pub const PREFILL_CHUNK_CHOICES: [usize; 2] = [4096, 8192];

/// Tensor-parallel ranks the prefill query rows are partitioned across
/// (`SPARK_PREFILL_TP_SPLIT`, spec 5.6.1).
pub const PREFILL_TP_RANKS: usize = 4;

/// Rows scored per bounded indexer tile (spec 5.6.1).
pub const INDEXER_TILE_ROWS: usize = 256;

/// Static indexer scratchpad reservation (spec 5.6.1, "a static 2.0 GiB scratchpad").
pub const INDEXER_SCRATCHPAD_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Bytes one score occupies in the bounded tile (the indexer scores in f32).
pub const INDEXER_SCORE_BYTES: usize = 4;

/// Bytes one running Top-k key occupies: an INT32 index plus its f32 score.
pub const INDEXER_TOPK_KEY_BYTES: usize = 8;

/// Bytes per entry of the all-at-once tensor the specification rejects as ~60 GiB: an f32
/// logit, an f32 score, an INT32 index and an INT32 staging slot.
pub const NAIVE_SCORE_BYTES: usize = 16;

/// Largest sequence the engine is configured for (`SPARK_MAX_SEQ_LEN`, decision D5).
pub const MAX_CONTEXT_TOKENS: usize = 1_048_576;

/// Short-request prefill bypass threshold in prompt tokens
/// (`DSV41_PREFILL_BYPASS = 256`, spec 5.6.2).
pub const PREFILL_BYPASS_MAX_TOKENS: usize = 256;

/// Decode steps that run between two prefill chunks (`--prefill-decode-interval 2`,
/// spec 5.6.2).
pub const PREFILL_DECODE_INTERVAL: usize = 2;

/// DSpark draft block size k (spec 5.6.3).
pub const DSPARK_BLOCK_SIZE: usize = 5;

/// DSpark layers: the three prediction blocks the drafter runs (spec 5.6.3).
pub const DSPARK_LAYERS: [usize; 3] = [37, 38, 39];

/// DSpark confidence cap: the smallest survival probability a draft position must reach to
/// be verified (spec 5.6.3).
pub const DSPARK_CONFIDENCE_CAP: f64 = 0.10;

/// DSpark Markov-head noise token id (`dspark_noise_token_id`, spec 5.6.3).
pub const DSPARK_NOISE_TOKEN_ID: u32 = 128_799;

/// Top-k experts the MoE router selects per token (spec 5.6.3, "the anchor token's Top-6
/// experts").
pub const ROUTER_TOP_K: usize = 6;

/// Verify rows per request in the CUDA graph: the anchor plus the k draft positions
/// (spec 5.6.3, "M = 6 verify rows per request").
pub const DSPARK_VERIFY_ROWS: usize = DSPARK_BLOCK_SIZE + 1;

/// Layer that projects the decoder KV from the encoder's final hidden state
/// (spec 6.2, C_20 = H_19 W_20^KV; decision D7).
pub const CED_SPLIT_LAYER: usize = 20;

/// Encoder layers that run over the whole prompt: layers 0..=20 (decision D7).
pub const CED_ENCODER_LAYERS: usize = CED_SPLIT_LAYER + 1;

/// L2 prefetch window the side-stream warner aims for (spec 6.2, "the first 6.0 MiB of the
/// next sub-layer's weight matrix").
pub const L2_PREFETCH_WINDOW_BYTES: usize = 6 * 1024 * 1024;

/// Cache line the L2 prefetch window is rounded down to.
pub const L2_CACHE_LINE_BYTES: usize = 64;

/// LPDDR5X bandwidth one DGX Spark feeds the side-stream warner with, in bytes per
/// microsecond (273 GB/s, the GB10's memory bandwidth).
pub const LPDDR5X_BYTES_PER_US: u64 = 273_000;

/// Cold-prefill time of the reference 400K prefill the specification quotes (spec 5.6.2,
/// "~78 seconds total").
pub const PREFILL_REFERENCE_SECONDS: f64 = 78.0;

/// Prompt length of that reference prefill (spec 5.6.2, "a 400K-1M token cold prefill").
pub const PREFILL_REFERENCE_PROMPT_TOKENS: usize = 400_000;

/// Decode-step cost the cost model assumes, in seconds.
///
/// This is the one number in the cost model the specification does not state: it gives
/// 0.55 ms as the *saving* from removing the per-step draft broadcast, not the step cost, so
/// the default below is an explicit assumption (55 ms) and every test that reports seconds
/// says so.
pub const ASSUMED_DECODE_STEP_SECONDS: f64 = 0.055;

/// Engram row latency the L2 prefetch lead-depth derivation assumes, in microseconds.
///
/// Also an assumption by construction: the lead depth is derived from the measured wait
/// window divided by this latency, and the default is a tunable placeholder, not a
/// specification number.
pub const ASSUMED_ENGRAM_ROW_LATENCY_US: f64 = 0.9;

/// One-shot RDMA arena slot: the largest payload the RoCEnante path carries in one shot.
pub use sparkflash_comm::ROCENANTE_MAX_BYTES;

/// Everything a scheduler plan needs in order to decide.
///
/// The struct is plain data with a validating constructor, so a test can start from
/// `SchedulerConfig::default()` and change exactly one knob to see which decision moves.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerConfig {
    /// Prefill chunk size in tokens; the specification allows 4096 or 8192.
    pub chunk_tokens: usize,
    /// Tensor-parallel ranks the chunk's query rows are split across.
    pub tp_ranks: usize,
    /// Query rows scored per bounded indexer tile.
    pub tile_rows: usize,
    /// Static indexer scratchpad the tiling must stay inside.
    pub scratchpad_bytes: usize,
    /// Context length the indexer plans for.
    pub max_context_tokens: usize,
    /// Prompt length at or below which a request bypasses a running prefill.
    pub bypass_max_tokens: usize,
    /// Decode steps between two prefill chunks.
    pub prefill_decode_interval: usize,
    /// Top-k positions the sparse indexer keeps (owned by `sparkflash-kv`).
    pub top_k: usize,
    /// Candidate blocks the hierarchical indexer builds (owned by `sparkflash-kv`).
    pub candidate_blocks: usize,
    /// Positions per candidate block (owned by `sparkflash-kv`).
    pub candidate_block_size: usize,
    /// Bytes one collective may carry in a single one-shot slot.
    pub slot_bytes: usize,
    /// DSpark draft positions per decode step.
    pub dspark_block_size: usize,
    /// DSpark survival-probability cap.
    pub dspark_confidence_cap: f64,
    /// DSpark Markov-head noise token id.
    pub dspark_noise_token_id: u32,
    /// L2 prefetch window the warner aims for.
    pub l2_prefetch_bytes: usize,
    /// Assumed decode-step cost, used only by the cost model.
    pub assumed_decode_step_seconds: f64,
    /// Assumed Engram row latency, used only by the lead-depth derivation.
    pub assumed_engram_row_latency_us: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            chunk_tokens: PREFILL_CHUNK_CHOICES[0],
            tp_ranks: PREFILL_TP_RANKS,
            tile_rows: INDEXER_TILE_ROWS,
            scratchpad_bytes: INDEXER_SCRATCHPAD_BYTES,
            max_context_tokens: MAX_CONTEXT_TOKENS,
            bypass_max_tokens: PREFILL_BYPASS_MAX_TOKENS,
            prefill_decode_interval: PREFILL_DECODE_INTERVAL,
            top_k: sparkflash_kv::INDEX_TOPK,
            candidate_blocks: sparkflash_kv::CANDIDATE_TOPK_BLOCKS,
            candidate_block_size: sparkflash_kv::CANDIDATE_BLOCK_SIZE,
            slot_bytes: ROCENANTE_MAX_BYTES,
            dspark_block_size: DSPARK_BLOCK_SIZE,
            dspark_confidence_cap: DSPARK_CONFIDENCE_CAP,
            dspark_noise_token_id: DSPARK_NOISE_TOKEN_ID,
            l2_prefetch_bytes: L2_PREFETCH_WINDOW_BYTES,
            assumed_decode_step_seconds: ASSUMED_DECODE_STEP_SECONDS,
            assumed_engram_row_latency_us: ASSUMED_ENGRAM_ROW_LATENCY_US,
        }
    }
}

impl SchedulerConfig {
    /// Checks every knob and returns the first one that is out of range.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidConfig` naming the offending knob and its value.
    pub fn validate(&self) -> Result<(), SchedulerError> {
        if !PREFILL_CHUNK_CHOICES.contains(&self.chunk_tokens) {
            return Err(SchedulerError::InvalidConfig(format!(
                "chunk_tokens {} is not one of {:?} (spec 5.6.1)",
                self.chunk_tokens, PREFILL_CHUNK_CHOICES
            )));
        }
        if self.tp_ranks == 0 {
            return Err(SchedulerError::InvalidConfig(
                "tp_ranks must be at least 1".to_owned(),
            ));
        }
        if !self.chunk_tokens.is_multiple_of(self.tp_ranks) {
            return Err(SchedulerError::InvalidConfig(format!(
                "chunk_tokens {} does not divide across {} ranks",
                self.chunk_tokens, self.tp_ranks
            )));
        }
        if self.tile_rows == 0 {
            return Err(SchedulerError::InvalidConfig(
                "tile_rows must be at least 1".to_owned(),
            ));
        }
        if self.scratchpad_bytes == 0 {
            return Err(SchedulerError::InvalidConfig(
                "scratchpad_bytes must be at least 1".to_owned(),
            ));
        }
        if self.max_context_tokens == 0 {
            return Err(SchedulerError::InvalidConfig(
                "max_context_tokens must be at least 1".to_owned(),
            ));
        }
        if self.max_context_tokens > MAX_CONTEXT_TOKENS {
            return Err(SchedulerError::InvalidConfig(format!(
                "max_context_tokens {} exceeds the model limit of {MAX_CONTEXT_TOKENS}",
                self.max_context_tokens
            )));
        }
        if self.bypass_max_tokens == 0 {
            return Err(SchedulerError::InvalidConfig(
                "bypass_max_tokens must be at least 1".to_owned(),
            ));
        }
        if self.prefill_decode_interval == 0 {
            return Err(SchedulerError::InvalidConfig(
                "prefill_decode_interval must be at least 1, otherwise a bypassed request has no inter-chunk decode step to join".to_owned(),
            ));
        }
        if self.top_k == 0 {
            return Err(SchedulerError::InvalidConfig(
                "top_k must be at least 1".to_owned(),
            ));
        }
        if self.candidate_blocks == 0 || self.candidate_block_size == 0 {
            return Err(SchedulerError::InvalidConfig(
                "candidate_blocks and candidate_block_size must be at least 1".to_owned(),
            ));
        }
        if self.slot_bytes == 0 {
            return Err(SchedulerError::InvalidConfig(
                "slot_bytes must be at least 1".to_owned(),
            ));
        }
        if self.dspark_block_size == 0 || self.dspark_block_size > 8 {
            return Err(SchedulerError::InvalidConfig(format!(
                "dspark_block_size {} must be in 1..=8 because the verification mask is one byte",
                self.dspark_block_size
            )));
        }
        if !(self.dspark_confidence_cap > 0.0 && self.dspark_confidence_cap < 1.0) {
            return Err(SchedulerError::InvalidConfig(format!(
                "dspark_confidence_cap {} must be in (0, 1)",
                self.dspark_confidence_cap
            )));
        }
        if self.l2_prefetch_bytes == 0 {
            return Err(SchedulerError::InvalidConfig(
                "l2_prefetch_bytes must be at least 1".to_owned(),
            ));
        }
        if self.assumed_decode_step_seconds <= 0.0 {
            return Err(SchedulerError::InvalidConfig(
                "assumed_decode_step_seconds must be positive".to_owned(),
            ));
        }
        if self.assumed_engram_row_latency_us <= 0.0 {
            return Err(SchedulerError::InvalidConfig(
                "assumed_engram_row_latency_us must be positive".to_owned(),
            ));
        }
        Ok(())
    }

    /// Query rows one TP rank owns for a chunk of `chunk_tokens`.
    ///
    /// # Panics
    /// Panics when the configuration was never validated; every plan validates first.
    #[must_use]
    pub fn rows_per_rank(&self) -> usize {
        assert!(
            self.chunk_tokens.is_multiple_of(self.tp_ranks),
            "configuration must be validated before use"
        );
        self.chunk_tokens / self.tp_ranks
    }

    /// Candidate positions the hierarchical indexer scores (2048 blocks x 8 positions).
    #[must_use]
    pub fn candidate_positions(&self) -> usize {
        self.candidate_blocks * self.candidate_block_size
    }

    /// Verify rows per request: the anchor plus the draft block.
    #[must_use]
    pub fn dspark_verify_rows(&self) -> usize {
        self.dspark_block_size + 1
    }

    /// Same configuration with a different chunk size.
    #[must_use]
    pub fn with_chunk_tokens(mut self, chunk_tokens: usize) -> Self {
        self.chunk_tokens = chunk_tokens;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_config_carries_the_spec_numbers_and_borrows_the_rest() {
        let config = SchedulerConfig::default();
        config
            .validate()
            .expect("the default configuration is valid");
        assert_eq!(config.chunk_tokens, 4096);
        assert_eq!(config.tp_ranks, PREFILL_TP_RANKS);
        assert_eq!(config.rows_per_rank(), 1024);
        assert_eq!(config.tile_rows, 256);
        assert_eq!(config.scratchpad_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(config.max_context_tokens, 1_048_576);
        assert_eq!(config.bypass_max_tokens, 256);
        assert_eq!(config.prefill_decode_interval, 2);
        assert_eq!(config.dspark_block_size, 5);
        assert_eq!(config.dspark_confidence_cap, 0.10);
        assert_eq!(config.dspark_noise_token_id, 128_799);
        assert_eq!(config.dspark_verify_rows(), DSPARK_VERIFY_ROWS);
        assert_eq!(config.dspark_verify_rows(), 6);
        assert_eq!(config.l2_prefetch_bytes, 6 * 1024 * 1024);
        // Borrowed, not restated: the indexer and candidate geometry belong to sparkflash-kv.
        assert_eq!(config.top_k, sparkflash_kv::INDEX_TOPK);
        assert_eq!(config.top_k, 512);
        assert_eq!(
            config.candidate_blocks,
            sparkflash_kv::CANDIDATE_TOPK_BLOCKS
        );
        assert_eq!(config.candidate_blocks, 2048);
        assert_eq!(
            config.candidate_block_size,
            sparkflash_kv::CANDIDATE_BLOCK_SIZE
        );
        assert_eq!(config.candidate_positions(), 16_384);
        assert_eq!(config.slot_bytes, sparkflash_comm::ROCENANTE_MAX_BYTES);
        assert_eq!(config.slot_bytes, 2 * 1024 * 1024);
    }

    #[test]
    fn both_chunk_sizes_the_spec_allows_validate_and_split_differently() {
        let mut sizes = Vec::new();
        for chunk in PREFILL_CHUNK_CHOICES {
            let config = SchedulerConfig::default().with_chunk_tokens(chunk);
            config.validate().expect("both spec chunk sizes are valid");
            sizes.push(config.rows_per_rank());
        }
        assert_eq!(sizes, vec![1024, 2048]);
    }

    #[test]
    fn validation_names_each_knob_it_rejects() {
        type Case = (&'static str, Box<dyn Fn(&mut SchedulerConfig)>);
        let cases: Vec<Case> = vec![
            (
                "chunk_tokens",
                Box::new(|c: &mut SchedulerConfig| c.chunk_tokens = 5000),
            ),
            (
                "chunk_tokens",
                Box::new(|c: &mut SchedulerConfig| c.chunk_tokens = 0),
            ),
            (
                "tp_ranks",
                Box::new(|c: &mut SchedulerConfig| c.tp_ranks = 0),
            ),
            (
                "does not divide",
                Box::new(|c: &mut SchedulerConfig| c.tp_ranks = 3),
            ),
            (
                "tile_rows",
                Box::new(|c: &mut SchedulerConfig| c.tile_rows = 0),
            ),
            (
                "scratchpad_bytes",
                Box::new(|c: &mut SchedulerConfig| c.scratchpad_bytes = 0),
            ),
            (
                "must be at least 1",
                Box::new(|c: &mut SchedulerConfig| c.max_context_tokens = 0),
            ),
            (
                "model limit",
                Box::new(|c: &mut SchedulerConfig| c.max_context_tokens = 2_097_152),
            ),
            (
                "bypass_max_tokens",
                Box::new(|c: &mut SchedulerConfig| c.bypass_max_tokens = 0),
            ),
            (
                "no inter-chunk decode step",
                Box::new(|c: &mut SchedulerConfig| c.prefill_decode_interval = 0),
            ),
            ("top_k", Box::new(|c: &mut SchedulerConfig| c.top_k = 0)),
            (
                "candidate_blocks",
                Box::new(|c: &mut SchedulerConfig| c.candidate_blocks = 0),
            ),
            (
                "candidate_blocks",
                Box::new(|c: &mut SchedulerConfig| c.candidate_block_size = 0),
            ),
            (
                "slot_bytes",
                Box::new(|c: &mut SchedulerConfig| c.slot_bytes = 0),
            ),
            (
                "dspark_block_size",
                Box::new(|c: &mut SchedulerConfig| c.dspark_block_size = 0),
            ),
            (
                "one byte",
                Box::new(|c: &mut SchedulerConfig| c.dspark_block_size = 9),
            ),
            (
                "dspark_confidence_cap",
                Box::new(|c: &mut SchedulerConfig| c.dspark_confidence_cap = 0.0),
            ),
            (
                "dspark_confidence_cap",
                Box::new(|c: &mut SchedulerConfig| c.dspark_confidence_cap = 1.0),
            ),
            (
                "l2_prefetch_bytes",
                Box::new(|c: &mut SchedulerConfig| c.l2_prefetch_bytes = 0),
            ),
            (
                "assumed_decode_step_seconds",
                Box::new(|c: &mut SchedulerConfig| c.assumed_decode_step_seconds = 0.0),
            ),
            (
                "assumed_engram_row_latency_us",
                Box::new(|c: &mut SchedulerConfig| c.assumed_engram_row_latency_us = 0.0),
            ),
        ];
        for (needle, mutate) in cases {
            let mut config = SchedulerConfig::default();
            mutate(&mut config);
            let error = config
                .validate()
                .expect_err("the mutated configuration must be rejected");
            assert!(
                error.to_string().contains(needle),
                "expected {needle:?} in {error}"
            );
        }
        SchedulerConfig::default()
            .validate()
            .expect("the unmutated configuration still validates");
    }
}
