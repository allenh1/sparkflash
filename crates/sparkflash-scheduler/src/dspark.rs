//! Spec 5.6.3: DSpark confidence-capped verification and live router expert reuse.
//!
//! The three-layer drafter (`DSPARK_LAYERS` = 37, 38, 39) proposes `DSPARK_BLOCK_SIZE` = 5
//! tokens and emits one confidence logit per position. DECISIONS.md D6 fixes the order the
//! specification leaves implicit: a position's confidence is the *sigmoid* of its raw logit and
//! the survival product is taken over those sigmoids before any truncation, so
//! `p_i = sigmoid(z_i)` and `P_survive(j) = prod_{i <= j} p_i`.
//!
//! The verified prefix is the longest `j` in `1..=5` whose survival probability is at least
//! `DSPARK_CONFIDENCE_CAP` = 0.10. When even `P_survive(1)` is below the cap the documented
//! answer is `k_verify = 0`: nothing in the block is verified, only the anchor token is
//! forwarded, and all five draft positions are masked. A masked position reuses the anchor
//! token's `ROUTER_TOP_K` = 6 experts (`DSV41_ROUTER_LIVE`), so a doomed position cannot pull
//! fresh experts into the routing decision - the mask is the complement of the verified prefix.
//!
//! This module is **policy only**: it decides a verified prefix and the router mask from the
//! confidence scores the engine's MTP stages produce. It never generates a draft token. The
//! serving path's single draft-token source is `sparkflash-engine`'s MTP forward
//! (`crates/sparkflash-engine/src/forward/mtp.rs`), and the target's acceptance rule decides how
//! many of those tokens were accepted.
//!
//! What this module does *not* model, stated so that "the DSpark parameters" cannot be read as
//! the whole draft head: the rank-256 Markov head (`dspark_markov_rank` = 256) and the draft's
//! own 3-of-128 expert selection. The head's logits arrive as inputs to [`sigmoid`], and the only
//! routing decision modelled here is the verified prefix's anchor-expert reuse for the *target*
//! model (`ROUTER_TOP_K` = 6, `DSV41_ROUTER_LIVE`). The DSpark parameters this module does model
//! are `DSPARK_LAYERS`, `DSPARK_BLOCK_SIZE`, `DSPARK_CONFIDENCE_CAP` and `DSPARK_VERIFY_ROWS`
//! (review M5 test-quality).

use crate::RequestId;
use crate::config::{DSPARK_BLOCK_SIZE, ROUTER_TOP_K, SchedulerConfig};
use crate::error::SchedulerError;
use crate::gates::CUDA_GRAPH_MAX_BATCH;

/// The sigmoid of a raw confidence logit.
///
/// This is the D6 step: the markov head emits logits, and the survival product is taken over
/// their sigmoids, never over the logits themselves.
#[must_use]
pub fn sigmoid(logit: f64) -> f64 {
    1.0 / (1.0 + (-logit).exp())
}

/// Which expert set one verification row uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSource {
    /// The row selects its own fresh Top-k experts.
    Fresh,
    /// The row reuses the anchor token's experts because it is not verified.
    Anchor,
}

/// The router mask implied by a verified prefix.
///
/// Verification row 0 is the anchor token and is always routed fresh; rows 1..=block_size are
/// the drafted positions, and only those inside the verified prefix may select fresh experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterMask {
    live_positions: usize,
    block_size: usize,
    top_k: usize,
}

impl RouterMask {
    /// The mask for a verified prefix of `live_positions` inside a `block_size` block.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` when the prefix leaves no room for the anchor
    /// row, when the block is empty, or when `top_k` is zero.
    pub fn new(
        live_positions: usize,
        block_size: usize,
        top_k: usize,
    ) -> Result<Self, SchedulerError> {
        if block_size == 0 {
            return Err(SchedulerError::InvalidArgument(
                "a DSpark block needs at least one position".to_owned(),
            ));
        }
        if live_positions > block_size {
            return Err(SchedulerError::InvalidArgument(format!(
                "a verified prefix of {live_positions} does not fit a {block_size}-position block"
            )));
        }
        if top_k == 0 {
            return Err(SchedulerError::InvalidArgument(
                "the router needs a non-zero Top-k".to_owned(),
            ));
        }
        Ok(Self {
            live_positions,
            block_size,
            top_k,
        })
    }

    /// Verification rows: the anchor plus every drafted position.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.block_size + 1
    }

    /// Drafted positions inside the verified prefix.
    #[must_use]
    pub const fn live_positions(&self) -> usize {
        self.live_positions
    }

    /// Drafted positions that were masked.
    #[must_use]
    pub const fn masked_positions(&self) -> usize {
        self.block_size - self.live_positions
    }

    /// Experts each routed row selects.
    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    /// True when `row` may select its own experts.
    ///
    /// Row 0 is the anchor token, so it is always fresh; a drafted row is fresh exactly when it
    /// is inside the verified prefix. An out-of-range row is never fresh.
    #[must_use]
    pub const fn routes_fresh_experts(&self, row: usize) -> bool {
        row < self.rows() && (row == 0 || row <= self.live_positions)
    }

    /// Where `row` takes its experts from.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for a row outside the block.
    pub fn source(&self, row: usize) -> Result<ExpertSource, SchedulerError> {
        if row >= self.rows() {
            return Err(SchedulerError::InvalidArgument(format!(
                "verification row {row} is outside a {}-row block",
                self.rows()
            )));
        }
        Ok(if self.routes_fresh_experts(row) {
            ExpertSource::Fresh
        } else {
            ExpertSource::Anchor
        })
    }

    /// The expert ids `row` routes to.
    ///
    /// A masked row takes the anchor's experts even when fresh experts are offered, so the mask
    /// fails closed: a caller cannot accidentally give a doomed position fresh experts.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for an out-of-range row, or when the set the
    /// row would use is not exactly `top_k` distinct ids.
    pub fn route(
        &self,
        row: usize,
        fresh: &[u32],
        anchor: &[u32],
    ) -> Result<Vec<u32>, SchedulerError> {
        let chosen = match self.source(row)? {
            ExpertSource::Fresh => fresh,
            ExpertSource::Anchor => anchor,
        };
        if chosen.len() != self.top_k {
            return Err(SchedulerError::InvalidArgument(format!(
                "row {row} routes {} experts, expected {}",
                chosen.len(),
                self.top_k
            )));
        }
        for (index, expert) in chosen.iter().enumerate() {
            if chosen[..index].contains(expert) {
                return Err(SchedulerError::InvalidArgument(format!(
                    "row {row} routes expert {expert} twice"
                )));
            }
        }
        Ok(chosen.to_vec())
    }
}

/// One DSpark block's confidence evidence and the decisions derived from it.
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkProposal {
    logits: Vec<f64>,
    cap: f64,
    block_size: usize,
    top_k: usize,
}

impl DsparkProposal {
    /// A proposal from the drafter's raw confidence logits.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` when the logit count is not the configured
    /// block size, when a logit is not finite, or when the configured cap is not inside (0, 1).
    pub fn from_logits(config: &SchedulerConfig, logits: &[f64]) -> Result<Self, SchedulerError> {
        config.validate()?;
        if logits.len() != config.dspark_block_size {
            return Err(SchedulerError::InvalidArgument(format!(
                "the drafter emitted {} confidences for a {}-position block",
                logits.len(),
                config.dspark_block_size
            )));
        }
        if let Some((index, logit)) = logits
            .iter()
            .enumerate()
            .find(|(_, logit)| !logit.is_finite())
        {
            return Err(SchedulerError::InvalidArgument(format!(
                "confidence {index} is not a finite logit: {logit}"
            )));
        }
        if !(config.dspark_confidence_cap > 0.0 && config.dspark_confidence_cap < 1.0) {
            return Err(SchedulerError::InvalidArgument(format!(
                "the confidence cap must be inside (0, 1), got {}",
                config.dspark_confidence_cap
            )));
        }
        Ok(Self {
            logits: logits.to_vec(),
            cap: config.dspark_confidence_cap,
            block_size: config.dspark_block_size,
            // The MoE router's Top-k is the specification's six. config.top_k is the HSI
            // indexer's candidate count (512) and must never leak into the expert mask.
            top_k: ROUTER_TOP_K,
        })
    }

    /// Positions in the block.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// The cap the prefix was taken against.
    #[must_use]
    pub const fn confidence_cap(&self) -> f64 {
        self.cap
    }

    /// `p_i = sigmoid(z_i)` for every position, in order.
    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        self.logits.iter().copied().map(sigmoid).collect()
    }

    /// `P_survive(j)` for every position, in order.
    #[must_use]
    pub fn survival(&self) -> Vec<f64> {
        let mut running = 1.0;
        self.probabilities()
            .into_iter()
            .map(|probability| {
                running *= probability;
                running
            })
            .collect()
    }

    /// `k_verify`: the longest verified prefix, or 0 when even the first position is below the
    /// cap.
    #[must_use]
    pub fn verified_prefix(&self) -> usize {
        let mut verified = 0;
        for (index, survival) in self.survival().into_iter().enumerate() {
            if survival >= self.cap {
                verified = index + 1;
            }
        }
        verified
    }

    /// Verification rows the scheduler forwards: the anchor plus the verified prefix.
    #[must_use]
    pub fn verify_rows(&self) -> usize {
        self.verified_prefix() + 1
    }

    /// The router mask for this block's verified prefix.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` when the mask cannot be built, which the
    /// constructor's own validation makes unreachable.
    pub fn router_mask(&self) -> Result<RouterMask, SchedulerError> {
        RouterMask::new(self.verified_prefix(), self.block_size, self.top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DSPARK_BLOCK_SIZE, DSPARK_CONFIDENCE_CAP, DSPARK_LAYERS, DSPARK_VERIFY_ROWS, ROUTER_TOP_K,
    };

    fn config() -> SchedulerConfig {
        SchedulerConfig::default()
    }

    fn proposal(logits: &[f64]) -> DsparkProposal {
        DsparkProposal::from_logits(&config(), logits).expect("a valid proposal")
    }

    /// D6, mechanically: a position's confidence is the sigmoid of its logit, so five logits of
    /// -1 give a first position of 0.269 (verified) and a second of 0.0724 (below the 0.10 cap).
    /// Multiplying the raw logits instead would give -1 and verify nothing, so this test tells
    /// the two orders apart.
    #[test]
    fn the_survival_product_uses_the_sigmoid_and_not_the_raw_logits() {
        let logits = [-1.0; DSPARK_BLOCK_SIZE];
        let proposal = proposal(&logits);
        let expected = 1.0 / (1.0 + 1.0f64.exp());
        for probability in proposal.probabilities() {
            assert!((probability - expected).abs() < 1e-12);
        }
        assert!(
            expected > DSPARK_CONFIDENCE_CAP,
            "the first position survives"
        );
        assert!(
            expected * expected < DSPARK_CONFIDENCE_CAP,
            "the second position does not"
        );
        assert_eq!(
            proposal.verified_prefix(),
            1,
            "sigmoid first, then the product"
        );
    }

    /// The cap is inclusive and the prefix is the longest one that survives.
    #[test]
    fn the_verified_prefix_is_the_longest_prefix_at_or_above_the_cap() {
        let cases: [(&[f64], usize); 4] = [
            // sigmoid(2) = 0.881, so even the fifth survival (0.530) clears the cap.
            (&[2.0, 2.0, 2.0, 2.0, 2.0], 5),
            // sigmoid(0) = 0.5: 0.125 clears the cap, 0.0625 does not.
            (&[0.0, 0.0, 0.0, 0.0, 0.0], 3),
            // sigmoid(-4) = 0.018 collapses the running product at the fourth position.
            (&[1.0, 1.0, 1.0, -4.0, -4.0], 3),
            // A first position below the cap verifies nothing, whatever follows it.
            (&[-6.0, 6.0, 6.0, 6.0, 6.0], 0),
        ];
        for (logits, expected) in cases {
            let proposal = proposal(logits);
            assert_eq!(proposal.verified_prefix(), expected, "logits {logits:?}");
            assert_eq!(proposal.verify_rows(), expected + 1);
        }
        // The 0.5-logit case is exactly at the cap only when the product stays above it.
        let at_the_cap = proposal(&[0.0; DSPARK_BLOCK_SIZE]);
        assert!((at_the_cap.survival()[4] - 0.5f64.powi(5)).abs() < 1e-15);
        assert!(at_the_cap.survival()[4] < DSPARK_CONFIDENCE_CAP);
        assert_eq!(
            at_the_cap.verified_prefix(),
            3,
            "0.5^3 = 0.125 >= 0.10 > 0.5^4"
        );
    }

    /// The empty verified prefix has a documented answer: only the anchor row is forwarded and
    /// every drafted position is masked.
    #[test]
    fn an_empty_prefix_forwards_only_the_anchor_and_masks_every_position() {
        let proposal = proposal(&[-6.0; DSPARK_BLOCK_SIZE]);
        assert_eq!(proposal.verified_prefix(), 0);
        assert_eq!(proposal.verify_rows(), 1);
        let mask = proposal.router_mask().expect("mask");
        assert_eq!(mask.rows(), DSPARK_VERIFY_ROWS);
        assert_eq!(mask.live_positions(), 0);
        assert_eq!(mask.masked_positions(), DSPARK_BLOCK_SIZE);
        assert!(mask.routes_fresh_experts(0), "the anchor is a real token");
        for row in 1..mask.rows() {
            assert!(!mask.routes_fresh_experts(row), "row {row} is doomed");
        }
    }

    /// The mask is exactly the complement of the verified prefix, for every prefix length.
    #[test]
    fn the_router_mask_is_the_complement_of_the_verified_prefix() {
        for live in 0..=DSPARK_BLOCK_SIZE {
            let mask = RouterMask::new(live, DSPARK_BLOCK_SIZE, ROUTER_TOP_K).expect("mask");
            let fresh: Vec<usize> = (0..mask.rows())
                .filter(|row| mask.routes_fresh_experts(*row))
                .collect();
            let expected: Vec<usize> = (0..=live).collect();
            assert_eq!(fresh, expected, "live prefix {live}");
            assert_eq!(
                mask.live_positions() + mask.masked_positions(),
                DSPARK_BLOCK_SIZE
            );
            assert_eq!(mask.rows(), DSPARK_VERIFY_ROWS);
            for row in mask.live_positions() + 1..mask.rows() {
                assert_eq!(mask.source(row).expect("in range"), ExpertSource::Anchor);
            }
        }
    }

    /// A doomed position takes the anchor's experts even when fresh ones are offered.
    #[test]
    fn a_doomed_position_cannot_select_fresh_experts() {
        let mask = RouterMask::new(2, DSPARK_BLOCK_SIZE, ROUTER_TOP_K).expect("mask");
        let fresh: Vec<u32> = (900..900 + ROUTER_TOP_K as u32).collect();
        let anchor: Vec<u32> = (100..100 + ROUTER_TOP_K as u32).collect();
        for row in 1..=2 {
            assert_eq!(
                mask.route(row, &fresh, &anchor).expect("live row"),
                fresh,
                "row {row} is inside the prefix"
            );
        }
        for row in 3..mask.rows() {
            assert_eq!(
                mask.route(row, &fresh, &anchor).expect("masked row"),
                anchor,
                "row {row} must reuse the anchor experts"
            );
            assert_eq!(mask.source(row).expect("in range"), ExpertSource::Anchor);
        }
        assert_eq!(mask.route(0, &fresh, &anchor).expect("anchor"), fresh);
        assert!(mask.route(mask.rows(), &fresh, &anchor).is_err());
        assert!(mask.route(1, &fresh[..2], &anchor).is_err(), "wrong Top-k");
        let repeated = vec![7u32; ROUTER_TOP_K];
        assert!(
            mask.route(1, &repeated, &anchor).is_err(),
            "duplicate expert"
        );
    }

    /// The drafter's block, the verify-row count and the layer ids are the specification's.
    #[test]
    fn the_spec_numbers_are_the_contract() {
        assert_eq!(DSPARK_BLOCK_SIZE, 5);
        assert_eq!(DSPARK_VERIFY_ROWS, 6);
        assert_eq!(ROUTER_TOP_K, 6);
        assert_eq!(DSPARK_LAYERS, [37, 38, 39]);
        assert!((DSPARK_CONFIDENCE_CAP - 0.10).abs() < 1e-15);
        let config = config();
        assert_eq!(config.dspark_block_size, DSPARK_BLOCK_SIZE);
        assert_eq!(config.dspark_noise_token_id, 128_799);
    }

    /// A malformed proposal is refused before any decision is made.
    #[test]
    fn a_malformed_proposal_is_refused() {
        let config = config();
        assert!(DsparkProposal::from_logits(&config, &[0.0; 4]).is_err());
        assert!(DsparkProposal::from_logits(&config, &[f64::NAN; 5]).is_err());
        assert!(DsparkProposal::from_logits(&config, &[f64::INFINITY; 5]).is_err());
        assert!(RouterMask::new(DSPARK_BLOCK_SIZE + 1, DSPARK_BLOCK_SIZE, 6).is_err());
        assert!(RouterMask::new(0, 0, 6).is_err());
        assert!(RouterMask::new(0, DSPARK_BLOCK_SIZE, 0).is_err());
    }

    /// The mask carries the MoE router's Top-k, not the indexer's candidate count.
    #[test]
    fn the_router_top_k_is_the_moe_router_top_k() {
        assert_ne!(
            config().top_k,
            ROUTER_TOP_K,
            "config.top_k is the indexer's candidate count; if these ever agree, revisit this test"
        );
        let mask = proposal(&[4.0; DSPARK_BLOCK_SIZE])
            .router_mask()
            .expect("mask");
        assert_eq!(mask.top_k(), ROUTER_TOP_K);
        assert_eq!(mask.top_k(), 6);
        let fresh: Vec<u32> = (900..906).collect();
        let anchor: Vec<u32> = (100..106).collect();
        assert_eq!(mask.route(1, &fresh, &anchor).expect("row"), fresh);
    }
}
// =============================================================================================
// WS-E: on-device DSpark MTP drafting (policy and measurement harness)
// =============================================================================================
//
// Everything below is policy and arithmetic. It plans the draft forward and measures a verify
// step; it never executes one on a device, and it says so in every report it produces. The WS-E
// plan is docs/build/ws-e-mtp-device-20261021.md; this is the testable half of it.

/// Drafts the MTP block proposes per verify step: the specification's block size, five.
pub const MTP_DRAFT_K: usize = DSPARK_BLOCK_SIZE;

/// Batch-size buckets the draft forward may use; the largest is the CUDA-graph ceiling.
///
/// The k drafts of one request are one batched forward, and several requests share a forward, so
/// the graph's batch dimension is the number of requests in a batch and the row count is their
/// draft tokens. Reusing the serving path's existing graph buckets is the WS-E plan's cost
/// argument: drafting more tokens per step does not add a launch.
pub const DRAFT_BATCH_BUCKETS: [usize; 5] = [1, 2, 4, 8, CUDA_GRAPH_MAX_BATCH as usize];

/// The WS-E serving gate: accepted tokens per verify step, per second.
pub const TARGET_TOKENS_PER_SECOND: f64 = 90.0;

/// Measured CPU host-twin accepted draft tokens (out of MEASURED_HOST_TWIN_VERIFY_STEPS).
///
/// Evidence: /home/user/dspark-accept-evidence/ (24 accepted in 42 verify steps, CI
/// 1.333..1.810 accepted tokens per verify step), described in docs/build/tp4-host-twin.md. This
/// is a measured *host-twin* number: it is not a device measurement and the WS-E plan must not
/// present it as one.
pub const MEASURED_HOST_TWIN_ACCEPTED: u64 = 24;

/// Measured CPU host-twin verify steps that produced MEASURED_HOST_TWIN_ACCEPTED.
pub const MEASURED_HOST_TWIN_VERIFY_STEPS: u64 = 42;

/// Measured CPU host-twin accepted tokens per verify step: 1 + 24/42 = 1.5714.
pub const MEASURED_HOST_TWIN_AL: f64 = 1.571_428_571_428_571_4;

/// Rolling window the adaptive draft length averages over.
pub const DRAFT_LENGTH_WINDOW: usize = 8;

/// Mean acceptance at or above which the adaptive policy drafts one more token.
pub const DRAFT_LENGTH_HIGH_ACCEPTANCE: f64 = 0.75;

/// Mean acceptance below which the adaptive policy drafts one fewer token.
pub const DRAFT_LENGTH_LOW_ACCEPTANCE: f64 = 0.25;

/// One corpus the acceptance-length harness reports separately.
///
/// The WS-E plan requires chat, code and natural-language prompts to be measured separately: a
/// single pooled number can hide a corpus whose drafts are never accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlCorpus {
    /// Conversational prompts.
    Chat,
    /// Code and code-adjacent prompts.
    Code,
    /// Natural-language prose.
    NaturalLanguage,
}

impl AlCorpus {
    /// Every corpus, in report order.
    pub const ALL: [Self; 3] = [Self::Chat, Self::Code, Self::NaturalLanguage];

    /// The corpus's report name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Code => "code",
            Self::NaturalLanguage => "natural-language",
        }
    }
}

/// One verify step's acceptance evidence and wall time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlSample {
    /// Which corpus the step belongs to.
    pub corpus: AlCorpus,
    /// Draft tokens the MTP block proposed.
    pub drafted: usize,
    /// Of those, tokens the target accepted.
    pub accepted: usize,
    /// Wall time of the verify step, in milliseconds.
    pub verify_ms: f64,
}

impl AlSample {
    /// A validated sample.
    ///
    /// # Errors
    /// Returns SchedulerError::InvalidArgument when the step drafted nothing, accepted more than
    /// it drafted, or did not take a finite positive time.
    pub fn new(
        corpus: AlCorpus,
        drafted: usize,
        accepted: usize,
        verify_ms: f64,
    ) -> Result<Self, SchedulerError> {
        if drafted == 0 {
            return Err(SchedulerError::InvalidArgument(
                "a verify step must draft at least one token".to_owned(),
            ));
        }
        if accepted > drafted {
            return Err(SchedulerError::InvalidArgument(format!(
                "a verify step accepted {accepted} of {drafted} drafts"
            )));
        }
        if !(verify_ms.is_finite() && verify_ms > 0.0) {
            return Err(SchedulerError::InvalidArgument(format!(
                "a verify step must take a finite positive time, got {verify_ms} ms"
            )));
        }
        Ok(Self {
            corpus,
            drafted,
            accepted,
            verify_ms,
        })
    }
}

/// One corpus's aggregated acceptance-length evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlMeasurement {
    /// Corpus measured.
    pub corpus: AlCorpus,
    /// Verify steps measured.
    pub steps: u64,
    /// Draft tokens proposed.
    pub drafted: u64,
    /// Draft tokens accepted.
    pub accepted: u64,
    /// Total verify time, in milliseconds.
    pub total_ms: f64,
}

impl AlMeasurement {
    /// Mean verify step time in milliseconds; zero for an empty measurement.
    #[must_use]
    pub fn mean_verify_step_ms(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.total_ms / self.steps as f64
        }
    }

    /// Accepted tokens per verify step: the anchor plus the accepted drafts.
    #[must_use]
    pub fn accepted_tokens_per_verify_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            (self.accepted + self.steps) as f64 / self.steps as f64
        }
    }

    /// Fraction of proposed draft tokens the target accepted.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    /// Accepted tokens per second: the serve gate's quantity.
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        let mean = self.mean_verify_step_ms();
        if mean <= 0.0 {
            0.0
        } else {
            self.accepted_tokens_per_verify_step() * 1000.0 / mean
        }
    }

    /// The gate report for this measurement, with a plain-decode baseline reported alongside.
    #[must_use]
    pub fn gate(&self, plain_decode: Option<PlainDecodeBaseline>) -> AlGateReport {
        let al = self.accepted_tokens_per_verify_step();
        let tokens_per_second = self.tokens_per_second();
        AlGateReport {
            corpus: self.corpus,
            accepted_tokens_per_verify_step: al,
            mean_verify_step_ms: self.mean_verify_step_ms(),
            tokens_per_second,
            required_verify_step_ms: required_verify_step_ms(al, TARGET_TOKENS_PER_SECOND)
                .unwrap_or(f64::INFINITY),
            passes: tokens_per_second >= TARGET_TOKENS_PER_SECOND,
            plain_decode_tokens_per_second: plain_decode
                .map(|baseline| baseline.tokens_per_second()),
            device_execution: false,
        }
    }
}

/// The verify-step time at which an acceptance length reaches a token-rate target.
///
/// Returns None for a non-finite or non-positive acceptance length or target.
#[must_use]
pub fn required_verify_step_ms(al: f64, target_tokens_per_second: f64) -> Option<f64> {
    if !(al.is_finite()
        && al > 0.0
        && target_tokens_per_second.is_finite()
        && target_tokens_per_second > 0.0)
    {
        return None;
    }
    Some(al * 1000.0 / target_tokens_per_second)
}

/// A plain decode's measured cost, so a speculative gate is never quoted alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlainDecodeBaseline {
    /// Mean decode step time, in milliseconds.
    pub ms_per_step: f64,
}

impl PlainDecodeBaseline {
    /// A validated baseline.
    ///
    /// # Errors
    /// Returns SchedulerError::InvalidArgument for a non-finite or non-positive step time.
    pub fn new(ms_per_step: f64) -> Result<Self, SchedulerError> {
        if !(ms_per_step.is_finite() && ms_per_step > 0.0) {
            return Err(SchedulerError::InvalidArgument(format!(
                "a plain decode step must take a finite positive time, got {ms_per_step} ms"
            )));
        }
        Ok(Self { ms_per_step })
    }

    /// Plain decode tokens per second.
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        1000.0 / self.ms_per_step
    }
}

/// A WS-E gate report: the arithmetic gate, explicitly not a device execution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlGateReport {
    /// Corpus measured.
    pub corpus: AlCorpus,
    /// Accepted tokens per verify step.
    pub accepted_tokens_per_verify_step: f64,
    /// Mean verify step time, in milliseconds.
    pub mean_verify_step_ms: f64,
    /// Accepted tokens per second.
    pub tokens_per_second: f64,
    /// Verify-step time the gate needs, in milliseconds.
    pub required_verify_step_ms: f64,
    /// Whether tokens_per_second reaches TARGET_TOKENS_PER_SECOND.
    pub passes: bool,
    /// The plain decode's token rate, when a baseline was supplied.
    pub plain_decode_tokens_per_second: Option<f64>,
    /// Always false: this module plans and measures host arithmetic, it does not run a device.
    pub device_execution: bool,
}

/// The acceptance-length measurement harness over the three corpora.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AlHarness {
    samples: Vec<AlSample>,
}

impl AlHarness {
    /// An empty harness.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one verify step.
    ///
    /// # Errors
    /// Returns SchedulerError::InvalidArgument for an invalid sample.
    pub fn record(&mut self, sample: AlSample) -> Result<(), SchedulerError> {
        AlSample::new(
            sample.corpus,
            sample.drafted,
            sample.accepted,
            sample.verify_ms,
        )?;
        self.samples.push(sample);
        Ok(())
    }

    /// Records a raw step, validating it first.
    ///
    /// # Errors
    /// As AlSample::new.
    pub fn record_step(
        &mut self,
        corpus: AlCorpus,
        drafted: usize,
        accepted: usize,
        verify_ms: f64,
    ) -> Result<(), SchedulerError> {
        self.record(AlSample::new(corpus, drafted, accepted, verify_ms)?)
    }

    /// Every recorded sample, in recording order.
    #[must_use]
    pub fn samples(&self) -> &[AlSample] {
        &self.samples
    }

    /// The aggregate for one corpus, or None when it has no sample.
    #[must_use]
    pub fn measurement(&self, corpus: AlCorpus) -> Option<AlMeasurement> {
        let mut measurement = AlMeasurement {
            corpus,
            steps: 0,
            drafted: 0,
            accepted: 0,
            total_ms: 0.0,
        };
        for sample in self.samples.iter().filter(|sample| sample.corpus == corpus) {
            measurement.steps += 1;
            measurement.drafted += sample.drafted as u64;
            measurement.accepted += sample.accepted as u64;
            measurement.total_ms += sample.verify_ms;
        }
        (measurement.steps > 0).then_some(measurement)
    }

    /// One aggregate per corpus that has samples, in AlCorpus::ALL order.
    #[must_use]
    pub fn measurements(&self) -> Vec<AlMeasurement> {
        AlCorpus::ALL
            .into_iter()
            .filter_map(|corpus| self.measurement(corpus))
            .collect()
    }

    /// The gate reports for every measured corpus, plain decode reported alongside.
    #[must_use]
    pub fn gate_reports(&self, plain_decode: Option<PlainDecodeBaseline>) -> Vec<AlGateReport> {
        self.measurements()
            .into_iter()
            .map(|measurement| measurement.gate(plain_decode))
            .collect()
    }

    /// Drops every sample.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// The adaptive draft-length policy: the k the MTP block should propose this step.
///
/// The k drafts still run as one batched forward; the policy only chooses how many. It is a
/// heuristic, not a measured optimum: the WS-E harness exists to decide whether backing off
/// helps, and nothing here claims it does on a device.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftLengthPolicy {
    min: usize,
    max: usize,
    window: usize,
    length: usize,
    samples: Vec<f64>,
}

impl DraftLengthPolicy {
    /// A policy drafting max initially, backing off below min and growing back above it.
    ///
    /// # Errors
    /// Returns SchedulerError::InvalidConfig unless 1 <= min <= max <= MTP_DRAFT_K and window is
    /// positive.
    pub fn new(min: usize, max: usize, window: usize) -> Result<Self, SchedulerError> {
        if min == 0 || max == 0 || min > max || max > MTP_DRAFT_K {
            return Err(SchedulerError::InvalidConfig(format!(
                "a draft-length policy needs 1 <= min <= max <= {MTP_DRAFT_K}, got min {min} and max {max}"
            )));
        }
        if window == 0 {
            return Err(SchedulerError::InvalidConfig(
                "an adaptive draft-length window must be at least one step".to_owned(),
            ));
        }
        Ok(Self {
            min,
            max,
            window,
            length: max,
            samples: Vec::with_capacity(window),
        })
    }

    /// The policy the specification's block implies: one to five drafts, the default window.
    ///
    /// # Errors
    /// This cannot fail at the specification's block size; the error is the shared constructor's.
    pub fn specification_default() -> Result<Self, SchedulerError> {
        Self::new(1, MTP_DRAFT_K, DRAFT_LENGTH_WINDOW)
    }

    /// The k the next verify step should draft.
    #[must_use]
    pub const fn draft_length(&self) -> usize {
        self.length
    }

    /// Smallest k the policy may choose.
    #[must_use]
    pub const fn min(&self) -> usize {
        self.min
    }

    /// Largest k the policy may choose.
    #[must_use]
    pub const fn max(&self) -> usize {
        self.max
    }

    /// Samples the mean is taken over.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.samples.len()
    }

    /// Mean acceptance over the window, or None until a full window has been observed.
    #[must_use]
    pub fn mean_acceptance(&self) -> Option<f64> {
        if self.samples.len() < self.window {
            return None;
        }
        Some(self.samples.iter().sum::<f64>() / self.samples.len() as f64)
    }

    /// Records one verify step's acceptance and adjusts the next draft length.
    ///
    /// The window is a plain mean of per-step acceptance rates. A full window at or above
    /// DRAFT_LENGTH_HIGH_ACCEPTANCE grows k by one; one below DRAFT_LENGTH_LOW_ACCEPTANCE
    /// shrinks it by one; anything between keeps it. Both moves stop at max and min.
    ///
    /// # Errors
    /// Returns SchedulerError::InvalidArgument when the step drafted nothing, accepted more than
    /// it drafted, or drafted more than this policy's max.
    pub fn observe(&mut self, drafted: usize, accepted: usize) -> Result<(), SchedulerError> {
        if drafted == 0 {
            return Err(SchedulerError::InvalidArgument(
                "a verify step must draft at least one token".to_owned(),
            ));
        }
        if accepted > drafted {
            return Err(SchedulerError::InvalidArgument(format!(
                "a verify step accepted {accepted} of {drafted} drafts"
            )));
        }
        if drafted > self.max {
            return Err(SchedulerError::InvalidArgument(format!(
                "a verify step drafted {drafted} tokens, above the policy's maximum of {}",
                self.max
            )));
        }
        self.samples.push(accepted as f64 / drafted as f64);
        if self.samples.len() > self.window {
            self.samples.remove(0);
        }
        if let Some(mean) = self.mean_acceptance() {
            if mean >= DRAFT_LENGTH_HIGH_ACCEPTANCE {
                self.length = (self.length + 1).min(self.max);
            } else if mean < DRAFT_LENGTH_LOW_ACCEPTANCE {
                self.length = self.length.saturating_sub(1).max(self.min);
            }
        }
        Ok(())
    }
}

/// One batched draft forward's membership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftBatch {
    /// CUDA-graph batch bucket the forward uses.
    pub bucket: usize,
    /// Requests in the batch, in plan order.
    pub requests: Vec<RequestId>,
    /// Draft tokens their k values sum to.
    pub draft_tokens: usize,
    /// Attention rows the forward pads to: the bucket times the batch's largest k.
    pub padded_rows: usize,
}

impl DraftBatch {
    /// Requests in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// True when the batch is empty, which the planner never produces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// The plan for drafting every running request's k tokens in one forward per batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftBatchPlan {
    batches: Vec<DraftBatch>,
    requests: usize,
    draft_tokens: usize,
    padded_rows: usize,
}

impl DraftBatchPlan {
    /// Plans the draft forward for one step's requests and their draft lengths.
    ///
    /// The plan fills the smallest existing bucket that covers the remaining requests, so the
    /// number of device forwards is the number of batches, never the number of drafts.
    ///
    /// # Errors
    /// Returns SchedulerError::InvalidArgument for a draft length outside 1..=MTP_DRAFT_K.
    pub fn plan(drafts: &[(RequestId, usize)]) -> Result<Self, SchedulerError> {
        for (seq, k) in drafts {
            if *k == 0 || *k > MTP_DRAFT_K {
                return Err(SchedulerError::InvalidArgument(format!(
                    "request {} drafts {k} tokens, outside 1..={MTP_DRAFT_K}",
                    seq.0
                )));
            }
        }
        let mut batches = Vec::new();
        let mut start = 0usize;
        let mut requests = 0usize;
        let mut draft_tokens = 0usize;
        let mut padded_rows = 0usize;
        while start < drafts.len() {
            let remaining = drafts.len() - start;
            let bucket = DRAFT_BATCH_BUCKETS
                .iter()
                .copied()
                .find(|candidate| *candidate >= remaining)
                .unwrap_or(
                    *DRAFT_BATCH_BUCKETS
                        .last()
                        .expect("the buckets are not empty"),
                );
            let take = remaining.min(bucket);
            let slice = &drafts[start..start + take];
            let widest = slice.iter().map(|(_, k)| *k).max().unwrap_or(0);
            let batch = DraftBatch {
                bucket,
                requests: slice.iter().map(|(seq, _)| *seq).collect(),
                draft_tokens: slice.iter().map(|(_, k)| *k).sum(),
                padded_rows: bucket * widest,
            };
            requests += take;
            draft_tokens += batch.draft_tokens;
            padded_rows += batch.padded_rows;
            batches.push(batch);
            start += take;
        }
        Ok(Self {
            batches,
            requests,
            draft_tokens,
            padded_rows,
        })
    }

    /// The planned batches, in execution order.
    #[must_use]
    pub fn batches(&self) -> &[DraftBatch] {
        &self.batches
    }

    /// Device forwards a step makes: one per batch, independent of the draft count.
    #[must_use]
    pub fn device_forwards(&self) -> usize {
        self.batches.len()
    }

    /// Requests the plan covers.
    #[must_use]
    pub const fn requests(&self) -> usize {
        self.requests
    }

    /// Draft tokens the plan proposes.
    #[must_use]
    pub const fn draft_tokens(&self) -> usize {
        self.draft_tokens
    }

    /// Attention rows across all batches, padding included.
    #[must_use]
    pub const fn padded_rows(&self) -> usize {
        self.padded_rows
    }
}

#[cfg(test)]
mod ws_e_tests {
    use super::*;

    fn measured_host_twin_harness(step_ms: f64) -> AlHarness {
        let mut harness = AlHarness::new();
        for index in 0..MEASURED_HOST_TWIN_VERIFY_STEPS {
            let accepted = if index < MEASURED_HOST_TWIN_ACCEPTED {
                1
            } else {
                0
            };
            harness
                .record_step(AlCorpus::Chat, MTP_DRAFT_K, accepted as usize, step_ms)
                .expect("a measured-shaped sample");
        }
        harness
    }

    fn plain() -> PlainDecodeBaseline {
        PlainDecodeBaseline::new(11.1).expect("a decode step time")
    }

    #[test]
    fn the_host_twin_al_is_the_measured_ratio() {
        assert!((MEASURED_HOST_TWIN_AL - 1.571_428_571_428_571_4).abs() < 1e-15);
        assert_eq!(MEASURED_HOST_TWIN_VERIFY_STEPS, 42);
        assert_eq!(MEASURED_HOST_TWIN_ACCEPTED, 24);
        let harness = measured_host_twin_harness(16.0);
        let measurement = harness.measurement(AlCorpus::Chat).expect("chat");
        assert_eq!(measurement.steps, 42);
        assert!(
            (measurement.accepted_tokens_per_verify_step() - MEASURED_HOST_TWIN_AL).abs() < 1e-12
        );
        assert!((measurement.acceptance_rate() - 24.0 / (42.0 * 5.0)).abs() < 1e-12);
    }

    #[test]
    fn the_gate_is_accepted_tokens_per_second_and_its_required_step_time_is_exact() {
        // AL 1.5714 needs 17.46 ms per verify step to reach 90 tok/s.
        let required = required_verify_step_ms(MEASURED_HOST_TWIN_AL, TARGET_TOKENS_PER_SECOND)
            .expect("a valid gate");
        assert!((required - 17.4603).abs() < 1e-3, "{required}");
        assert!(required_verify_step_ms(0.0, 90.0).is_none());
        assert!(required_verify_step_ms(f64::NAN, 90.0).is_none());
        assert!(required_verify_step_ms(1.5, 0.0).is_none());

        let passing = measured_host_twin_harness(17.0)
            .measurement(AlCorpus::Chat)
            .expect("chat")
            .gate(Some(plain()));
        assert!(passing.passes, "{passing:?}");
        assert!((passing.tokens_per_second - MEASURED_HOST_TWIN_AL * 1000.0 / 17.0).abs() < 1e-9);
        assert!(!passing.device_execution, "the harness is not a device run");
        assert_eq!(
            passing.plain_decode_tokens_per_second,
            Some(1000.0 / 11.1),
            "the gate always reports the plain decode alongside"
        );

        let failing = measured_host_twin_harness(18.0)
            .measurement(AlCorpus::Chat)
            .expect("chat")
            .gate(Some(plain()));
        assert!(!failing.passes, "{failing:?}");
        assert!(failing.tokens_per_second < TARGET_TOKENS_PER_SECOND);
    }

    #[test]
    fn the_harness_reports_each_corpus_separately() {
        let mut harness = AlHarness::new();
        harness
            .record_step(AlCorpus::Chat, 5, 1, 10.0)
            .expect("chat step");
        harness
            .record_step(AlCorpus::Code, 5, 4, 12.0)
            .expect("code step");
        harness
            .record_step(AlCorpus::NaturalLanguage, 5, 0, 9.0)
            .expect("prose step");
        harness
            .record_step(AlCorpus::Code, 5, 2, 14.0)
            .expect("code step two");
        let reports = harness.gate_reports(Some(plain()));
        assert_eq!(reports.len(), 3);
        assert_eq!(
            reports
                .iter()
                .map(|report| report.corpus)
                .collect::<Vec<_>>(),
            AlCorpus::ALL.to_vec()
        );
        let code = harness.measurement(AlCorpus::Code).expect("code");
        assert_eq!(code.steps, 2);
        assert_eq!(code.drafted, 10);
        assert_eq!(code.accepted, 6);
        assert!((code.mean_verify_step_ms() - 13.0).abs() < 1e-12);
        assert!(
            (code.accepted_tokens_per_verify_step() - 4.0).abs() < 1e-12,
            "2 + 6/2"
        );
        let prose = harness
            .measurement(AlCorpus::NaturalLanguage)
            .expect("prose");
        assert!((prose.accepted_tokens_per_verify_step() - 1.0).abs() < 1e-12);
        assert!((prose.acceptance_rate() - 0.0).abs() < 1e-12);
        harness.clear();
        assert!(harness.measurements().is_empty());
        assert!(harness.measurement(AlCorpus::Chat).is_none());
    }

    #[test]
    fn malformed_samples_are_refused_before_they_reach_a_gate() {
        assert!(AlSample::new(AlCorpus::Chat, 0, 0, 10.0).is_err());
        assert!(AlSample::new(AlCorpus::Chat, 2, 3, 10.0).is_err());
        assert!(AlSample::new(AlCorpus::Chat, 2, 1, 0.0).is_err());
        assert!(AlSample::new(AlCorpus::Chat, 2, 1, -1.0).is_err());
        assert!(AlSample::new(AlCorpus::Chat, 2, 1, f64::NAN).is_err());
        assert!(AlSample::new(AlCorpus::Chat, 2, 1, f64::INFINITY).is_err());
        assert!(PlainDecodeBaseline::new(0.0).is_err());
        assert!(PlainDecodeBaseline::new(-2.0).is_err());
        assert!(PlainDecodeBaseline::new(f64::NAN).is_err());
    }

    #[test]
    fn the_adaptive_policy_backs_off_below_the_low_watermark_and_recovers_above_the_high_one() {
        let mut policy = DraftLengthPolicy::specification_default().expect("policy");
        assert_eq!(policy.draft_length(), MTP_DRAFT_K);
        assert_eq!(policy.min(), 1);
        assert_eq!(policy.max(), MTP_DRAFT_K);
        // A full window of perfect acceptance keeps k at its ceiling.
        for _ in 0..DRAFT_LENGTH_WINDOW {
            policy.observe(MTP_DRAFT_K, MTP_DRAFT_K).expect("perfect");
        }
        assert_eq!(policy.draft_length(), MTP_DRAFT_K);
        assert_eq!(policy.mean_acceptance(), Some(1.0));
        // A collapse takes k down one step per full window, never below the minimum.
        for _ in 0..(DRAFT_LENGTH_WINDOW * 6) {
            policy.observe(MTP_DRAFT_K, 0).expect("collapse");
        }
        assert_eq!(policy.draft_length(), 1);
        // Recovery grows it back, never above the maximum.
        for _ in 0..(DRAFT_LENGTH_WINDOW * 8) {
            policy.observe(MTP_DRAFT_K, MTP_DRAFT_K).expect("recovery");
        }
        assert_eq!(policy.draft_length(), MTP_DRAFT_K);
        assert!(policy.observe(0, 0).is_err());
        assert!(policy.observe(2, 3).is_err());
        assert!(policy.observe(MTP_DRAFT_K + 1, 0).is_err());
        assert!(DraftLengthPolicy::new(0, 5, 8).is_err());
        assert!(DraftLengthPolicy::new(3, 2, 8).is_err());
        assert!(DraftLengthPolicy::new(1, MTP_DRAFT_K + 1, 8).is_err());
        assert!(DraftLengthPolicy::new(1, 5, 0).is_err());
    }

    #[test]
    fn one_forward_carries_every_request_draft_in_a_batch_bucket() {
        let drafts: Vec<(RequestId, usize)> = (0..4).map(|index| (RequestId(index), 5)).collect();
        let plan = DraftBatchPlan::plan(&drafts).expect("plan");
        assert_eq!(plan.device_forwards(), 1, "four requests share one forward");
        assert_eq!(plan.requests(), 4);
        assert_eq!(plan.draft_tokens(), 20);
        assert_eq!(plan.padded_rows(), 20);
        assert_eq!(plan.batches()[0].bucket, 4);
        assert_eq!(plan.batches()[0].requests.len(), 4);

        let single = DraftBatchPlan::plan(&[(RequestId(7), 3)]).expect("plan");
        assert_eq!(single.device_forwards(), 1);
        assert_eq!(single.draft_tokens(), 3);
        assert_eq!(single.batches()[0].padded_rows, 3);

        let many: Vec<(RequestId, usize)> = (0..17).map(|index| (RequestId(index), 5)).collect();
        let plan = DraftBatchPlan::plan(&many).expect("plan");
        assert_eq!(
            plan.device_forwards(),
            2,
            "17 requests is a 16 bucket and a 1 bucket"
        );
        assert_eq!(plan.batches()[0].bucket, CUDA_GRAPH_MAX_BATCH as usize);
        assert_eq!(plan.batches()[1].bucket, 1);
        assert_eq!(plan.requests(), 17);
        assert_eq!(plan.draft_tokens(), 85);
        assert_ne!(
            plan.device_forwards(),
            plan.draft_tokens() / MTP_DRAFT_K,
            "the forward count is batch-count, not draft-count"
        );

        let empty = DraftBatchPlan::plan(&[]).expect("an empty step");
        assert_eq!(empty.device_forwards(), 0);
        assert_eq!(empty.requests(), 0);
        assert!(DraftBatchPlan::plan(&[(RequestId(0), 0)]).is_err());
        assert!(DraftBatchPlan::plan(&[(RequestId(0), MTP_DRAFT_K + 1)]).is_err());
    }

    #[test]
    fn the_draft_buckets_are_sorted_and_end_at_the_cuda_graph_ceiling() {
        assert_eq!(DRAFT_BATCH_BUCKETS, [1, 2, 4, 8, 16]);
        assert_eq!(
            *DRAFT_BATCH_BUCKETS.last().expect("buckets"),
            CUDA_GRAPH_MAX_BATCH as usize
        );
        assert!(DRAFT_BATCH_BUCKETS.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(MTP_DRAFT_K, DSPARK_BLOCK_SIZE);
    }

    #[test]
    fn the_required_step_time_falls_as_the_acceptance_length_rises() {
        // Arithmetic, not a device measurement: 3.2 and 5.8 are counterfactual acceptance lengths
        // used to show the shape of the trade. The only measured AL this crate uses is the host
        // twin's 1.571, which leaves 17.46 ms per verify step.
        let higher_low = required_verify_step_ms(3.2, TARGET_TOKENS_PER_SECOND).expect("gate");
        let higher_high = required_verify_step_ms(5.8, TARGET_TOKENS_PER_SECOND).expect("gate");
        let host_twin =
            required_verify_step_ms(MEASURED_HOST_TWIN_AL, TARGET_TOKENS_PER_SECOND).expect("gate");
        assert!((higher_low - 35.555).abs() < 1e-2, "{higher_low}");
        assert!((higher_high - 64.444).abs() < 1e-2, "{higher_high}");
        assert!((host_twin - 17.4603).abs() < 1e-3, "{host_twin}");
        assert!(host_twin < higher_low);
    }
}
