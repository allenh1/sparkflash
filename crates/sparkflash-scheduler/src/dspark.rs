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

use crate::config::{ROUTER_TOP_K, SchedulerConfig};
use crate::error::SchedulerError;

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
