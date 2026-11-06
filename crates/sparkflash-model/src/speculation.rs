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

//! How many of a drafter's tokens a speculative step verifies.
//!
//! A verified draft costs a row of the target's forward whether or not the target agrees with
//! it, so a long chain of unlikely drafts is slower than a short one. Every drafting backend - the
//! V4.1 MTP stages, Qwen4's DFlash 2 block - faces the same choice once a step, and this policy
//! makes it the same way for each: the drafter's own confidence in each draft, calibrated by the
//! acceptances the sequence has seen, against the backend's modelled step cost.
//!
//! The policy is deterministic given its inputs - no clock is read - so a backend that runs one
//! sequence across several ranks picks the same depth on every rank.
//!
//! **Attribution:** Online calibrated acceptance profiles paired with a step cost model for
//! adaptive speculative verification depth draw on prior art in
//! [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference)
//! (`helios/src/scheduler/speculation.rs`, `cost.rs`, `learning.rs`). See the root `NOTICE` file.

/// The modelled wall clock of one speculative step: a fixed part (the draft, the target's first
/// row, the head) and a part per verified draft. Only the differences between depths steer the
/// choice; `prior_tokens_per_ms` stands in for the sequence's rate before its first step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCost {
    /// A step that verifies no draft (ms).
    pub step_ms: f64,
    /// Each verified draft on top of it (ms).
    pub row_ms: f64,
    /// The token rate the first step assumes, before any step has been observed.
    pub prior_tokens_per_ms: f64,
}

impl StepCost {
    /// The modelled wall of a step that verifies `depth` drafts (ms).
    #[must_use]
    pub fn ms(&self, depth: usize) -> f64 {
        self.step_ms + self.row_ms * depth as f64
    }
}

/// The draft-probability bins the acceptance calibration counts in.
const CALIBRATION_EDGES: [f32; 7] = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99];
/// The pseudo-count of the calibration prior (each bin starts at the draft's own probability).
const CALIBRATION_PRIOR: f64 = 4.0;
const BINS: usize = CALIBRATION_EDGES.len() + 1;
/// The acceptances a bin needs before the policy stops exploring.
///
/// A policy that verifies nothing never sees an acceptance, so its calibration stays at the prior
/// and it goes on verifying nothing: the depth it decides on is the one that decides whether it
/// learns. The first steps therefore verify one draft - at worst a row of work - and the policy
/// takes over once it has evidence. Eight is enough to move a bin's calibration off the prior and
/// small enough that the exploration costs a sequence a few rows at most.
const EXPLORATION_STEPS: f64 = 8.0;

/// How many of the k drafts a step verifies. Adaptive picks the depth d that maximises
///
///   sum_{j <= d} prod_{i <= j} q_i  -  rate * cost.ms(d),
///
/// the expected accepted drafts less what the extra rows cost at the sequence's modelled token
/// rate, where q_i is the acceptance probability of draft i given the ones before it were
/// accepted: the draft's probability under the drafter, calibrated by the acceptances seen so far
/// in its bin. Under a sampler the probabilities are of the noised scores; the calibration learns
/// what they are worth. Fixed verifies every draft.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthPolicy {
    adaptive: bool,
    k: usize,
    min_depth: usize,
    cost: StepCost,
    accepted: [f64; BINS],
    seen: [f64; BINS],
    tokens: f64,
    ms: f64,
}

impl DepthPolicy {
    /// An adaptive policy over chains of up to `k` drafts.
    #[must_use]
    pub fn adaptive(k: usize, cost: StepCost) -> Self {
        Self {
            adaptive: true,
            k,
            min_depth: 0,
            cost,
            accepted: [0.0; BINS],
            seen: [0.0; BINS],
            tokens: 0.0,
            ms: 0.0,
        }
    }

    /// A policy that verifies all `k` drafts every step.
    #[must_use]
    pub fn fixed(k: usize, cost: StepCost) -> Self {
        Self {
            adaptive: false,
            ..Self::adaptive(k, cost)
        }
    }

    /// The same policy, never verifying fewer than `min_depth` drafts (capped at k).
    #[must_use]
    pub fn with_min_depth(self, min_depth: usize) -> Self {
        Self {
            min_depth: min_depth.min(self.k),
            ..self
        }
    }

    /// The policy the installed profile names, over chains of up to `k` drafts.
    #[must_use]
    pub fn from_profile(k: usize, cost: StepCost) -> Self {
        let speculation = &crate::active_profile().speculation;
        let policy = match speculation.depth_policy {
            Some(crate::profile::DepthMode::Fixed) => Self::fixed(k, cost),
            Some(crate::profile::DepthMode::Adaptive) | None => Self::adaptive(k, cost),
        };
        policy.with_min_depth(speculation.min_depth.unwrap_or(0))
    }

    /// Whether the depth follows the drafts' probabilities; a fixed policy verifies all k.
    #[must_use]
    pub fn is_adaptive(&self) -> bool {
        self.adaptive
    }

    /// The step cost the policy weighs depths with.
    #[must_use]
    pub fn cost(&self) -> StepCost {
        self.cost
    }

    fn bin(p: f32) -> usize {
        CALIBRATION_EDGES
            .iter()
            .take_while(|edge| p >= **edge)
            .count()
    }

    /// How many of the drafts whose probabilities are `probs` to verify.
    #[must_use]
    pub fn depth(&self, probs: &[f32]) -> usize {
        if !self.adaptive {
            return self.k;
        }

        let rate = if self.ms > 0.0 {
            self.tokens / self.ms
        } else {
            self.cost.prior_tokens_per_ms
        };
        let (mut best, mut best_score) = (0usize, -rate * self.cost.ms(0));
        let mut gain = 0.0f64;
        let chain = self.chain(probs);
        for (d, chained) in (1..).zip(&chain) {
            gain += chained;
            let score = gain - rate * self.cost.ms(d);
            if score > best_score {
                (best, best_score) = (d, score);
            }
        }
        let best = best.max(self.min_depth);
        // A policy that verifies nothing sees no acceptances, so its calibration stays at the prior
        // and it goes on verifying nothing: the depth it decides is the one that decides whether it
        // learns. While it has nothing sampled and its priors choose no drafts, it verifies one - a
        // row of work at worst - so the bin that draft's probability falls in can be learnt.
        if best == 0 && self.seen.iter().sum::<f64>() < EXPLORATION_STEPS {
            return usize::from(self.k > 0);
        }
        best
    }

    /// The calibrated probability that drafts 0..=j are all accepted, for each of the k drafts.
    #[must_use]
    pub fn chain(&self, probs: &[f32]) -> Vec<f64> {
        let mut chain = 1.0f64;
        probs
            .iter()
            .take(self.k)
            .map(|&p| {
                let b = Self::bin(p);
                chain *= (self.accepted[b] + CALIBRATION_PRIOR * f64::from(p))
                    / (self.seen[b] + CALIBRATION_PRIOR);
                chain
            })
            .collect()
    }

    /// A step verified `depth` drafts and accepted the first `m`: drafts 0..m were accepted and
    /// draft m (when verified) was not.
    pub fn observe(&mut self, probs: &[f32], depth: usize, m: usize) {
        for (i, p) in probs.iter().enumerate().take(depth) {
            let b = Self::bin(*p);
            self.seen[b] += 1.0;
            if i < m {
                self.accepted[b] += 1.0;
            } else {
                break;
            }
        }
        self.tokens += (m + 1) as f64;
        self.ms += self.cost.ms(depth);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_fresh_policy_explores_before_it_can_calibrate() {
        let cost = StepCost {
            step_ms: 40.0,
            row_ms: 6.0,
            prior_tokens_per_ms: 0.025,
        };
        // Confidences low enough that the priors alone would choose no drafts at all.
        let lows = [0.01f32; 7];
        let mut policy = DepthPolicy::adaptive(7, cost);
        assert_eq!(policy.depth(&lows), 1, "a fresh policy verifies one draft");
        // Once a bin has been sampled, the policy is free to decide the priors were right.
        for _ in 0..8 {
            policy.observe(&lows, 1, 0);
        }
        assert_eq!(
            policy.depth(&lows),
            0,
            "the calibration learned the drafts are not worth it"
        );
        // A fixed policy never explores: it verifies every draft it was given.
        assert_eq!(DepthPolicy::fixed(7, cost).depth(&lows), 7);
        // And a minimum depth is honoured while exploring.
        assert_eq!(
            DepthPolicy::adaptive(7, cost)
                .with_min_depth(3)
                .depth(&lows),
            3
        );
    }

    use super::*;

    const COST: StepCost = StepCost {
        step_ms: 27.0,
        row_ms: 4.0,
        prior_tokens_per_ms: 0.06,
    };

    #[test]
    fn a_step_costs_its_fixed_part_and_a_row_a_draft() {
        assert_eq!(COST.ms(0), 27.0);
        assert_eq!(COST.ms(5), 47.0);
    }

    #[test]
    fn a_fixed_policy_verifies_every_draft() {
        let policy = DepthPolicy::fixed(5, COST);
        assert!(!policy.is_adaptive());
        assert_eq!(policy.depth(&[0.0; 5]), 5);
    }

    #[test]
    fn confident_drafts_are_all_verified_and_unlikely_ones_none() {
        let mut policy = DepthPolicy::adaptive(5, COST);
        assert_eq!(policy.depth(&[0.99; 5]), 5);
        // Unlikely drafts are not worth verifying, but an unsampled bin cannot know that: the first
        // step verifies one, and a few rejections in its bin are what teach the policy to stop.
        assert_eq!(policy.depth(&[0.05; 5]), 1);
        for _ in 0..8 {
            policy.observe(&[0.05; 5], 1, 0);
        }
        assert_eq!(policy.depth(&[0.05; 5]), 0);
    }

    #[test]
    fn the_chain_stops_where_its_tail_costs_more_than_it_brings() {
        // 0.9 then 0.1: the second draft's chained 0.09 is below rate * row (0.24 at the prior).
        let policy = DepthPolicy::adaptive(5, COST);
        assert_eq!(policy.depth(&[0.9, 0.1, 0.9, 0.9, 0.9]), 1);
    }

    #[test]
    fn rejections_in_a_bin_teach_the_policy_to_stop_trusting_it() {
        let mut policy = DepthPolicy::adaptive(5, COST);
        let probs = [0.92_f32; 5];
        assert_eq!(policy.depth(&probs), 5);
        for _ in 0..200 {
            policy.observe(&probs, 5, 0);
        }
        assert_eq!(policy.depth(&probs), 0);
    }

    #[test]
    fn the_minimum_depth_holds_and_is_capped_at_k() {
        let policy = DepthPolicy::adaptive(5, COST).with_min_depth(3);
        assert_eq!(policy.depth(&[0.01; 5]), 3);
        let policy = DepthPolicy::adaptive(5, COST).with_min_depth(9);
        assert_eq!(policy.depth(&[0.01; 5]), 5);
    }
}
