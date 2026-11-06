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

//! The speculative acceptance rule and its accounting, written once.
//!
//! Every arm that drafts runs the same round: draft a chain after the anchor, verify the chain
//! and the anchor in one step, keep the leading drafts the model would have picked itself, and
//! take the model's own choice at the last kept row. The chain's rows stand in for one-token
//! steps, so the rule is what makes speculation exact: a draft is kept only when the verify row
//! that would have produced it chose the same token.
//!
//! Three arms had written this out separately - the GLM harness, Qwen4-Exp's device sequence and
//! DeepSeek's serving loop - and the copies differed in how they counted (`kept` against
//! `accepted`, a step counted only when it offered something). This module is the one rule and
//! the one set of counters; an arm supplies the device calls and nothing else.
//!
//! # The rule
//!
//! A chunk is the anchor followed by `n` drafts, `chunk[0..=n]`. The verify's picks are the
//! model's own choice at each row, `picks[0..=n]`, where `picks[i]` is the token that follows
//! `chunk[i]`. The anchor is always kept. Draft `i` is kept if and only if `picks[i - 1] ==
//! chunk[i]` **and** every draft before it was kept: after the first disagreement, every later
//! row's prefix holds a token the model did not choose, so its pick says nothing about the chain
//! the arm will actually emit. The kept count is therefore one plus the length of the leading
//! run of matches.
//!
//! Nothing here touches a device: [verdict] decides on tokens and [Speculator] decides on
//! confidences, so both are testable without a GPU and neither can drift with an arm's copy.

use std::collections::VecDeque;

use crate::backend::Speculation;
use crate::speculation::DepthPolicy;

/// How many rows of a verify the arm keeps: the anchor, then the leading drafts it agreed with.
///
/// `picks` may be shorter than `chunk` when an arm only has the picks it needs; a missing pick
/// ends the run, which is the conservative reading.
///
/// # Example
///
/// ```
/// use sparkflash_model::spec::verdict;
/// // The anchor and three drafts; the model chose all three, then its own token.
/// assert_eq!(verdict(&[7, 11, 13, 17], &[11, 13, 17, 19]), 4);
/// // The second draft was not the model's choice, so the third row's pick is not its subtree.
/// assert_eq!(verdict(&[7, 11, 13, 17], &[11, 99, 17, 19]), 2);
/// // A single-row step agrees with nothing and keeps its one row.
/// assert_eq!(verdict(&[7], &[11]), 1);
/// ```
#[must_use]
pub fn verdict(chunk: &[u32], picks: &[u32]) -> usize {
    let mut kept = 1;
    while kept < chunk.len() {
        match picks.get(kept - 1) {
            Some(&pick) if pick == chunk[kept] => kept += 1,
            _ => break,
        }
    }
    kept
}

/// A token a verify produced that the serving adapter has not been handed yet.
///
/// The adapter's `decode` is called once per token the request emits. A round emitted several,
/// so the ones after the first wait here until their turn; the model has already consumed them,
/// which is why they are never forwarded again.
#[derive(Debug, Clone, PartialEq)]
pub struct Pending {
    /// The token the adapter will feed back.
    pub token: u32,
    /// The token that follows it, sampled once so the request emits it.
    pub next_token: Option<u32>,
    /// The verify row that chose [Pending::next_token], which a logprob request scores.
    pub logits: Vec<f32>,
}

/// The tokens a round has produced but the adapter has not been handed.
///
/// A step's rows are the anchor and its drafts; the round's picks are the model's own choice at
/// each row, and the emitted tokens are those picks up to the kept count. The first goes back
/// with the step, the rest wait here (see [Pending]).
#[derive(Debug, Default, Clone)]
pub struct Queue {
    pending: VecDeque<Pending>,
}

impl Queue {
    /// An empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the adapter has been handed everything the last round produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// How many tokens are waiting.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// The token the adapter must feed next, if the last round left a chain.
    #[must_use]
    pub fn front(&self) -> Option<&Pending> {
        self.pending.front()
    }

    /// Forgets the queue, which a new prompt or a new request does.
    pub fn clear(&mut self) {
        self.pending.clear();
    }

    /// Takes the front entry, if it is the token the adapter just fed.
    ///
    /// `None` means the queue is empty or the adapter diverged; a caller that gets `Some` may
    /// return the entry's row as the step. Divergence is the caller's to diagnose: it means a
    /// token the verify did not produce reached the engine.
    pub fn pop(&mut self, token: u32) -> Option<Pending> {
        match self.pending.front() {
            Some(front) if front.token == token => self.pending.pop_front(),
            _ => None,
        }
    }

    /// Queues what a round produced after its first emitted token.
    ///
    /// `chunk` is the anchor followed by the drafts, `picks` the model's choice at each row,
    /// `rows` the logits those picks came from, and `kept` [verdict]'s answer. A token in
    /// `eos` ends the request, so the entry that would follow one carries `None`.
    pub fn extend_round(
        &mut self,
        chunk: &[u32],
        picks: &[u32],
        rows: &[Vec<f32>],
        kept: usize,
        eos: &[u32],
    ) {
        self.pending.clear();
        for (index, &token) in chunk.iter().enumerate().take(kept).skip(1) {
            let next_token = picks.get(index).copied().filter(|next| !eos.contains(next));
            self.pending.push_back(Pending {
                token,
                next_token,
                logits: rows.get(index).cloned().unwrap_or_default(),
            });
        }
    }
}

/// One sequence's speculative state: how deep to draft, and what the drafts were worth.
///
/// The arm owns the drafter; this owns the decision. A [DepthPolicy] may shorten a chain the
/// drafter already produced (a block drafter proposes a whole block and says how sure it is of
/// each token), and the counters the serving layer reports come from the same call that decides
/// the depth, so a report cannot describe a round that did not run.
#[derive(Debug, Clone)]
pub struct Speculator {
    max_depth: usize,
    policy: Option<DepthPolicy>,
    stats: Speculation,
}

impl Speculator {
    /// A speculator that verifies at most `max_depth` drafts a step.
    ///
    /// With a `policy` the depth is the policy's choice from the drafter's confidences; without
    /// one it is always `max_depth`, which is what a drafter with no confidences to offer (the
    /// MTP layer) needs.
    #[must_use]
    pub fn new(max_depth: usize, policy: Option<DepthPolicy>) -> Self {
        Self {
            max_depth,
            policy,
            stats: Speculation::default(),
        }
    }

    /// The most drafts this speculator verifies in one step.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// What this sequence's drafts have been worth so far.
    ///
    /// A request reports its own counts, so a serving adapter resets the speculator when it
    /// starts one ([Self::reset]).
    #[must_use]
    pub fn stats(&self) -> Speculation {
        self.stats
    }

    /// Starts a new request's accounting.
    ///
    /// A [DepthPolicy]'s calibration is kept: it describes how this drafter scores on this
    /// machine, which a new request does not change.
    pub fn reset(&mut self) {
        self.stats = Speculation::default();
    }

    /// How many of `confidence`'s drafts to verify, given `room` drafts the caller can still use.
    ///
    /// `room` is derived from the request's remaining output budget: a chain longer than the
    /// tokens left would verify rows whose results are thrown away.
    #[must_use]
    pub fn depth(&self, confidence: &[f32], room: usize) -> usize {
        let want = match self.policy.as_ref() {
            Some(policy) => policy.depth(confidence),
            None => self.max_depth,
        };
        want.min(room).min(self.max_depth)
    }

    /// Records one round: `offered` drafts verified, `kept - 1` of them accepted.
    ///
    /// A round that offered nothing is not a verify step, which is why a plain step's accounting
    /// and a speculative one's do not diverge: both count the tokens they produced.
    pub fn observe(&mut self, confidence: &[f32], offered: usize, kept: usize) {
        if offered > 0 {
            self.stats.verify_steps += 1;
        }
        self.stats.drafts_proposed += offered;
        self.stats.drafts_accepted += kept.saturating_sub(1).min(offered);
        if let Some(policy) = self.policy.as_mut() {
            // A step the policy gave no drafts is still a step of the drafted sequence: its cost
            // is the rate the next depth is weighed at.
            policy.observe(confidence, offered, kept.saturating_sub(1).min(offered));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculation::StepCost;

    const COST: StepCost = StepCost {
        step_ms: 40.0,
        row_ms: 3.0,
        prior_tokens_per_ms: 0.02,
    };

    #[test]
    fn the_verdict_keeps_the_leading_run_and_always_the_anchor() {
        assert_eq!(verdict(&[], &[]), 1);
        assert_eq!(verdict(&[7], &[9]), 1);
        assert_eq!(verdict(&[7], &[]), 1);
        assert_eq!(verdict(&[7, 11], &[11]), 2);
        assert_eq!(verdict(&[7, 11], &[12]), 1);
        assert_eq!(verdict(&[7, 11, 13, 17], &[11, 13, 17, 19]), 4);
        // The first disagreement ends the run even when a later row happens to match: that row's
        // prefix holds a token the model did not choose, so its pick belongs to another chain.
        assert_eq!(verdict(&[7, 11, 13, 17], &[11, 99, 13]), 2);
        // A missing pick is not an agreement.
        assert_eq!(verdict(&[7, 11, 13], &[11]), 2);
    }

    #[test]
    fn a_speculator_without_a_policy_drafts_its_maximum() {
        let speculator = Speculator::new(3, None);
        assert_eq!(speculator.depth(&[], 16), 3);
        assert_eq!(speculator.depth(&[0.9, 0.3, 0.1], 16), 3);
        assert_eq!(speculator.depth(&[0.9, 0.3, 0.1], 1), 1);
        assert_eq!(Speculator::new(0, None).depth(&[], 16), 0);
    }

    #[test]
    fn a_policy_chooses_the_depth_and_the_room_caps_it() {
        let mut speculator = Speculator::new(4, Some(DepthPolicy::fixed(4, COST)));
        // A fixed policy verifies all k of its chain.
        assert_eq!(speculator.depth(&[0.5; 2], 16), 4);
        assert_eq!(
            speculator.depth(&[0.5; 2], 2),
            2,
            "the request's room caps it"
        );
        speculator.observe(&[0.5; 2], 2, 2);
        assert_eq!(speculator.stats().verify_steps, 1);
        // A caller with no confidences still gets the maximum, and never more than max_depth.
        assert_eq!(speculator.depth(&[], 16), 4);
        assert_eq!(speculator.depth(&[], 0), 0);
    }

    #[test]
    fn the_counters_read_as_one_step_per_round_that_offered_something() {
        let mut speculator = Speculator::new(2, None);
        speculator.observe(&[], 0, 1);
        assert_eq!(
            speculator.stats().verify_steps,
            0,
            "a plain step verifies nothing"
        );
        speculator.observe(&[], 2, 3);
        speculator.observe(&[], 2, 1);
        let stats = speculator.stats();
        assert_eq!(stats.verify_steps, 2);
        assert_eq!(stats.drafts_proposed, 4);
        assert_eq!(stats.drafts_accepted, 2);
        speculator.reset();
        assert_eq!(speculator.stats(), Speculation::default());
    }

    fn row(value: f32) -> Vec<f32> {
        vec![value; 2]
    }

    #[test]
    fn a_round_queues_every_token_after_its_first() {
        let mut queue = Queue::new();
        assert!(queue.is_empty());
        // The anchor 7 and three drafts, all kept: the request emits 11, 13, 17 and then 19,
        // and the adapter will feed 11, 13 and 17 back to us.
        let chunk = [7, 11, 13, 17];
        let picks = [11, 13, 17, 19];
        let rows = [row(0.0), row(1.0), row(2.0), row(3.0)];
        queue.extend_round(&chunk, &picks, &rows, 4, &[]);
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.front().map(|pending| pending.token), Some(11));
        assert_eq!(
            queue.front().and_then(|pending| pending.next_token),
            Some(13)
        );
        // Each entry's logits are the row that chose its next token, which is what a logprob
        // request scores.
        assert_eq!(queue.front().map(|pending| pending.logits[0]), Some(1.0));

        let first = queue.pop(11).expect("the first queued token");
        assert_eq!(first.next_token, Some(13));
        assert!(
            queue.pop(99).is_none(),
            "a token the verify did not produce"
        );
        assert_eq!(queue.len(), 2, "a mismatched pop leaves the queue alone");
        assert_eq!(queue.pop(13).and_then(|entry| entry.next_token), Some(17));
        assert_eq!(queue.pop(17).and_then(|entry| entry.next_token), Some(19));
        assert!(queue.is_empty());
    }

    #[test]
    fn a_partial_keep_queues_only_the_kept_rows() {
        let mut queue = Queue::new();
        let chunk = [7, 11, 13, 17];
        let picks = [11, 99, 17, 19];
        let rows = [row(0.0), row(1.0), row(2.0), row(3.0)];
        // Only the first draft matched, so the request emits 11 and then the model's own 99.
        queue.extend_round(&chunk, &picks, &rows, 2, &[]);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.pop(11).and_then(|entry| entry.next_token), Some(99));
        assert!(queue.is_empty());
    }

    #[test]
    fn an_end_of_generation_token_carries_no_successor() {
        let mut queue = Queue::new();
        let rows = [row(0.0), row(1.0), row(2.0)];
        queue.extend_round(&[7, 11, 13], &[11, 13, 2], &rows, 3, &[2]);
        let first = queue.pop(11).expect("the first queued token");
        assert_eq!(first.next_token, Some(13));
        let last = queue.pop(13).expect("the second queued token");
        assert_eq!(last.next_token, None, "2 ends the request");
    }

    #[test]
    fn a_single_kept_row_leaves_nothing_queued() {
        let mut queue = Queue::new();
        let rows = [row(0.0), row(1.0)];
        queue.extend_round(&[7, 11], &[99, 42], &rows, 1, &[]);
        assert!(queue.is_empty());
    }

    #[test]
    fn an_over_long_acceptance_cannot_outrun_the_drafts_it_offered() {
        let mut speculator = Speculator::new(2, None);
        speculator.observe(&[], 1, 9);
        let stats = speculator.stats();
        assert_eq!(stats.drafts_accepted, 1);
        assert_eq!(stats.drafts_proposed, 1);
    }
}
