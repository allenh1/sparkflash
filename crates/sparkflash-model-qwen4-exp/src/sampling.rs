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

//! The sampling the served arm applies to the logits row it already has on the host.
//!
//! The device forward produces one f32 logits row per step and copies it back; the penalties,
//! temperature, top-k, top-p and the draw are therefore host arithmetic over a row the step
//! already paid for, not a reason to touch the device again. The order of the stages is vLLM's
//! (`_apply_penalties`, then temperature, then top-k, then top-p, then a multinomial draw), so a
//! request that names a temperature behaves the way it does against the reference server.
//!
//! A temperature of zero is greedy and never consumes random numbers, which is what lets the
//! default path keep its exact argmax behaviour.

use std::collections::{HashMap, HashSet};

use sparkflash_model::Sampling;
use sparkflash_weights::linalg::finite_argmax;

use crate::ngram::splitmix64;

/// Token history and the draw state one sequence carries.
///
/// The history is every token the sequence has consumed, the prompt's first. vLLM's repetition
/// penalty covers all of it; its presence and frequency penalties count only the generated tokens
/// after the prompt, the way the OpenAI API defines them.
#[derive(Debug, Clone, Default)]
pub struct Sampler {
    history: Vec<u32>,
    prompt_len: usize,
    draws: u64,
    seed: u64,
}

impl Sampler {
    /// A sampler for a sequence that has consumed nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The tokens this sequence has consumed, oldest first.
    #[must_use]
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Record a prompt. It is recorded before its logits are used, so the first sample already
    /// sees it, and every token consumed so far counts as prompt to the penalties.
    pub fn record_prompt(&mut self, tokens: &[u32]) {
        self.history.extend_from_slice(tokens);
        self.prompt_len = self.history.len();
    }

    /// Record a generated token.
    pub fn record(&mut self, token: u32) {
        self.history.push(token);
    }

    /// Forget every token past the first `len`: a speculative verify records the drafts it
    /// agreed with only while it chooses the rows after them, and the decode steps that emit
    /// those tokens record them again.
    pub fn truncate(&mut self, len: usize) {
        self.history.truncate(len);
        self.prompt_len = self.prompt_len.min(len);
    }

    /// Reset the history for a new request on the same session.
    pub fn reset(&mut self) {
        self.history.clear();
        self.prompt_len = 0;
        self.draws = 0;
    }

    /// Choose the next token from one logits row.
    ///
    /// # Panics
    /// When the row is empty; a model with an empty vocabulary cannot choose a token.
    #[must_use]
    pub fn choose(&mut self, logits: &[f32], sampling: &Sampling) -> u32 {
        assert!(!logits.is_empty(), "cannot sample from an empty logit row");
        let greedy = sampling.temperature.is_nan() || sampling.temperature <= 0.0;
        let generated = &self.history[self.prompt_len..];
        if greedy && penalties_are_neutral(sampling, &self.history, generated) {
            // The released default needs no rewritten copy of the row to pick its token.
            return argmax(logits);
        }
        let mut scores: Vec<f32> = logits.to_vec();
        // Penalties apply to a greedy request too: vLLM's penalties rewrite the row before the
        // temperature is even considered, so a repetition penalty at temperature 0 changes which
        // token wins. Only the draw is skipped.
        apply_penalties(&mut scores, &self.history, generated, sampling);
        if greedy {
            // Greedy: the path that must not consume a draw.
            return argmax(&scores);
        }
        if sampling.seed != 0 {
            self.seed = sampling.seed;
        }
        let temperature = sampling.temperature;
        for value in &mut scores {
            *value /= temperature;
        }
        apply_top_k(&mut scores, sampling.top_k);
        apply_top_p(&mut scores, sampling.top_p);
        let chosen = self.draw(&scores);
        self.draws = self.draws.wrapping_add(1);
        chosen
    }

    /// Draw one id from a row that has already been filtered, in f64.
    ///
    /// The exponentiation is done relative to the row maximum, so a row of large logits cannot
    /// overflow to infinity and hand every token the same weight.
    fn draw(&mut self, scores: &[f32]) -> u32 {
        let mut max = f32::NEG_INFINITY;
        for value in scores {
            if *value > max {
                max = *value;
            }
        }
        if !max.is_finite() {
            return argmax(scores);
        }
        let mut total = 0.0f64;
        let mut weights: Vec<f64> = Vec::with_capacity(scores.len());
        for value in scores {
            if value.is_finite() {
                let weight = ((value - max) as f64).exp();
                weights.push(weight);
                total += weight;
            } else {
                weights.push(0.0);
            }
        }
        if total <= 0.0 || total.is_nan() {
            return argmax(scores);
        }
        self.seed = splitmix64(self.seed.wrapping_add(self.draws ^ 0x9E37_79B9_7F4A_7C15));
        // One uniform in [0, 1) from the top 24 bits, the same construction the Engram hash draw
        // uses for its multipliers.
        let uniform = ((self.seed >> 40) as f64) / ((1u64 << 24) as f64);
        let mut remaining = uniform * total;
        for (index, weight) in weights.iter().enumerate() {
            if *weight <= 0.0 {
                continue;
            }
            remaining -= weight;
            if remaining <= 0.0 {
                return index as u32;
            }
        }
        argmax(scores)
    }
}

/// The index of the largest value, first one on a tie.
#[must_use]
pub fn argmax(values: &[f32]) -> u32 {
    if let Some(index) = finite_argmax(values) {
        return index as u32;
    }
    // A NaN never wins and a row with nothing above negative infinity answers 0.
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best as u32
}

/// Whether the penalties leave every score as it is: the repetition penalty is neutral or there
/// is no consumed token, and the presence and frequency penalties are zero or nothing has been
/// generated yet.
fn penalties_are_neutral(sampling: &Sampling, history: &[u32], generated: &[u32]) -> bool {
    let repetition = sampling.repetition_penalty != 1.0 && !history.is_empty();
    let additive = (sampling.presence_penalty != 0.0 || sampling.frequency_penalty != 0.0)
        && !generated.is_empty();
    !repetition && !additive
}

/// vLLM's `apply_penalties`: the repetition penalty over every consumed token, then the frequency
/// and presence penalties over the generated ones only.
fn apply_penalties(scores: &mut [f32], history: &[u32], generated: &[u32], sampling: &Sampling) {
    if penalties_are_neutral(sampling, history, generated) {
        return;
    }
    let repetition = sampling.repetition_penalty;
    if repetition != 1.0 {
        let seen: HashSet<u32> = history.iter().copied().collect();
        for token in seen {
            if let Some(value) = scores.get_mut(token as usize) {
                *value = if *value > 0.0 {
                    *value / repetition
                } else {
                    *value * repetition
                };
            }
        }
    }
    let presence = sampling.presence_penalty;
    let frequency = sampling.frequency_penalty;
    if presence == 0.0 && frequency == 0.0 {
        return;
    }
    let mut counts: HashMap<u32, usize> = HashMap::with_capacity(generated.len());
    for token in generated {
        *counts.entry(*token).or_insert(0) += 1;
    }
    for (token, count) in counts {
        if let Some(value) = scores.get_mut(token as usize) {
            // vLLM subtracts the frequency term and then the presence term, in that order.
            *value -= frequency * count as f32;
            *value -= presence;
        }
    }
}

/// Keep the k highest scores (`0` and negative values disable the filter, so does `k >= len`).
fn apply_top_k(scores: &mut [f32], top_k: i32) {
    if top_k <= 0 || top_k as usize >= scores.len() {
        return;
    }
    let mut sorted: Vec<f32> = scores.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let threshold = sorted[sorted.len() - top_k as usize];
    for value in scores {
        if *value < threshold {
            *value = f32::NEG_INFINITY;
        }
    }
}

/// Nucleus filter: keep the shortest prefix of the descending distribution whose mass reaches
/// top_p, always keeping at least one token.
fn apply_top_p(scores: &mut [f32], top_p: f32) {
    if top_p.is_nan() || top_p >= 1.0 {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for value in scores.iter() {
        if *value > max {
            max = *value;
        }
    }
    if !max.is_finite() {
        return;
    }
    let mut probabilities: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(index, value)| {
            (
                index,
                if value.is_finite() {
                    ((value - max) as f64).exp()
                } else {
                    0.0
                },
            )
        })
        .collect();
    let total: f64 = probabilities.iter().map(|(_, weight)| weight).sum();
    if total <= 0.0 || total.is_nan() {
        return;
    }
    probabilities.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    let mut cumulative = 0.0f64;
    let mut keep = 0usize;
    for (index, weight) in &probabilities {
        cumulative += weight / total;
        keep += 1;
        if cumulative >= f64::from(top_p) {
            break;
        }
        let _ = index;
    }
    // Everything past the prefix is dropped.
    for (index, _) in probabilities.iter().skip(keep) {
        scores[*index] = f32::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling(temperature: f32) -> Sampling {
        Sampling {
            temperature,
            ..Sampling::default()
        }
    }

    #[test]
    fn a_greedy_choice_is_the_argmax_and_consumes_no_draw() {
        let mut sampler = Sampler::new();
        sampler.record(1);
        let logits = [0.1f32, 3.0, -1.0, 2.0];
        assert_eq!(sampler.choose(&logits, &sampling(0.0)), 1);
        assert_eq!(sampler.draws, 0);
    }

    #[test]
    fn top_k_one_is_the_argmax_even_with_a_temperature() {
        let mut sampler = Sampler::new();
        let logits = [0.1f32, 3.0, -1.0, 2.0];
        let mut request = sampling(1.5);
        request.top_k = 1;
        assert_eq!(sampler.choose(&logits, &request), 1);
    }

    #[test]
    fn a_nucleus_of_almost_zero_keeps_only_the_best_token() {
        let mut sampler = Sampler::new();
        let logits = [0.0f32, 0.5, 4.0, 1.0];
        let mut request = sampling(1.0);
        request.top_p = 1e-6;
        assert_eq!(sampler.choose(&logits, &request), 2);
    }

    #[test]
    fn the_repetition_penalty_pushes_a_seen_token_below_an_unseen_one() {
        // The token is only in the prompt: vLLM's repetition penalty covers the prompt too.
        let mut sampler = Sampler::new();
        sampler.record_prompt(&[0]);
        let logits = [2.0f32, 1.9];
        assert_eq!(sampler.choose(&logits, &sampling(0.0)), 0);
        let mut request = sampling(0.0);
        request.repetition_penalty = 8.0;
        assert_eq!(sampler.choose(&logits, &request), 1);
    }

    #[test]
    fn a_seed_makes_a_temperature_draw_reproducible() {
        let logits = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut request = sampling(1.0);
        request.seed = 42;
        let mut first = Sampler::new();
        let mut second = Sampler::new();
        let left: Vec<u32> = (0..16).map(|_| first.choose(&logits, &request)).collect();
        let right: Vec<u32> = (0..16).map(|_| second.choose(&logits, &request)).collect();
        assert_eq!(left, right, "the same seed must draw the same ids");
        // A flat row over six tokens drawing only one id sixteen times would be a broken draw.
        let distinct: std::collections::BTreeSet<u32> = left.iter().copied().collect();
        assert!(distinct.len() > 1, "the draw never moved: {distinct:?}");
    }

    #[test]
    fn presence_and_frequency_penalties_are_ordered_the_way_vllm_orders_them() {
        // Token 0 seen twice: a repetition penalty divides it, presence subtracts once and
        // frequency subtracts twice. With a penalty large enough the unseen token wins.
        let mut sampler = Sampler::new();
        sampler.record(0);
        sampler.record(0);
        let logits = [1.0f32, 0.5];
        let mut request = sampling(0.0);
        request.presence_penalty = 0.5;
        request.frequency_penalty = 0.5;
        assert_eq!(sampler.choose(&logits, &request), 1);
    }

    #[test]
    fn presence_and_frequency_penalties_count_only_generated_tokens() {
        let logits = [1.0f32, 0.5];
        let mut request = sampling(0.0);
        request.presence_penalty = 0.5;
        request.frequency_penalty = 0.5;
        // Token 0 in the prompt, twice: neither penalty sees it, so it still wins.
        let mut sampler = Sampler::new();
        sampler.record_prompt(&[0, 0]);
        assert_eq!(sampler.choose(&logits, &request), 0);
        // Generated once: 1.0 - 0.5 * 1 - 0.5 = 0.0, below the unseen token.
        sampler.record(0);
        assert_eq!(sampler.choose(&logits, &request), 1);
        // A verify that rolls the generated token back leaves the prompt boundary where it was.
        sampler.truncate(2);
        assert_eq!(sampler.choose(&logits, &request), 0);
        // A new request starts with no prompt at all.
        sampler.reset();
        sampler.record(0);
        assert_eq!(sampler.choose(&logits, &request), 1);
    }

    #[test]
    fn the_frequency_term_is_subtracted_before_the_presence_term() {
        // In f32, 1.0 - 2^-25 is halfway between 1.0 and the float below it and rounds to 1.0 (to
        // even), so (1.0 - 2^-25) - 1.0 is 0.0 while 1.0 - 1.0 - 2^-25 is -2^-25: only vLLM's order
        // leaves the seen token tied with an unseen 0.0, and argmax picks the first on a tie.
        let tiny = f32::EPSILON / 4.0;
        let logits = [1.0f32, 0.0];
        let mut request = sampling(0.0);
        request.frequency_penalty = tiny;
        request.presence_penalty = 1.0;
        let mut sampler = Sampler::new();
        sampler.record(0);
        assert_eq!(sampler.choose(&logits, &request), 0);
    }
}
