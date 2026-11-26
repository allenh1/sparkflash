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

//! Gumbel-max sampling with counter-based noise, shared by every arm that samples.
//!
//! The token at sequence index `t` is `argmax_v (logit_v / T + G(seed, t, v))` over the request's
//! candidate set, where `G` is a Gumbel variate drawn from a hash of (seed, t, v)
//! (`kernels/opt_sample_sm121.cuh` on the device, [`gumbel`] here). That is an exact sample of
//! softmax(logits / T) restricted to the candidates, and it needs no state: every rank draws the
//! same noise, and a verify and a drafter perturb with the same function, so a block of drafts is
//! accepted while each draft equals the target's sample at its index. The emitted token at an index
//! is then a function of the target's logits and the seed alone, whatever the drafter proposed:
//! speculative and plain decoding emit the same tokens for the same seed.
//!
//! Three modes, by request:
//! - greedy (temperature 0): the argmax;
//! - temperature only (top_p 1, top_k 0): the argmax of the perturbed scores over every token
//!   ([`GumbelSampler::pick_row`] on the host; DeepSeek adds the noise inside its head kernel);
//! - truncated (top_k or top_p): the cut is taken over a list of the largest logits with the row's
//!   maximum and normaliser - DeepSeek's ranks each keep their slice's [`TOPC`] largest
//!   (`opt_rows_topc`) and [`GumbelSampler::pick_truncated`] merges the exchanged slots; a whole
//!   row on the host goes through [`GumbelSampler::pick_row`].
//!
//! [`GumbelSampler::pick_truncated`] cuts a nucleus wider than the merged list at the list, which
//! only a near-flat distribution reaches; [`GumbelSampler::pick_row`] has the whole row and widens
//! to it instead, so its cut is always exact.

/// Candidates each rank keeps per row for the truncated cut (`OR_TOPC` in `opt_rows_sm121.cu`).
pub const TOPC: usize = 128;

/// Words per row in a rank's `opt_rows_topc` slot: max, normaliser, then TOPC (logit, id) pairs.
pub const TOPC_WORDS: usize = 2 + 2 * TOPC;

/// A bound past the largest gap between two Gumbel variates from [`gumbel`]: the uniform lies in
/// [2^-24, 1 - 2^-24], so G lies in about [-2.81, 16.64]. A token whose scaled logit trails the
/// row's best by more than this cannot win the perturbed argmax.
const GUMBEL_SPAN: f32 = 21.0;

/// splitmix64's finaliser, `opt_sample_mix`.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// G(seed, position, id), the host twin of `opt_sample_gumbel`: the same hash and uniform; the
/// logarithms may differ from the device's in the last ulp.
#[must_use]
pub fn gumbel(seed: u64, position: u32, id: u32) -> f32 {
    let counter = (u64::from(position) << 32) | u64::from(id);
    let h = mix(mix(seed) ^ counter);
    let u = (((h >> 41) as u32) as f32 + 0.5) * (1.0 / 8_388_608.0);
    -(-u.ln()).ln()
}

/// One request's sampling parameters, as the Gumbel paths apply them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelSampler {
    /// 0 is greedy.
    pub temperature: f32,
    /// 1 keeps the whole distribution.
    pub top_p: f32,
    /// 0 keeps every token.
    pub top_k: u32,
    /// The Gumbel stream's seed: two requests with the same seed draw the same noise.
    pub seed: u64,
}

/// The better of two (score, id) candidates: the higher score, the lower id on a tie.
fn better(best: Option<(f32, u32)>, (score, id): (f32, u32)) -> Option<(f32, u32)> {
    match best {
        Some((value, index)) if value > score || (value == score && index < id) => best,
        _ => Some((score, id)),
    }
}

impl GumbelSampler {
    /// The argmax: the harness's and the reference's rule.
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
    };

    /// True when the request is the plain argmax (a temperature of zero or below).
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// True when the candidate set is cut (top-k or top-p) and the host picks.
    pub fn truncated(&self) -> bool {
        !self.is_greedy() && (self.top_k > 0 || self.top_p < 1.0)
    }

    /// 1 / T, or 0 for greedy (the kernels' "no noise" value).
    pub fn inv_t(&self) -> f32 {
        if self.is_greedy() {
            0.0
        } else {
            1.0 / self.temperature
        }
    }

    /// The head kernel's noise arguments `(inv_t bits, seed)`: zero unless the device argmax is the
    /// sample (temperature only). The truncated mode takes raw logits for the host cut.
    pub fn head_noise(&self) -> (u64, u64) {
        if self.is_greedy() || self.truncated() {
            (0, 0)
        } else {
            (u64::from(self.inv_t().to_bits()), self.seed)
        }
    }

    /// The drafter's noise arguments `(inv_t bits, seed)`: the full tempered distribution in both
    /// sampling modes (a draft is only a proposal; the target's cut decides what is emitted).
    pub fn draft_noise(&self) -> (u64, u64) {
        if self.is_greedy() {
            (0, 0)
        } else {
            (u64::from(self.inv_t().to_bits()), self.seed)
        }
    }

    /// The sample for `row` from every rank's exchanged `opt_rows_topc` slot: `slots` is
    /// `[world][rows][TOPC_WORDS]` u32, `position` the sequence index the row's token lands at.
    /// Every rank runs this on the same bytes and returns the same id.
    ///
    /// `None` when the slots hold no candidate at all: no rank's slice contributed a pair, so
    /// there is nothing to cut or sample. `world` of zero is the shape that does it (a device
    /// exchange that names no peer). The caller refuses by name; a panic on the empty list instead
    /// takes down the walk server the caller runs in, and with it every later check of the run.
    #[must_use]
    pub fn pick_truncated(
        &self,
        slots: &[u32],
        rows: usize,
        row: usize,
        world: usize,
        position: u32,
    ) -> Option<u32> {
        let inv_t = self.inv_t();
        let slot = |rank: usize| {
            let at = (rank * rows + row) * TOPC_WORDS;
            &slots[at..at + TOPC_WORDS]
        };
        // The global maximum and normaliser from the per-slice pairs.
        let m = (0..world)
            .map(|rank| f32::from_bits(slot(rank)[0]))
            .fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = (0..world)
            .map(|rank| {
                let s = slot(rank);
                f32::from_bits(s[1]) * ((f32::from_bits(s[0]) - m) * inv_t).exp()
            })
            .sum();
        // Every token above the largest per-slice floor is in the merged list; keep those (and the
        // floor's ties the slices hold), best first, ties to the lower id.
        let mut floor = f32::NEG_INFINITY;
        let mut merged: Vec<(f32, u32)> = Vec::with_capacity(world * TOPC);
        for rank in 0..world {
            let s = slot(rank);
            let mut least = f32::INFINITY;
            for &[bits, id] in s[2..].as_chunks::<2>().0 {
                let logit = f32::from_bits(bits);
                least = least.min(logit);
                merged.push((logit, id));
            }
            floor = floor.max(least);
        }
        merged.retain(|&(logit, _)| logit >= floor);
        merged.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        if merged.is_empty() {
            return None;
        }
        Some(self.cut(&mut merged, m, z, floor, false, position).0)
    }

    /// The truncated cut over `merged` (best first, ties to the lower id) and the sample from the
    /// kept prefix. `m` and `z` are the whole row's maximum and normaliser, `floor` the least logit
    /// the list is known to hold every token above, `complete` whether it holds the whole row.
    ///
    /// The second value is whether the cut is the whole row's: the top-k set and the nucleus each
    /// end strictly above `floor` (a token outside the list at the floor's logit could otherwise
    /// belong in them), or the list is complete.
    fn cut(
        &self,
        merged: &mut Vec<(f32, u32)>,
        m: f32,
        z: f32,
        floor: f32,
        complete: bool,
        position: u32,
    ) -> (u32, bool) {
        let inv_t = self.inv_t();
        let listed = merged.len();
        if self.top_k > 0 {
            merged.truncate(self.top_k as usize);
        }
        let weight = |logit: f32| ((logit - m) * inv_t).exp();
        // top_p over the top-k renormalised distribution (vLLM's order), else over the whole one.
        let mass = if self.top_k > 0 {
            merged.iter().map(|&(logit, _)| weight(logit)).sum()
        } else {
            z
        };
        let mut kept = 0usize;
        let mut cum = 0.0f32;
        let mut reached = false;
        for &(logit, _) in merged.iter() {
            kept += 1;
            cum += weight(logit) / mass;
            if cum >= self.top_p {
                reached = true;
                break;
            }
        }
        let kept = kept.max(1);
        let id = merged[..kept]
            .iter()
            .map(|&(logit, id)| (logit * inv_t + gumbel(self.seed, position, id), id))
            .fold(None, better)
            .map_or(0, |(_, id)| id);
        let top_k_whole = self.top_k == 0
            || (listed >= self.top_k as usize && merged[self.top_k as usize - 1].0 > floor);
        let nucleus_whole = merged[kept - 1].0 > floor && (reached || self.top_k > 0);
        (id, complete || (top_k_whole && nucleus_whole))
    }

    /// The sample from one whole row of logits, whose token lands at sequence index `position`:
    /// the argmax (lowest id on a tie) when greedy; with a temperature alone, the perturbed argmax
    /// over every token; truncated, the cut over the row's largest logits, widened to the whole
    /// row when they do not decide it. Linear in the row but for a cut that has to widen.
    #[must_use]
    pub fn pick_row(&self, logits: &[f32], position: u32) -> u32 {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if self.is_greedy() {
            return logits.iter().position(|&logit| logit == m).unwrap_or(0) as u32;
        }
        let inv_t = self.inv_t();
        if !self.truncated() {
            // Only a token within the noise's span of the best scaled logit can win.
            let least = m * inv_t - GUMBEL_SPAN;
            return logits
                .iter()
                .enumerate()
                .filter(|&(_, &logit)| logit * inv_t >= least)
                .map(|(id, &logit)| {
                    let id = id as u32;
                    (logit.mul_add(inv_t, gumbel(self.seed, position, id)), id)
                })
                .fold(None, better)
                .map_or(0, |(_, id)| id);
        }
        // One pass for the normaliser and a histogram of the weights by distance from the best
        // scaled logit; the cut's boundary lies in the first bucket whose running mass (or
        // count, under a top-k) reaches it, so only the buckets up to the next one are sorted.
        let mut z = 0.0f32;
        let mut mass = [0.0f64; BUCKETS];
        let mut count = [0usize; BUCKETS];
        for &logit in logits {
            let weight = ((logit - m) * inv_t).exp();
            z += weight;
            let at = bucket(logit, m, inv_t);
            mass[at] += f64::from(weight);
            count[at] += 1;
        }
        let k = self.top_k as usize;
        let target = (f64::from(self.top_p) + 1e-4) * f64::from(z);
        let mut last = BUCKETS - 1;
        let (mut held_mass, mut held) = (0.0f64, 0usize);
        for at in 0..BUCKETS {
            held_mass += mass[at];
            held += count[at];
            if (k > 0 && held >= k) || (k == 0 && held_mass >= target) {
                last = (at + 1).min(BUCKETS - 1);
                break;
            }
        }
        // Every token left out is below every token kept (the bucket falls as the logit does),
        // so the list holds every token above the largest left out.
        let mut floor = f32::NEG_INFINITY;
        let mut merged: Vec<(f32, u32)> = Vec::new();
        for (id, &logit) in logits.iter().enumerate() {
            if bucket(logit, m, inv_t) <= last {
                merged.push((logit, id as u32));
            } else {
                floor = floor.max(logit);
            }
        }
        let complete = merged.len() == logits.len();
        merged.sort_by(best_first);
        let (id, whole) = self.cut(&mut merged, m, z, floor, complete, position);
        if whole {
            return id;
        }
        let mut all: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .map(|(id, &logit)| (logit, id as u32))
            .collect();
        all.sort_by(best_first);
        self.cut(&mut all, m, z, f32::NEG_INFINITY, true, position)
            .0
    }
}

/// Histogram buckets [GumbelSampler::pick_row] sorts by: an eighth of a nat each, the last
/// holding everything 64 nats or more below the best (each of those weighs under e^-64).
const BUCKETS: usize = 512;

/// A logit's bucket: its scaled distance from the row's best, `m`. Never rises with the logit.
fn bucket(logit: f32, m: f32, inv_t: f32) -> usize {
    let distance = (m - logit) * inv_t * 8.0;
    if distance >= (BUCKETS - 1) as f32 {
        BUCKETS - 1
    } else if distance > 0.0 {
        distance as usize
    } else {
        0
    }
}

/// Best first: the higher logit, the lower id on a tie.
fn best_first(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One rank's slot from its slice of logits (the kernel's contract, by sorting).
    fn slot(logits: &[f32], id_base: u32, inv_t: f32) -> Vec<u32> {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|&l| ((l - m) * inv_t).exp()).sum();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let mut out = vec![m.to_bits(), z.to_bits()];
        for &v in order.iter().take(TOPC) {
            out.extend_from_slice(&[logits[v].to_bits(), id_base + v as u32]);
        }
        out
    }

    #[test]
    fn gumbel_is_a_pure_function_in_range() {
        let a = gumbel(7, 100, 5);
        assert_eq!(a.to_bits(), gumbel(7, 100, 5).to_bits());
        assert_ne!(a.to_bits(), gumbel(8, 100, 5).to_bits());
        for id in 0..10_000 {
            let g = gumbel(3, 9, id);
            assert!(g.is_finite());
            assert!((-3.0..17.0).contains(&g), "{g}");
        }
    }

    #[test]
    fn top_k_one_is_the_argmax() {
        let a: Vec<f32> = (0..300).map(|v| (v as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..300).map(|v| (v as f32 * 0.11).cos() * 1.5).collect();
        let mut slots = slot(&a, 0, 1.0);
        slots.extend(slot(&b, 300, 1.0));
        let best = a
            .iter()
            .chain(&b)
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |acc, (i, &l)| if l > acc.1 { (i, l) } else { acc },
            );
        let sampler = GumbelSampler {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            seed: 42,
        };
        let row: Vec<f32> = a.iter().chain(&b).copied().collect();
        for position in 0..32 {
            assert_eq!(
                sampler.pick_truncated(&slots, 1, 0, 2, position),
                Some(best.0 as u32)
            );
            assert_eq!(sampler.pick_row(&row, position), best.0 as u32);
        }
    }

    /// A run whose device exchange names no peer hands the sampler no slice at all. That used to
    /// index an empty list (the panic took the walk server down and every later check with it); it
    /// is now `None`, so the caller refuses with the world it was given.
    #[test]
    fn a_world_of_no_slices_has_no_candidate_to_pick() {
        let sampler = GumbelSampler {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: 1234,
        };
        assert_eq!(sampler.pick_truncated(&[], 1, 0, 0, 0), None);
    }

    #[test]
    fn top_p_keeps_only_the_nucleus() {
        // One dominant token against 399 at logit 0: p(top) = e^10 / (e^10 + 399) ~ 0.982 at T = 1,
        // so top_p 0.9 keeps it alone.
        let mut a = vec![0.0f32; 200];
        a[17] = 10.0;
        let b = vec![0.0f32; 200];
        let mut slots = slot(&a, 0, 1.0);
        slots.extend(slot(&b, 200, 1.0));
        let sampler = GumbelSampler {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            seed: 1,
        };
        let row: Vec<f32> = a.iter().chain(&b).copied().collect();
        for position in 0..64 {
            assert_eq!(sampler.pick_truncated(&slots, 1, 0, 2, position), Some(17));
            assert_eq!(sampler.pick_row(&row, position), 17);
        }
        // With the whole distribution kept, the other tokens are drawn too.
        let wide = GumbelSampler {
            top_p: 1.0,
            top_k: 400,
            ..sampler
        };
        let picks: std::collections::HashSet<u32> = (0..512)
            .filter_map(|p| wide.pick_truncated(&slots, 1, 0, 2, p))
            .collect();
        assert!(picks.len() > 1);
    }

    /// The cut and sample over the whole row sorted, the definition [GumbelSampler::pick_row]
    /// must equal.
    fn by_sorting(sampler: &GumbelSampler, logits: &[f32], position: u32) -> u32 {
        let inv_t = sampler.inv_t();
        if !sampler.truncated() {
            return logits
                .iter()
                .enumerate()
                .map(|(id, &l)| {
                    (
                        l.mul_add(inv_t, gumbel(sampler.seed, position, id as u32)),
                        id as u32,
                    )
                })
                .fold(None, better)
                .map_or(0, |(_, id)| id);
        }
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|&l| ((l - m) * inv_t).exp()).sum();
        let mut all: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .map(|(id, &l)| (l, id as u32))
            .collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        sampler
            .cut(&mut all, m, z, f32::NEG_INFINITY, true, position)
            .0
    }

    /// A row of bf16-rounded logits (many ties), sharp or flat by `scale`.
    fn row(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|v| {
                let h = mix(seed ^ (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let u = (h >> 40) as f32 / (1u64 << 24) as f32;
                // A heavy upper tail so the nucleus is sometimes small, sometimes wide.
                let x = scale * (-(1.0 - u).ln()).ln().max(-6.0);
                f32::from_bits(x.to_bits() & 0xFFFF_0000)
            })
            .collect()
    }

    #[test]
    fn a_whole_row_pick_equals_the_sorted_definition() {
        let mut widened = 0;
        for case in 0..60u64 {
            let scale = [0.05, 0.5, 1.5, 4.0][case as usize % 4];
            let logits = row(case, 5_000, scale);
            for &(temperature, top_p, top_k) in &[
                (1.0, 0.95, 0),
                (0.7, 0.8, 0),
                (1.0, 1.0, 50),
                (1.0, 0.9, 300),
                (1.3, 0.5, 20),
                (1.0, 1.0, 0),
                (0.6, 1.0, 0),
            ] {
                let sampler = GumbelSampler {
                    temperature,
                    top_p,
                    top_k,
                    seed: case * 31 + 7,
                };
                for position in [0u32, 9, 1000] {
                    assert_eq!(
                        sampler.pick_row(&logits, position),
                        by_sorting(&sampler, &logits, position),
                        "case {case} T {temperature} p {top_p} k {top_k} at {position}"
                    );
                }
                if sampler.truncated() {
                    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let z: f32 = logits.iter().map(|&l| ((l - m) / temperature).exp()).sum();
                    let mut list: Vec<(f32, u32)> = logits
                        .iter()
                        .enumerate()
                        .map(|(id, &l)| (l, id as u32))
                        .collect();
                    list.sort_by(best_first);
                    let floor = list[TOPC.max(top_k as usize)].0;
                    list.retain(|&(l, _)| l > floor);
                    if !sampler.cut(&mut list, m, z, floor, false, 0).1 {
                        widened += 1;
                    }
                }
            }
        }
        // The flat rows widen; the sharp ones decide on the list.
        assert!(widened > 0 && widened < 60 * 5, "{widened}");
    }

    #[test]
    fn a_greedy_row_pick_is_the_first_argmax() {
        let logits = [0.5f32, 2.0, -1.0, 2.0];
        assert_eq!(GumbelSampler::GREEDY.pick_row(&logits, 3), 1);
    }
}
