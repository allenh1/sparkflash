//! The serving sampler on the device path: Gumbel-max sampling with counter-based noise.
//!
//! The token at sequence index `t` is `argmax_v (logit_v / T + G(seed, t, v))` over the request's
//! candidate set, where `G` is a Gumbel variate drawn from a hash of (seed, t, v)
//! (`kernels/opt_sample_sm121.cuh` on the device, [`gumbel`] here). That is an exact sample of
//! softmax(logits / T) restricted to the candidates, and it needs no state: every rank draws the
//! same noise, and the verify head and the drafter perturb with the same function, so a block of
//! drafts is accepted while each draft equals the target's sample at its index. The emitted token at
//! an index is then a function of the target's logits and the seed alone, whatever the drafter
//! proposed: speculative and plain decoding emit the same tokens for the same seed.
//!
//! Three modes, by request:
//! - greedy (temperature 0): the head's argmax, the unchanged path;
//! - temperature only (top_p 1, top_k 0): the noise is added inside the head kernel and the
//!   cross-rank argmax picks the sample;
//! - truncated (top_k or top_p): the head writes raw logits, each rank keeps its slice's
//!   [`TOPC`] largest with its slice normaliser (`opt_rows_topc`), the slots are exchanged, and every
//!   rank cuts and samples on the host ([`Sampler::pick_truncated`]) from the same bytes.
//!
//! The truncated cut is exact while the nucleus fits in the merged candidate list (at least
//! [`TOPC`] tokens are guaranteed); a nucleus wider than that is cut at the list, which only a
//! near-flat distribution reaches.

/// Candidates each rank keeps per row for the truncated cut (`OR_TOPC` in `opt_rows_sm121.cu`).
pub(super) const TOPC: usize = 128;

/// Words per row in a rank's `opt_rows_topc` slot: max, normaliser, then TOPC (logit, id) pairs.
pub(super) const TOPC_WORDS: usize = 2 + 2 * TOPC;

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
pub(super) fn gumbel(seed: u64, position: u32, id: u32) -> f32 {
    let counter = (u64::from(position) << 32) | u64::from(id);
    let h = mix(mix(seed) ^ counter);
    let u = (((h >> 41) as u32) as f32 + 0.5) * (1.0 / 8_388_608.0);
    -(-u.ln()).ln()
}

/// One request's sampling parameters, as the device path applies them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sampler {
    /// 0 is greedy.
    pub(crate) temperature: f32,
    /// 1 keeps the whole distribution.
    pub(crate) top_p: f32,
    /// 0 keeps every token.
    pub(crate) top_k: u32,
    pub(crate) seed: u64,
}

impl Sampler {
    /// The argmax: the harness's and the reference's rule.
    pub(crate) const GREEDY: Self = Self {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
    };

    pub(crate) fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// True when the candidate set is cut (top-k or top-p) and the host picks.
    pub(crate) fn truncated(&self) -> bool {
        !self.is_greedy() && (self.top_k > 0 || self.top_p < 1.0)
    }

    /// 1 / T, or 0 for greedy (the kernels' "no noise" value).
    pub(crate) fn inv_t(&self) -> f32 {
        if self.is_greedy() {
            0.0
        } else {
            1.0 / self.temperature
        }
    }

    /// The head kernel's noise arguments `(inv_t bits, seed)`: zero unless the device argmax is the
    /// sample (temperature only). The truncated mode takes raw logits for the host cut.
    pub(crate) fn head_noise(&self) -> (u64, u64) {
        if self.is_greedy() || self.truncated() {
            (0, 0)
        } else {
            (u64::from(self.inv_t().to_bits()), self.seed)
        }
    }

    /// The drafter's noise arguments `(inv_t bits, seed)`: the full tempered distribution in both
    /// sampling modes (a draft is only a proposal; the target's cut decides what is emitted).
    pub(crate) fn draft_noise(&self) -> (u64, u64) {
        if self.is_greedy() {
            (0, 0)
        } else {
            (u64::from(self.inv_t().to_bits()), self.seed)
        }
    }

    /// The sample for `row` from every rank's exchanged `opt_rows_topc` slot: `slots` is
    /// `[world][rows][TOPC_WORDS]` u32, `position` the sequence index the row's token lands at.
    /// Every rank runs this on the same bytes and returns the same id.
    pub(super) fn pick_truncated(
        &self,
        slots: &[u32],
        rows: usize,
        row: usize,
        world: usize,
        position: u32,
    ) -> u32 {
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
        for &(logit, _) in &merged {
            kept += 1;
            cum += weight(logit) / mass;
            if cum >= self.top_p {
                break;
            }
        }
        merged[..kept.max(1)]
            .iter()
            .map(|&(logit, id)| (logit * inv_t + gumbel(self.seed, position, id), id))
            .fold(None::<(f32, u32)>, |best, (score, id)| match best {
                Some((value, index)) if value > score || (value == score && index < id) => best,
                _ => Some((score, id)),
            })
            .map_or(0, |(_, id)| id)
    }
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
            assert!(gumbel(3, 9, id).is_finite());
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
        let sampler = Sampler {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            seed: 42,
        };
        for position in 0..32 {
            assert_eq!(
                sampler.pick_truncated(&slots, 1, 0, 2, position),
                best.0 as u32
            );
        }
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
        let sampler = Sampler {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            seed: 1,
        };
        for position in 0..64 {
            assert_eq!(sampler.pick_truncated(&slots, 1, 0, 2, position), 17);
        }
        // With the whole distribution kept, the other tokens are drawn too.
        let wide = Sampler {
            top_p: 1.0,
            top_k: 400,
            ..sampler
        };
        let picks: std::collections::HashSet<u32> = (0..512)
            .map(|p| wide.pick_truncated(&slots, 1, 0, 2, p))
            .collect();
        assert!(picks.len() > 1);
    }
}
