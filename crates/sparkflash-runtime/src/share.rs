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

//! Which drafts a shared speculative step verifies.
//!
//! Several sequences verify their drafts in one walk. The walk has a row budget, and a draft is
//! worth its row only while the tokens it is expected to add beat what the row costs at the step's
//! token rate. [share_rows] gives each row to the draft likeliest accepted, then keeps the count
//! of rows with the most expected tokens a modelled millisecond. The model is the arm's
//! [StepWall]: DeepSeek's measured two-band line ([BandWall]), or for a mixture of experts whose
//! rows rarely share an expert, the expected union of the experts the rows pick
//! ([ExpertUnionWall]).
//!
//! This was DeepSeek's `share_rows` (`tp4_forward/batch.rs`); it moved here unchanged so every
//! drafting arm shares one decision.

/// What a walk costs: `seqs` sequences, `rows` rows in all (every anchor plus the drafts verified
/// with it), in modelled milliseconds. Only ratios matter to [share_rows].
pub trait StepWall {
    /// The modelled wall of one walk.
    fn ms(&self, seqs: usize, rows: usize) -> f64;
}

/// A linear wall in two bands, `a0 + a1 * rows` up to `band_rows` rows and `b0 + b1 * rows`
/// past them, plus `seq_ms` for each sequence past the first (DeepSeek's fitted step model).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandWall {
    /// The first band's intercept, in modelled milliseconds.
    pub a0: f64,
    /// The first band's slope a row, in modelled milliseconds.
    pub a1: f64,
    /// The second band's intercept.
    pub b0: f64,
    /// The second band's slope a row.
    pub b1: f64,
    /// The wall each sequence past the first adds.
    pub seq_ms: f64,
    /// The last row count the first band covers; past it the second band's pair applies.
    pub band_rows: usize,
}

impl BandWall {
    /// The rows term alone (ms): the band `rows` falls in.
    #[must_use]
    pub fn rows_ms(&self, rows: usize) -> f64 {
        if rows <= self.band_rows {
            self.a0 + self.a1 * rows as f64
        } else {
            self.b0 + self.b1 * rows as f64
        }
    }
}

impl StepWall for BandWall {
    fn ms(&self, seqs: usize, rows: usize) -> f64 {
        // The per-sequence term is charged past the first: the model's own intercept covers it.
        self.rows_ms(rows) + self.seq_ms * seqs.saturating_sub(1) as f64
    }
}

/// A mixture-of-experts walk whose cost is the experts it reads: `fixed_ms` for everything one
/// row reads whatever its picks (the dense weights, the head, the collectives), `expert_ms` for
/// each distinct expert any row picked, `row_ms` a row, and `seq_ms` a sequence past the first
/// (its own attention walk and draft pass). A row picks `top_k` of `experts` experts, and rows
/// are taken to pick independently, so `rows` rows read `experts * (1 - (1 - top_k/experts)^rows)`
/// distinct experts in expectation: the first rows each bring nearly `top_k` new experts, later
/// ones fewer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertUnionWall {
    /// What one row reads whatever its picks: the dense weights, the head, the collectives.
    pub fixed_ms: f64,
    /// The wall each distinct expert any row picked adds.
    pub expert_ms: f64,
    /// The wall a row adds.
    pub row_ms: f64,
    /// The wall each sequence past the first adds: its own attention walk and draft pass.
    pub seq_ms: f64,
    /// How many experts the layer routes over.
    pub experts: usize,
    /// How many experts a row picks.
    pub top_k: usize,
}

impl ExpertUnionWall {
    /// The distinct experts `rows` rows are expected to read.
    #[must_use]
    pub fn distinct(&self, rows: usize) -> f64 {
        if self.experts == 0 {
            return 0.0;
        }
        let n = self.experts as f64;
        let miss = 1.0 - (self.top_k.min(self.experts) as f64) / n;
        n * (1.0 - miss.powi(i32::try_from(rows).unwrap_or(i32::MAX)))
    }
}

impl StepWall for ExpertUnionWall {
    fn ms(&self, seqs: usize, rows: usize) -> f64 {
        self.fixed_ms
            + self.expert_ms * self.distinct(rows)
            + self.row_ms * rows as f64
            + self.seq_ms * seqs.saturating_sub(1) as f64
    }
}

/// Shares up to `budget` verify rows among the sequences' drafts, the likeliest accepted first:
/// each row goes to the sequence whose next draft has the highest chain probability (the lowest
/// sequence on a tie), while that is at least `floor`. A sequence's chain falls with depth, so
/// each count of rows holds the most expected accepted drafts it can; of those counts the step
/// takes the one with the most expected tokens a modelled millisecond (`wall`). `chains[s][j]` is
/// the chance that sequence `s`'s drafts `0..=j` are all accepted.
#[must_use]
pub fn share_rows(
    chains: &[Vec<f64>],
    budget: usize,
    floor: f64,
    wall: &dyn StepWall,
) -> Vec<usize> {
    let seqs = chains.len();
    let mut depths = vec![0usize; seqs];
    let mut order = Vec::with_capacity(budget);
    for _ in 0..budget {
        let mut best: Option<(usize, f64)> = None;
        for (index, (chain, &depth)) in chains.iter().zip(&depths).enumerate() {
            if let Some(&p) = chain.get(depth)
                && p >= floor
                && best.is_none_or(|(_, q)| p > q)
            {
                best = Some((index, p));
            }
        }
        match best {
            Some((index, p)) => {
                depths[index] += 1;
                order.push((index, p));
            }
            None => break,
        }
    }
    if seqs == 0 {
        return depths;
    }
    let mut tokens = seqs as f64;
    let (mut keep, mut best_rate) = (0, tokens / wall.ms(seqs, seqs));
    for (count, &(_, p)) in (1..).zip(&order) {
        tokens += p;
        let rate = tokens / wall.ms(seqs, seqs + count);
        if rate > best_rate {
            (keep, best_rate) = (count, rate);
        }
    }
    depths.fill(0);
    for &(index, _) in &order[..keep] {
        depths[index] += 1;
    }
    depths
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek's unset model: 27 ms for a c=1 step's anchor plus 4 ms a verified row, 2.3 ms a
    /// sequence past the first, in one band.
    const FLAT: BandWall = BandWall {
        a0: 23.0,
        a1: 4.0,
        b0: 23.0,
        b1: 4.0,
        seq_ms: 2.3,
        band_rows: 16,
    };

    #[test]
    fn rows_go_to_the_likeliest_drafts_while_they_pay() {
        let chains = [vec![0.9, 0.8, 0.3], vec![0.95, 0.5, 0.1], vec![0.15]];
        // 0.95 (b), 0.9 (a), 0.8 (a), 0.5 (b) raise the modelled rate (3.95 / 43.6 up to 6.15 /
        // 55.6 tokens a ms); 0.3 (a) lowers it; 0.15 and 0.1 are under the floor.
        assert_eq!(share_rows(&chains, 8, 0.2, &FLAT), [2, 2, 0]);
        assert_eq!(share_rows(&chains, 3, 0.2, &FLAT), [2, 1, 0]);
        assert_eq!(share_rows(&chains, 0, 0.2, &FLAT), [0, 0, 0]);
        // A tie goes to the lower sequence.
        assert_eq!(share_rows(&[vec![0.9], vec![0.9]], 1, 0.0, &FLAT), [1, 0]);
        // Drafts that are nearly sure take the whole budget.
        let sure = vec![vec![0.99; 5]; 8];
        assert_eq!(share_rows(&sure, 24, 0.2, &FLAT).iter().sum::<usize>(), 24);
        // A lone unlikely draft is not worth a row.
        assert_eq!(share_rows(&[vec![0.1], vec![0.1]], 4, 0.0, &FLAT), [0, 0]);
        assert!(share_rows(&[], 4, 0.0, &FLAT).is_empty());
    }

    #[test]
    fn a_steep_wide_band_stops_the_share_at_the_bend() {
        // Sixteen rows are cheap and everything past them dear: the share stops at 16.
        let steep = BandWall {
            a0: 5.0,
            a1: 1.0,
            b0: 400.0,
            b1: 1.0,
            seq_ms: 0.0,
            band_rows: 16,
        };
        let seqs = 8;
        let chains = vec![vec![0.95; 5]; seqs];
        let kept: usize = share_rows(&chains, 40, 0.0, &steep).iter().sum();
        assert_eq!(kept, 16 - seqs, "the bend, then stop");
        // A step whose rows cost a millisecond at every width takes the whole budget.
        let cheap = BandWall { b0: 5.0, ..steep };
        assert_eq!(
            share_rows(&chains, 40, 0.0, &cheap).iter().sum::<usize>(),
            40
        );
    }

    #[test]
    fn the_expert_union_charges_new_experts_only() {
        let wall = ExpertUnionWall {
            fixed_ms: 24.0,
            expert_ms: 1.0,
            row_ms: 0.0,
            seq_ms: 0.0,
            experts: 384,
            top_k: 8,
        };
        // One row reads its eight experts.
        assert!((wall.distinct(1) - 8.0).abs() < 1e-9);
        // Four rows read about 31 of them, sixteen about 110: each row adds fewer.
        assert!(
            (wall.distinct(4) - 31.0).abs() < 0.5,
            "{}",
            wall.distinct(4)
        );
        assert!(
            (wall.distinct(16) - 110.0).abs() < 1.0,
            "{}",
            wall.distinct(16)
        );
        let first = wall.ms(1, 2) - wall.ms(1, 1);
        let late = wall.ms(1, 17) - wall.ms(1, 16);
        assert!(
            late < first,
            "a late row costs {late} ms, the second {first}"
        );
        // Likely drafts across four sequences all pay at this price; unlikely ones do not.
        let likely = vec![vec![0.9, 0.8, 0.7]; 4];
        assert_eq!(share_rows(&likely, 28, 0.0, &wall), [3, 3, 3, 3]);
        let unlikely = vec![vec![0.05, 0.01]; 4];
        assert_eq!(share_rows(&unlikely, 28, 0.0, &wall), [0, 0, 0, 0]);
    }
}
