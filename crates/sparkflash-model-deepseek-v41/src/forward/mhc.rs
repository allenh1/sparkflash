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

//! V4.1's mHC: the shared [sparkflash_weights::mhc] at this model's geometry.
//!
//! The arithmetic lives in `sparkflash-weights` because GLM-5.3-Flash runs the same schedule; this
//! module re-exports it and gives [ModelConfig] its [HcGeometry], so every V4.1 caller passes its
//! configuration as before. The tests below pin the shared code to V4.1's specification and
//! captured fixtures.

pub use sparkflash_weights::mhc::*;

use crate::config::ModelConfig;

impl From<&ModelConfig> for HcGeometry {
    fn from(cfg: &ModelConfig) -> Self {
        Self {
            hc_mult: cfg.hc_mult,
            dim: cfg.dim,
            hc_sinkhorn_iters: cfg.hc_sinkhorn_iters,
            hc_eps: cfg.hc_eps as f32,
            norm_eps: cfg.norm_eps as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{self, EngineError};
    use crate::tensor::TensorPlan;

    /// A `[mix_hc, width]` `hc_*_fn` weight whose listed rows read `x_hc[0]` and whose other rows
    /// are zero.
    fn fn_weight(mix_hc: usize, width: usize, rows: &[(usize, f32)]) -> Vec<f32> {
        let mut weight = vec![0.0; mix_hc * width];
        for (row, value) in rows {
            assert!(*row < mix_hc, "row {row} is outside {mix_hc} coefficients");
            weight[row * width] = *value;
        }
        weight
    }

    /// `pre` and `post` are one mix per token and `comb` is one matrix per token: `hc_mult`,
    /// `hc_mult` and `hc_mult * hc_mult`. The `hc_*_fn` weight is the wider `hc_mix_count()` rows,
    /// so the two sizes have to be told apart: 2/2/4 and 8 rows at the tiny configuration's
    /// `hc_mult = 2`, 4/4/16 and 24 rows at the released 4. The plan carries the same geometry, and
    /// that is what is asserted here rather than a literal.
    #[test]
    fn the_coefficient_sets_are_sized_from_the_configuration() {
        let cfg = ModelConfig::tiny();
        let mixes = Mixes::for_config(&cfg);
        assert_eq!(
            (mixes.pre.len(), mixes.post.len(), mixes.comb.len()),
            (cfg.hc_mult, cfg.hc_mult, cfg.hc_mult * cfg.hc_mult)
        );
        assert_eq!((mixes.pre.len(), mixes.comb.len()), (2, 4));
        assert_eq!(cfg.hc_mix_count(), 8);

        let released = ModelConfig::v41_flash_cached();
        let mixes = Mixes::for_config(released);
        assert_eq!(
            (mixes.pre.len(), mixes.post.len(), mixes.comb.len()),
            (
                released.hc_mult,
                released.hc_mult,
                released.hc_mult * released.hc_mult
            )
        );
        assert_eq!((mixes.pre.len(), mixes.comb.len()), (4, 16));
        assert_eq!(released.hc_mix_count(), 24);

        let plan = TensorPlan::language_model(&cfg);
        for (name, shape) in [
            (
                "layers.0.hc_attn_fn",
                vec![cfg.hc_mix_count(), cfg.hc_width()],
            ),
            ("layers.0.hc_attn_base", vec![cfg.hc_mix_count()]),
            ("layers.0.hc_attn_scale", vec![3]),
        ] {
            let spec = plan
                .specs()
                .iter()
                .find(|spec| spec.name == name)
                .unwrap_or_else(|| panic!("the plan names {name}"));
            assert_eq!(spec.shape, shape, "{name}");
        }
    }

    /// `identity_pre_mix` puts 1 on copy 0 and 0 on every other copy, and refuses a buffer that is
    /// not `hc_mult` long.
    #[test]
    fn the_identity_pre_mix_is_a_one_hot_on_copy_zero() {
        let cfg = ModelConfig::tiny();
        let mut out = vec![9.0; cfg.hc_mult];
        identity_pre_mix(&cfg, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 0.0]);

        let released = ModelConfig::v41_flash_cached();
        let mut out = vec![9.0; released.hc_mult];
        identity_pre_mix(released, &mut out).unwrap();
        assert_eq!(out[0], 1.0);
        assert!(out[1..].iter().all(|value| *value == 0.0));

        assert!(matches!(
            identity_pre_mix(&cfg, &mut [0.0; 3]),
            Err(EngineError::Shape { .. })
        ));
    }

    /// The mix is `fn_w * x * rsqrt(mean(x^2) + norm_eps)` with `norm_eps = 1e-20` under the root
    /// and the mean taken over the whole flattened token. Two mutants are pinned by two tokens: the
    /// deliberately tiny one makes `hc_eps` (1e-6) as the epsilon visible, and the one whose second
    /// copy dominates the mean but not the projection makes a mean over `dim` instead of over
    /// `hc_mult * dim` visible, because the two denominators differ by `sqrt(hc_mult)`.
    #[test]
    fn the_mixes_are_the_rms_scaled_projection_with_norm_eps() {
        let cfg = ModelConfig::tiny();
        let width = cfg.hc_width();
        let dim = cfg.dim;
        let mix_hc = cfg.hc_mix_count();
        let base = vec![0.0; mix_hc];
        let fn_w = fn_weight(mix_hc, width, &[(0, 1.0)]);

        // A token so small that the epsilon under the root decides the value.
        let mut tiny = vec![0.0; width];
        tiny[0] = 1e-4;
        let mut mixes = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &tiny, &fn_w, &[1.0, 1.0, 1.0], &base, &mut mixes).unwrap();
        let mean_square = 1e-4f32 * 1e-4 / width as f32;
        let expected = linalg::sigmoid(1e-4 * (1.0 / (mean_square + 1e-20f32).sqrt())) + 1e-6;
        assert!(
            (mixes.pre[0] - expected).abs() < 1e-6,
            "pre[0] is {} but the specification's rsqrt gives {expected}",
            mixes.pre[0]
        );
        let with_hc_eps = linalg::sigmoid(1e-4 * (1.0 / (mean_square + 1e-6f32).sqrt())) + 1e-6;
        assert!(
            (expected - with_hc_eps).abs() > 1e-3,
            "the two epsilons must be distinguishable at this token"
        );

        // A token whose copy 1 dominates the mean but not the row 0 projection.
        let mut spread = vec![0.0; width];
        spread[0] = 1.0;
        spread[dim] = 3.0;
        let mut mixes = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &spread, &fn_w, &[1.0, 1.0, 1.0], &base, &mut mixes).unwrap();
        let sum_squares = 1.0f32 + 3.0 * 3.0;
        let flattened =
            linalg::sigmoid(1.0 / (sum_squares / width as f32 + 1e-20f32).sqrt()) + 1e-6;
        let per_dim = linalg::sigmoid(1.0 / (sum_squares / dim as f32 + 1e-20f32).sqrt()) + 1e-6;
        assert!(
            (mixes.pre[0] - flattened).abs() < 1e-6,
            "pre[0] is {} but the flattened mean gives {flattened}",
            mixes.pre[0]
        );
        assert!(
            (flattened - per_dim).abs() > 1e-3,
            "the mean must be over the flattened token: {flattened} against {per_dim}"
        );
    }

    /// `pre` is `sigmoid(...) + hc_eps` and `post` is `2 * sigmoid(...)` with no epsilon
    /// (kernel.py:427, 429). A zero weight makes both mixes their base value, so the epsilon is
    /// the whole difference: dropping it from `pre` or adding it to `post` fails here.
    #[test]
    fn pre_carries_an_epsilon_and_post_does_not() {
        let cfg = ModelConfig::tiny();
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let x_hc = vec![0.0; width];
        let fn_w = vec![0.0; mix_hc * width];
        let base = vec![0.0; mix_hc];
        let mut mixes = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &x_hc, &fn_w, &[1.0, 1.0, 1.0], &base, &mut mixes).unwrap();
        for value in &mixes.pre {
            assert_eq!(*value, 0.5 + 1e-6, "pre adds hc_eps to sigmoid(0)");
        }
        for value in &mixes.post {
            assert_eq!(*value, 1.0, "post is 2 * sigmoid(0) with no epsilon");
        }
    }

    /// The split sinkhorn (kernel.py:436-458) runs `2 * iters` passes: a row softmax plus `hc_eps`,
    /// then ONE column normalisation outside the loop, then `iters - 1` pairs of a row and a column
    /// normalisation. The release's `hc_sinkhorn_iters = 20` is therefore 20 row and 20 column passes,
    /// and the last one is a column pass, so the result is column-stochastic and not row-stochastic:
    /// at `iters = 1` the single unconditional column pass already makes the columns sum to 1 (to
    /// within the `hc_eps` in every denominator). The test
    /// writes that schedule out from the kernel lines and compares the two entry for entry at iters 1,
    /// 2, 3 and 20, then asserts the power that comparison actually has: the softmax alone (the
    /// unconditional column pass folded into the loop) is 0.5 away at 1, and at 20 a schedule one pass
    /// short is 0.026 away and the `iters`-pass schedule is 0.025 away. All of those are far above the
    /// 1e-6 comparison, so neither a count nor a final-axis error can hide in convergence.
    #[test]
    fn the_split_sinkhorn_runs_two_passes_per_iteration() {
        let hc = 4usize;
        let hc_eps = 1e-6f32;
        // A wide dynamic range, so every normalisation keeps moving. A comb whose rows are
        // proportional - an affine ramp, say - is at the fixed point after one column pass and cannot
        // tell the schedules apart at all.
        let raw: Vec<f32> = (0..hc * hc)
            .map(|index| (0.4 * ((index * 3) % (hc * hc)) as f32).exp())
            .collect();
        let max_diff = |left: &[f32], right: &[f32]| {
            left.iter()
                .zip(right.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max)
        };
        let row_pass = |comb: &mut [f32]| {
            for row in comb.chunks_mut(hc) {
                let sum: f32 = row.iter().sum();
                for value in row.iter_mut() {
                    *value /= sum + hc_eps;
                }
            }
        };
        let column_pass = |comb: &mut [f32]| {
            for col in 0..hc {
                let sum: f32 = (0..hc).map(|row| comb[row * hc + col]).sum();
                for row in 0..hc {
                    comb[row * hc + col] /= sum + hc_eps;
                }
            }
        };
        // The reference ends on a column pass, so any schedule that ends on a row pass, or that
        // leaves the unconditional column pass out, is a different matrix here. The column SUMS are
        // checked end to end below, on the engine's own comb: with this comb's range the `hc_eps` in
        // every denominator is not negligible against a vanishing column sum, so the reference only
        // makes the columns sum to 1 to within `hc_eps`, and the tight version of that property belongs
        // on a comb whose columns are O(1).
        for iters in [1usize, 2, 3, 20] {
            let mut mine = raw.clone();
            split_sinkhorn(&mut mine, hc, iters, hc_eps);
            let mut expected = raw.clone();
            for row in expected.chunks_mut(hc) {
                linalg::softmax(row);
                for value in row.iter_mut() {
                    *value += hc_eps;
                }
            }
            column_pass(&mut expected);
            for _ in 1..iters {
                row_pass(&mut expected);
                column_pass(&mut expected);
            }
            for (index, (got, want)) in mine.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-6,
                    "iters={iters} entry {index}: {got} against {want}"
                );
            }
        }

        // The power the comparison above has, asserted rather than assumed. First the count: the
        // `iters`-pass schedule the module used to run, which is the softmax and then `iters - 1`
        // passes alternating column and row.
        let mut full = raw.clone();
        split_sinkhorn(&mut full, hc, 20, hc_eps);
        let mut alternating = raw.clone();
        for row in alternating.chunks_mut(hc) {
            linalg::softmax(row);
            for value in row.iter_mut() {
                *value += hc_eps;
            }
        }
        for pass in 1..20 {
            if pass % 2 == 1 {
                column_pass(&mut alternating);
            } else {
                row_pass(&mut alternating);
            }
        }
        let count = max_diff(&full, &alternating);
        assert!(
            count > 1e-3,
            "the 20-pass schedule must be distinguishable from the kernel's 40, got {count}"
        );

        // Then a schedule one pass short at 20: the kernel's, without its last column pass. This is
        // what an off-by-one loop gives, and it must be visible for the comparison to mean anything.
        let mut one_short = raw.clone();
        for row in one_short.chunks_mut(hc) {
            linalg::softmax(row);
            for value in row.iter_mut() {
                *value += hc_eps;
            }
        }
        column_pass(&mut one_short);
        for _ in 1..19 {
            row_pass(&mut one_short);
            column_pass(&mut one_short);
        }
        row_pass(&mut one_short);
        let short_at_twenty = max_diff(&full, &one_short);
        assert!(
            short_at_twenty > 1e-3,
            "a schedule one pass short must be visible at 20, got {short_at_twenty}"
        );

        // And the unconditional column pass folded into the loop, which at iters = 1 leaves the softmax
        // alone: that is what the module used to do, and what the old test asserted the columns of.
        let mut softmax_only = raw.clone();
        for row in softmax_only.chunks_mut(hc) {
            linalg::softmax(row);
            for value in row.iter_mut() {
                *value += hc_eps;
            }
        }
        let mut one_iter = raw.clone();
        split_sinkhorn(&mut one_iter, hc, 1, hc_eps);
        let folded = max_diff(&one_iter, &softmax_only);
        assert!(
            folded > 1e-2,
            "the unconditional column pass must move the iters = 1 result, got {folded}"
        );

        // End to end through hc_mixes: the columns already sum to 1 at iters = 1, because the column
        // pass after the softmax is unconditional. The module used to leave them at 0.25 and 1.75.
        let rows = [(4usize, 0.1f32), (5, 0.5), (6, 0.2), (7, 0.3)];
        let mut cfg = ModelConfig::tiny();
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let mut x_hc = vec![0.0; width];
        x_hc[0] = 1.0;
        let fn_w = fn_weight(mix_hc, width, &rows);
        let base = vec![0.0; mix_hc];
        for iters in [1usize, 2, 20] {
            cfg.hc_sinkhorn_iters = iters;
            let mut mixes = Mixes::for_config(&cfg);
            hc_mixes(&cfg, &x_hc, &fn_w, &[1.0, 1.0, 1.0], &base, &mut mixes).unwrap();
            let hc_mult = cfg.hc_mult;
            for col in 0..hc_mult {
                let sum: f32 = (0..hc_mult)
                    .map(|row| mixes.comb[row * hc_mult + col])
                    .sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "iters={iters} column {col} sums to {sum}"
                );
            }
            if iters == 1 {
                // At iters = 1 the single column pass is the last pass, so the columns are 1 and the
                // rows are not: the module used to run the softmax alone, which leaves the rows at 1 and
                // the columns at 0.25 and 1.75. After 40 passes the matrix is converged and its rows are
                // 1 as well, so this contrast belongs here; the final axis at 20 is pinned by the direct
                // comparison above, where a row-ending schedule is 0.026 away.
                let row_sums: Vec<f32> = (0..hc_mult)
                    .map(|row| mixes.comb[row * hc_mult..(row + 1) * hc_mult].iter().sum())
                    .collect();
                assert!(
                    row_sums.iter().any(|sum| (sum - 1.0).abs() > 1e-3),
                    "iters=1 rows {row_sums:?} must not all sum to 1: the last pass normalised columns"
                );
            }
        }
    }

    /// `hc_pre` collapses the copies with the pre-mix: `sum_c pre[c] * x_hc[c * dim + j]`.
    #[test]
    fn hc_pre_collapses_the_copies_with_the_pre_mix() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let mut x_hc = vec![0.0; hc * dim];
        for j in 0..dim {
            x_hc[j] = j as f32;
            x_hc[dim + j] = 2.0 * j as f32;
        }
        let mut out = vec![0.0; dim];
        hc_pre(&cfg, &[0.25, 0.75], &x_hc, &mut out).unwrap();
        for (j, value) in out.iter().enumerate() {
            assert_eq!(*value, 1.75 * j as f32);
        }
    }

    /// `hc_post` is `post[k] * x + sum_j comb[j][k] * residual[j]`, the released orientation: row
    /// `j` is the source copy and column `k` the destination copy. The comb here is asymmetric, so
    /// applying its transpose gives 1.9 and 2.05 instead of the released 1.5 and 2.25.
    #[test]
    fn hc_post_is_post_times_x_plus_the_comb_mix() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let x = vec![1.0; dim];
        let mut residual = vec![0.0; hc * dim];
        for j in 0..dim {
            residual[j] = 1.0;
            residual[dim + j] = 2.0;
        }
        let comb = [0.6, 0.4, 0.2, 0.8];
        let mut out = vec![0.0; hc * dim];
        hc_post(&cfg, &x, &residual, &[0.5, 0.25], &comb, &mut out).unwrap();
        for (j, value) in out.iter().take(dim).enumerate() {
            assert!(
                (*value - 1.5).abs() < 1e-6,
                "copy 0 value at {j} is {value}"
            );
        }
        for (j, value) in out.iter().skip(dim).enumerate() {
            assert!(
                (*value - 2.25).abs() < 1e-6,
                "copy 1 value at {j} is {value}"
            );
        }
    }

    /// The released `hc_post` formula value by value on a comb whose transpose gives different
    /// answers: `out[k] = post[k] * y + sum_j comb[j][k] * residual[j]` (model.py:965;
    /// `docs/architecture/02-mhc.md:91-99`). The comb below is `C[source][destination]` and
    /// unnormalised; the expected copies are 1.2 and 2.65, and applying `C` transposed would give
    /// 1.8 and 2.35 instead.
    #[test]
    fn hc_post_mixes_each_destination_from_its_comb_column() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        assert_eq!(hc, 2, "the test's expected copies assume two mHC copies");
        let x = vec![1.0; dim];
        let mut residual = vec![0.0; hc * dim];
        for j in 0..dim {
            residual[j] = 1.0;
            residual[dim + j] = 2.0;
        }
        let comb = [0.1, 0.6, 0.3, 0.9];
        let mut out = vec![0.0; hc * dim];
        hc_post(&cfg, &x, &residual, &[0.5, 0.25], &comb, &mut out).unwrap();
        // Destination 0: 0.5 * 1 + 0.1 * 1 + 0.3 * 2 = 1.2; destination 1:
        // 0.25 * 1 + 0.6 * 1 + 0.9 * 2 = 2.65.
        for (j, value) in out.iter().take(dim).enumerate() {
            assert!(
                (*value - 1.2).abs() < 1e-6,
                "copy 0 value at {j} is {value}"
            );
        }
        for (j, value) in out.iter().skip(dim).enumerate() {
            assert!(
                (*value - 2.65).abs() < 1e-6,
                "copy 1 value at {j} is {value}"
            );
        }
    }

    /// The same token and the same weights give bit-identical mixes, and a different token does
    /// not: the second assertion is what fails if a term of the projection is dropped.
    #[test]
    fn the_mixes_are_deterministic_and_depend_on_the_token() {
        let cfg = ModelConfig::tiny();
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let mut x_hc = vec![0.0; width];
        x_hc[0] = 2.0;
        x_hc[width - 1] = -1.5;
        let rows: Vec<(usize, f32)> = (0..mix_hc)
            .map(|row| (row, 0.5 + row as f32 * 0.25))
            .collect();
        let fn_w = fn_weight(mix_hc, width, &rows);
        let base: Vec<f32> = (0..mix_hc).map(|index| index as f32 * 0.01).collect();
        let scale = [0.5, 1.5, 2.0];

        let mut first = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &x_hc, &fn_w, &scale, &base, &mut first).unwrap();
        let mut second = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &x_hc, &fn_w, &scale, &base, &mut second).unwrap();
        assert_eq!(first, second, "the same token must give the same mixes");

        let mut other_token = x_hc.clone();
        other_token[0] = 3.0;
        let mut third = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &other_token, &fn_w, &scale, &base, &mut third).unwrap();
        assert_ne!(first.pre, third.pre, "the mixes must depend on the token");
    }

    /// Every function refuses a geometry the configuration does not describe rather than reading
    /// past a slice.
    #[test]
    fn the_mhc_reductions_refuse_geometry_that_does_not_fit() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let x_hc = vec![0.0; width];
        let fn_w = vec![0.0; mix_hc * width];
        let base = vec![0.0; mix_hc];
        let scale = [1.0, 1.0, 1.0];
        let mut mixes = Mixes::for_config(&cfg);

        assert!(hc_mixes(&cfg, &x_hc[1..], &fn_w, &scale, &base, &mut mixes).is_err());
        assert!(hc_mixes(&cfg, &x_hc, &fn_w[1..], &scale, &base, &mut mixes).is_err());
        assert!(hc_mixes(&cfg, &x_hc, &fn_w, &scale[..2], &base, &mut mixes).is_err());
        assert!(hc_mixes(&cfg, &x_hc, &fn_w, &scale, &base[1..], &mut mixes).is_err());
        let mut wrong = Mixes {
            pre: vec![0.0; hc - 1],
            post: vec![0.0; hc],
            comb: vec![0.0; hc * hc],
        };
        assert!(hc_mixes(&cfg, &x_hc, &fn_w, &scale, &base, &mut wrong).is_err());

        assert!(hc_pre(&cfg, &[1.0], &x_hc, &mut vec![0.0; dim]).is_err());
        assert!(hc_pre(&cfg, &[0.5, 0.5], &x_hc[1..], &mut vec![0.0; dim]).is_err());
        assert!(hc_pre(&cfg, &[0.5, 0.5], &x_hc, &mut vec![0.0; dim - 1]).is_err());

        let residual = vec![0.0; hc * dim];
        let comb = vec![0.0; hc * hc];
        let post = vec![0.0; hc];
        assert!(
            hc_post(
                &cfg,
                &[0.0; 1],
                &residual,
                &post,
                &comb,
                &mut vec![0.0; hc * dim]
            )
            .is_err()
        );
        assert!(
            hc_post(
                &cfg,
                &[0.0; 64],
                &residual[1..],
                &post,
                &comb,
                &mut vec![0.0; hc * dim]
            )
            .is_err()
        );
        assert!(
            hc_post(
                &cfg,
                &[0.0; 64],
                &residual,
                &post[1..],
                &comb,
                &mut vec![0.0; hc * dim]
            )
            .is_err()
        );
        assert!(
            hc_post(
                &cfg,
                &[0.0; 64],
                &residual,
                &post,
                &comb[1..],
                &mut vec![0.0; hc * dim]
            )
            .is_err()
        );
        assert!(
            hc_post(
                &cfg,
                &[0.0; 64],
                &residual,
                &post,
                &comb,
                &mut vec![0.0; hc * dim - 1]
            )
            .is_err()
        );
    }

    /// A chunk gets one coefficient set per token, laid out token by token: each slot equals what a
    /// one-token call on that token gives, and a `Mixes` sized for one token is refused for a
    /// two-token chunk rather than half-filled.
    #[test]
    fn a_chunk_gets_one_coefficient_set_per_token() {
        let cfg = ModelConfig::tiny();
        let hc = cfg.hc_mult;
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let mut first = vec![0.0f32; width];
        first[0] = 2.0;
        let mut second = vec![0.0f32; width];
        second[1] = -3.0;
        second[width - 1] = 4.0;
        let rows: Vec<(usize, f32)> = (0..mix_hc)
            .map(|row| (row, 0.25 + row as f32 * 0.5))
            .collect();
        let fn_w = fn_weight(mix_hc, width, &rows);
        let base: Vec<f32> = (0..mix_hc).map(|index| index as f32 * 0.02).collect();
        let scale = [1.0, 1.5, 0.5];

        let mut chunk = vec![0.0f32; 2 * width];
        chunk[..width].copy_from_slice(&first);
        chunk[width..].copy_from_slice(&second);
        let mut mixes = Mixes::for_tokens(&cfg, 2);
        hc_mixes(&cfg, &chunk, &fn_w, &scale, &base, &mut mixes).unwrap();

        for (slot, token) in [&first, &second].into_iter().enumerate() {
            let mut alone = Mixes::for_config(&cfg);
            hc_mixes(&cfg, token, &fn_w, &scale, &base, &mut alone).unwrap();
            assert_eq!(&mixes.pre[slot * hc..(slot + 1) * hc], &alone.pre[..]);
            assert_eq!(&mixes.post[slot * hc..(slot + 1) * hc], &alone.post[..]);
            assert_eq!(
                &mixes.comb[slot * hc * hc..(slot + 1) * hc * hc],
                &alone.comb[..]
            );
        }

        let mut too_small = Mixes::for_config(&cfg);
        assert!(
            hc_mixes(&cfg, &chunk, &fn_w, &scale, &base, &mut too_small).is_err(),
            "a one-token Mixes cannot hold a two-token chunk"
        );
    }

    /// The reductions work a whole chunk: every token's row is the collapse of its own copies and
    /// the mix of its own residual, so a loop that reused token 0's buffers fails here.
    #[test]
    fn the_reductions_work_a_whole_chunk() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let token_width = hc * dim;
        let mut x_hc = vec![0.0f32; 2 * token_width];
        for token in 0..2 {
            for copy in 0..hc {
                for j in 0..dim {
                    x_hc[token * token_width + copy * dim + j] =
                        (token + 1) as f32 + copy as f32 * 10.0 + j as f32 * 0.5;
                }
            }
        }
        let mut pre = vec![0.0f32; 2 * hc];
        for token in 0..2 {
            pre[token * hc] = 0.25;
            pre[token * hc + 1] = 0.75;
        }
        let mut out = vec![0.0f32; 2 * dim];
        hc_pre(&cfg, &pre, &x_hc, &mut out).unwrap();
        for token in 0..2 {
            for j in 0..dim {
                let expected = 0.25 * x_hc[token * token_width + j]
                    + 0.75 * x_hc[token * token_width + dim + j];
                assert!(
                    (out[token * dim + j] - expected).abs() < 1e-4,
                    "token {token} dim {j}"
                );
            }
        }

        let x = vec![1.0f32; 2 * dim];
        let post_mix = [0.5f32, 0.25];
        let comb = [0.6f32, 0.4, 0.2, 0.8];
        let mut post = vec![0.0f32; 2 * hc];
        let mut combs = vec![0.0f32; 2 * hc * hc];
        for token in 0..2 {
            post[token * hc..(token + 1) * hc].copy_from_slice(&post_mix);
            combs[token * hc * hc..(token + 1) * hc * hc].copy_from_slice(&comb);
        }
        let mut out_hc = vec![0.0f32; 2 * token_width];
        hc_post(&cfg, &x, &x_hc, &post, &combs, &mut out_hc).unwrap();
        for token in 0..2 {
            for j in 0..dim {
                // Destination copy 0 reads column 0 of the released comb: C[source][destination].
                let expected = 0.5 * x[token * dim + j]
                    + 0.6 * x_hc[token * token_width + j]
                    + 0.2 * x_hc[token * token_width + dim + j];
                assert!(
                    (out_hc[token * token_width + j] - expected).abs() < 1e-4,
                    "token {token} copy 0 dim {j}"
                );
                let expected = 0.25 * x[token * dim + j]
                    + 0.4 * x_hc[token * token_width + j]
                    + 0.8 * x_hc[token * token_width + dim + j];
                assert!(
                    (out_hc[token * token_width + dim + j] - expected).abs() < 1e-4,
                    "token {token} copy 1 dim {j}"
                );
            }
        }
    }

    /// The released geometry, for the captured rows: the fixture is 5120 wide with 4 mHC copies.
    fn captured_config() -> ModelConfig {
        let mut cfg = ModelConfig::tiny();
        cfg.dim = 5120;
        cfg.hc_mult = 4;
        cfg
    }

    /// One committed captured-activation fixture; provenance is in the directory's
    /// `provenance.json` (phase1 boundary v1 host capture, layer 1, position 0).
    fn captured_fixture(name: &str) -> Vec<f32> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/native-rounding")
            .join(name);
        let bytes =
            std::fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(
            bytes.len() % 4,
            0,
            "{} is not whole f32 values",
            path.display()
        );
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    /// The capture is the f32 reference chain; the engine's own hc_post over the captured inputs
    /// must reproduce it bit for bit or the fixture is not the chain these tests claim.
    #[test]
    fn the_captured_hc_post_row_is_the_reference_stream() {
        let cfg = captured_config();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let x = captured_fixture("attention-out-layer1-pos0.f32");
        let residual = captured_fixture("residual_in-layer1-pos0.f32");
        let post = captured_fixture("hc-attn-post-layer1-pos0.f32");
        let comb = captured_fixture("hc-attn-comb-layer1-pos0.f32");
        let expected = captured_fixture("post-attention-hidden-layer1-pos0.f32");
        let mut out = vec![0.0f32; hc * dim];
        hc_post(&cfg, &x, &residual, &post, &comb, &mut out).expect("hc_post");
        assert_eq!(
            out, expected,
            "the engine hc_post must reproduce the captured reference stream exactly"
        );
    }

    #[test]
    fn the_captured_hc_post_native_rounds_and_keeps_the_release_order() {
        let cfg = captured_config();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let x = captured_fixture("attention-out-layer1-pos0.f32");
        let residual = captured_fixture("residual_in-layer1-pos0.f32");
        let post = captured_fixture("hc-attn-post-layer1-pos0.f32");
        let comb = captured_fixture("hc-attn-comb-layer1-pos0.f32");
        let mut native = vec![0.0f32; hc * dim];
        hc_post_bf16(&cfg, &x, &residual, &post, &comb, &mut native).expect("hc_post_bf16");
        // The independent release-order reference: the comb sum, then one post-term addition,
        // then the BF16 cast. Written out so the test does not call the function it checks.
        for copy in 0..hc {
            for j in 0..dim {
                let mut comb_sum = 0.0f32;
                for source in 0..hc {
                    comb_sum += comb[source * hc + copy] * residual[source * dim + j];
                }
                let post_term = post[copy] * x[j];
                let reference = crate::quant::bf16_round(post_term + comb_sum);
                assert_eq!(
                    native[copy * dim + j],
                    reference,
                    "copy {copy} dim {j} is not the release arithmetic"
                );
            }
        }
        // Every value is BF16-representable, and the boundary actually moves values: the f32 chain
        // on the same captured row is not bf16-representable on most entries.
        let mut f32_out = vec![0.0f32; hc * dim];
        hc_post(&cfg, &x, &residual, &post, &comb, &mut f32_out).expect("hc_post");
        let mut changed = 0usize;
        for (native, f32_value) in native.iter().zip(f32_out.iter()) {
            assert_eq!(
                crate::quant::bf16_round(*native),
                *native,
                "a native output is not BF16-representable"
            );
            if native != f32_value {
                changed += 1;
            }
        }
        assert!(
            changed > 0,
            "the captured row must contain values the BF16 boundary changes"
        );
        eprintln!(
            "captured hc_post row: {changed}/{} values move at the BF16 boundary",
            hc * dim
        );
    }

    #[test]
    fn the_captured_hc_pre_native_rounds_the_collapse() {
        let cfg = captured_config();
        let dim = cfg.dim;
        let residual = captured_fixture("residual_in-layer1-pos0.f32");
        let pre = captured_fixture("hc-in-pre-layer1-pos0.f32");
        let mut native = vec![0.0f32; dim];
        hc_pre_bf16(&cfg, &pre, &residual, &mut native).expect("hc_pre_bf16");
        let mut f32_out = vec![0.0f32; dim];
        hc_pre(&cfg, &pre, &residual, &mut f32_out).expect("hc_pre");
        let mut changed = 0usize;
        for (native, f32_value) in native.iter().zip(f32_out.iter()) {
            assert_eq!(
                *native,
                crate::quant::bf16_round(*f32_value),
                "the collapse must be its f32 value rounded once"
            );
            if native != f32_value {
                changed += 1;
            }
        }
        assert!(
            changed > 0,
            "the captured collapse must contain values the BF16 round changes"
        );
        eprintln!("captured hc_pre row: {changed}/{dim} values move at the BF16 boundary");
    }

    /// A deterministic f32 stream for the projection tests: values in the range the coefficient
    /// arithmetic actually sees, from a 64-bit LCG so the test has no dependency.
    fn projection_values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = ((state >> 40) as u32) & 0x00ff_ffff;
                (f32::from_bits(bits | 0x3f00_0000) - 1.5) * 0.25
            })
            .collect()
    }

    /// [hc_mixes] is exactly [hc_projection] plus [hc_mixes_from_projection]: no operation, order or
    /// rounding moves when the two halves are called separately.
    #[test]
    fn the_split_is_bit_identical_to_the_combined_function() {
        let cfg = crate::config::ModelConfig::tiny();
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let scale = vec![1.25f32, -0.5, 0.75];
        let base = projection_values(mix_hc, 0x5555_aaaa);
        for tokens in [1usize, 3] {
            let stream = projection_values(tokens * width, 0x1234_5678 + tokens as u64);
            let fn_w = projection_values(mix_hc * width, 0x9abc_def0);
            let mut combined = Mixes::for_tokens(&cfg, tokens);
            hc_mixes(&cfg, &stream, &fn_w, &scale, &base, &mut combined).expect("combined");
            let projection = hc_projection(&cfg, &stream, &fn_w).expect("projection");
            let mut split = Mixes::for_tokens(&cfg, tokens);
            hc_mixes_from_projection(&cfg, &projection, &scale, &base, &mut split).expect("split");
            assert_eq!(combined, split, "{tokens} tokens");
        }
    }

    /// The interleaved loop reorders iterations, not operations: every coefficient row's dot product
    /// and every token's sum of squares is the same expression sequence in the same order, so the
    /// transposed pass is the ordered pass bit for bit - at the shipped geometry and at a small one.
    #[test]
    fn the_transposed_projection_is_the_ordered_projection() {
        let shipped = crate::config::ModelConfig::v41_flash_cached();
        let tiny = crate::config::ModelConfig::tiny();
        for (cfg, tokens) in [(shipped, 1usize), (shipped, 3), (&tiny, 1), (&tiny, 2)] {
            let width = cfg.hc_width();
            let mix_hc = cfg.hc_mix_count();
            let stream = projection_values(tokens * width, 0x1111_2222 + tokens as u64);
            let fn_w = projection_values(mix_hc * width, 0x3333_4444);
            let mut transposed = vec![0.0f32; fn_w.len()];
            hc_transpose_predictor(cfg, &fn_w, &mut transposed).expect("transpose");
            let ordered = hc_projection(cfg, &stream, &fn_w).expect("ordered");
            let interleaved =
                hc_projection_transposed(cfg, &stream, &transposed).expect("interleaved");
            assert_eq!(
                interleaved
                    .sum_squares
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                ordered
                    .sum_squares
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{tokens} tokens: the sum of squares"
            );
            assert_eq!(
                interleaved
                    .raw
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                ordered
                    .raw
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{tokens} tokens: the coefficient rows"
            );
        }
    }

    /// The transposed mixes are the ordered mixes, and the transpose itself only moves values.
    #[test]
    fn the_transposed_mixes_are_the_ordered_mixes() {
        let cfg = crate::config::ModelConfig::tiny();
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let stream = projection_values(width, 0x0f0f_0f0f);
        let fn_w = projection_values(mix_hc * width, 0xf0f0_f0f0);
        let scale = vec![1.25f32, -0.5, 0.75];
        let base = projection_values(mix_hc, 0x2468_ace0);
        let mut transposed = vec![0.0f32; fn_w.len()];
        hc_transpose_predictor(&cfg, &fn_w, &mut transposed).expect("transpose");
        // The transposed copy holds the same multiset of bits, and every value is in its slot.
        let mut sorted_original = fn_w.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        let mut sorted_transposed = transposed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        sorted_original.sort_unstable();
        sorted_transposed.sort_unstable();
        assert_eq!(sorted_original, sorted_transposed);
        for row in 0..mix_hc {
            for k in 0..width {
                assert_eq!(
                    transposed[k * mix_hc + row].to_bits(),
                    fn_w[row * width + k].to_bits(),
                    "row {row} k {k}"
                );
            }
        }
        let mut ordered = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &stream, &fn_w, &scale, &base, &mut ordered).expect("ordered");
        let mut interleaved = Mixes::for_config(&cfg);
        hc_mixes_transposed(&cfg, &stream, &transposed, &scale, &base, &mut interleaved)
            .expect("interleaved");
        assert_eq!(ordered, interleaved);
        // The transpose refuses a buffer that is not the whole weight.
        assert!(hc_transpose_predictor(&cfg, &fn_w, &mut vec![0.0f32; fn_w.len() - 1]).is_err());
    }

    #[test]
    fn the_captured_norm_output_is_bf16_before_the_sublayer() {
        let cfg = captured_config();
        let dim = cfg.dim;
        let residual = captured_fixture("residual_in-layer1-pos0.f32");
        let pre = captured_fixture("hc-in-pre-layer1-pos0.f32");
        let weight = vec![1.0f32; dim];
        let mut collapsed = vec![0.0f32; dim];
        hc_pre_bf16(&cfg, &pre, &residual, &mut collapsed).expect("hc_pre_bf16");
        let mut normed = collapsed.clone();
        rms_norm_bf16(&mut normed, &weight, dim, cfg.norm_eps as f32).expect("rms_norm_bf16");
        let mut reference = collapsed.clone();
        linalg::rms_norm(&mut reference, &weight, dim, cfg.norm_eps as f32).expect("rms_norm");
        bf16_round_row(&mut reference);
        assert_eq!(
            normed, reference,
            "the norm boundary is one BF16 round of the f32 norm"
        );
        assert!(
            normed
                .iter()
                .all(|value| crate::quant::bf16_round(*value) == *value),
            "the norm output must be BF16-representable"
        );
    }
}
