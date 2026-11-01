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

//! The recurrent pieces of delta-rule linear attention: the depthwise causal convolution in front
//! of the projections and the per-token state update.
//!
//! Gated DeltaNet (Qwen3.8-Flash-Next) and Kimi Delta Attention (GLM-5.3-Flash) share both. They
//! differ in where the convolution sits (one over the fused q/k/v projection, or one per
//! projection) and in the decay: Gated DeltaNet decays a head's whole state by one factor, KDA
//! each key channel's row by its own ([Decay]). Both L2-normalise each query and key head the FLA
//! way before the update ([l2_normalise]) and pass each head's output through a gated RMSNorm
//! ([rms_norm_gated]). The activations stay with the packs, whose numerics differ: a pack passes
//! its own in.
//!
//! # Attribution
//!
//! The gated delta rule is the method of Songlin Yang, Jan Kautz and Ali Hatamizadeh ("Gated Delta
//! Networks: Improving Mamba2 with Delta Rule", ICLR 2025); the per-channel decay is Kimi Delta
//! Attention's (Kimi Team, "Kimi Linear", arXiv:2510.26692). No source is copied. See the root
//! NOTICE file.

/// How a head's delta-rule state decays before a token writes to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decay<'a> {
    /// One factor for the head's whole `[key_dim][value_dim]` state (Gated DeltaNet).
    Head(f32),
    /// One factor per key channel, scaling that channel's `value_dim` row (Kimi Delta Attention).
    Channel(&'a [f32]),
}

/// One token's delta-rule step on one head.
///
/// `state` is the head's `[key_dim][value_dim]` memory, row-major by key channel, with
/// `key_dim = k.len()` and `value_dim = v.len()`. The step decays it, recalls what `k` reads from
/// it, writes `beta * (v - recall)` along `k`, and reads the updated state with `q` into `out`:
///
/// ```text
/// S = decay (.) S;   recall = S^T k;   S += k (beta (v - recall))^T;   out = S^T q
/// ```
///
/// Each sum is one accumulator in ascending key channel. `k` and `q` arrive normalised and scaled.
///
/// # Panics
///
/// When the slices do not describe one head: `state` is not `key_dim * value_dim`, `q` is not
/// `key_dim` long, `out` not `value_dim`, or a channel decay not `key_dim`.
pub fn delta_rule_step(
    state: &mut [f32],
    decay: Decay<'_>,
    k: &[f32],
    v: &[f32],
    beta: f32,
    q: &[f32],
    out: &mut [f32],
) {
    let key_dim = k.len();
    let value_dim = v.len();
    assert!(
        state.len() == key_dim * value_dim && q.len() == key_dim && out.len() == value_dim,
        "a [{key_dim}][{value_dim}] head needs that state, a {key_dim} query and a {value_dim} \
         output, got {}, {} and {}",
        state.len(),
        q.len(),
        out.len()
    );
    match decay {
        Decay::Head(factor) => {
            for entry in state.iter_mut() {
                *entry *= factor;
            }
        }
        Decay::Channel(factors) => {
            assert_eq!(factors.len(), key_dim, "one decay per key channel");
            for (row, factor) in state.chunks_mut(value_dim).zip(factors) {
                for entry in row {
                    *entry *= factor;
                }
            }
        }
    }
    let mut delta = vec![0.0f32; value_dim];
    for (dim, delta_value) in delta.iter_mut().enumerate() {
        let mut recall = 0.0f32;
        for (key_index, key_value) in k.iter().enumerate() {
            recall += state[key_index * value_dim + dim] * key_value;
        }
        *delta_value = (v[dim] - recall) * beta;
    }
    for (row, key_value) in state.chunks_mut(value_dim).zip(k) {
        for (entry, delta_value) in row.iter_mut().zip(&delta) {
            *entry += key_value * delta_value;
        }
    }
    for (dim, out_value) in out.iter_mut().enumerate() {
        let mut accumulator = 0.0f32;
        for (key_index, query_value) in q.iter().enumerate() {
            accumulator += state[key_index * value_dim + dim] * query_value;
        }
        *out_value = accumulator;
    }
}

/// Depthwise causal convolution, then `activation`, one output row per input row.
///
/// `kernel` is `[width][taps]`. PyTorch's `F.conv1d` is cross-correlation: tap `taps - 1`
/// multiplies the current row and tap 0 the row `taps - 1` steps back, reaching into `history`
/// (oldest first) before the chunk and reading zeros before that.
#[must_use]
pub fn causal_conv(
    rows: &[Vec<f32>],
    history: &[Vec<f32>],
    kernel: &[f32],
    width: usize,
    taps: usize,
    activation: impl Fn(f32) -> f32,
) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(rows.len());
    for index in 0..rows.len() {
        let mut convolved = vec![0.0f32; width];
        for channel in 0..width {
            let mut accumulator = 0.0f32;
            for tap in 0..taps {
                let back = taps - 1 - tap;
                let source = if index >= back {
                    rows[index - back][channel]
                } else {
                    let hist_back = back - index;
                    if hist_back <= history.len() {
                        history[history.len() - hist_back][channel]
                    } else {
                        0.0
                    }
                };
                accumulator += kernel[channel * taps + tap] * source;
            }
            convolved[channel] = activation(accumulator);
        }
        out.push(convolved);
    }
    out
}

/// Append a chunk's pre-convolution rows to a [causal_conv] history and keep the last `taps - 1`.
pub fn advance_conv_history(history: &mut Vec<Vec<f32>>, rows: Vec<Vec<f32>>, taps: usize) {
    let keep = taps.saturating_sub(1);
    history.extend(rows);
    if history.len() > keep {
        history.drain(..history.len() - keep);
    }
}

/// Kimi Delta Attention's decay factor for one key channel:
/// `exp(lower_bound * sigmoid(rate * (forget + dt_bias)))`.
///
/// `rate` is `exp(A_log)` of the channel's head, `forget` the channel's low-rank forget projection
/// and `dt_bias` its bias; `sigmoid` is the pack's. The factor is what [Decay::Channel] takes.
pub fn kda_channel_decay(
    lower_bound: f32,
    rate: f32,
    forget: f32,
    dt_bias: f32,
    sigmoid: impl Fn(f32) -> f32,
) -> f32 {
    (lower_bound * sigmoid(rate * (forget + dt_bias))).exp()
}

/// The epsilon inside the FLA L2 norm, added to the sum of squares rather than taken as a floor.
pub const L2_NORM_EPSILON: f32 = 1e-6;

/// One query or key head, L2-normalised the FLA way and then scaled, into `out`:
/// `x * (1 / sqrt(sum(x^2) + 1e-6)) * scale`.
///
/// The FLA kernels the references transcribe add [L2_NORM_EPSILON] inside the square root, where
/// `F.normalize` would floor the norm. Queries pass `1 / sqrt(key_dim)` as `scale`, keys 1.
///
/// # Panics
///
/// When `out` is not as long as `head`.
pub fn l2_normalise(head: &[f32], scale: f32, out: &mut [f32]) {
    assert_eq!(head.len(), out.len(), "one output per head element");
    let inverse_norm = 1.0 / (head.iter().map(|v| v * v).sum::<f32>() + L2_NORM_EPSILON).sqrt();
    for (slot, value) in out.iter_mut().zip(head) {
        *slot = value * inverse_norm * scale;
    }
}

/// The gated RMSNorm on one head's output: `x * rsqrt(mean(x^2) + eps) * weight *
/// activation(gate)`, element by element.
///
/// # Panics
///
/// When `weight` or `gate` is not as long as `input`.
#[must_use]
pub fn rms_norm_gated(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    eps: f32,
    activation: impl Fn(f32) -> f32,
) -> Vec<f32> {
    let width = input.len();
    assert!(
        weight.len() == width && gate.len() == width,
        "a {width}-wide head needs that weight and gate, got {} and {}",
        weight.len(),
        gate.len()
    );
    let mean_square: f32 = input.iter().map(|value| value * value).sum::<f32>() / width as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    (0..width)
        .map(|index| input[index] * scale * weight[index] * activation(gate[index]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_head_decay_is_a_channel_decay_with_one_value() {
        let k = [0.6f32, -0.8];
        let v = [1.0f32, 2.0, -0.5];
        let q = [0.25f32, 0.5];
        let start: Vec<f32> = (0..6).map(|i| 0.1 * i as f32 - 0.2).collect();
        let (mut head, mut channel) = (start.clone(), start);
        let (mut a, mut b) = ([0.0f32; 3], [0.0f32; 3]);
        delta_rule_step(&mut head, Decay::Head(0.9), &k, &v, 0.5, &q, &mut a);
        delta_rule_step(
            &mut channel,
            Decay::Channel(&[0.9, 0.9]),
            &k,
            &v,
            0.5,
            &q,
            &mut b,
        );
        assert_eq!((head, a), (channel, b));
    }

    #[test]
    fn a_channel_decay_scales_only_its_own_row() {
        // With k = 0 nothing is written, so the step is the decay and a read with q.
        let mut state = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 2];
        delta_rule_step(
            &mut state,
            Decay::Channel(&[0.5, 0.25]),
            &[0.0, 0.0],
            &[9.0, 9.0],
            1.0,
            &[1.0, 1.0],
            &mut out,
        );
        assert_eq!(state, vec![0.5, 1.0, 0.75, 1.0]);
        assert_eq!(out, [1.25, 2.0]);
    }

    #[test]
    fn a_unit_key_with_full_beta_overwrites_what_it_recalls() {
        // beta = 1 and a unit k: the row k selects becomes v exactly.
        let mut state = vec![0.5f32, -0.75, 0.25, 1.0];
        let mut out = [0.0f32; 2];
        delta_rule_step(
            &mut state,
            Decay::Head(1.0),
            &[1.0, 0.0],
            &[5.0, -6.0],
            1.0,
            &[1.0, 0.0],
            &mut out,
        );
        assert_eq!(state, vec![5.0, -6.0, 0.25, 1.0]);
        assert_eq!(out, [5.0, -6.0]);
    }

    #[test]
    fn the_convolution_reads_history_then_zeros() {
        // One channel, three taps [1, 10, 100]: the current row times 100, one back times 10.
        let kernel = [1.0f32, 10.0, 100.0];
        let rows = vec![vec![1.0f32], vec![2.0]];
        let history = vec![vec![3.0f32]];
        let out = causal_conv(&rows, &history, &kernel, 1, 3, |x| x);
        // Row 0: 100 * 1 + 10 * 3 (history) + 1 * 0 (before the history).
        // Row 1: 100 * 2 + 10 * 1 + 1 * 3.
        assert_eq!(out, vec![vec![130.0], vec![213.0]]);
        let mut history = history;
        advance_conv_history(&mut history, rows, 3);
        assert_eq!(history, vec![vec![1.0], vec![2.0]]);
    }

    #[test]
    fn the_l2_norm_adds_its_epsilon_inside_the_root_and_then_scales() {
        let mut out = [0.0f32; 2];
        l2_normalise(&[3.0, 4.0], 0.5, &mut out);
        let norm = (25.0f32 + L2_NORM_EPSILON).sqrt();
        assert_eq!(out, [3.0 * (1.0 / norm) * 0.5, 4.0 * (1.0 / norm) * 0.5]);
        // A zero head stays zero instead of dividing by zero.
        l2_normalise(&[0.0, 0.0], 1.0, &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn the_gated_norm_scales_by_the_rms_then_the_weight_then_the_activated_gate() {
        let input = [1.0f32, -2.0, 2.0, -1.0];
        let gated = rms_norm_gated(
            &input,
            &[1.0, 2.0, 1.0, 1.0],
            &[1.0, 1.0, 0.5, 0.0],
            0.0,
            |g| g,
        );
        // mean square 2.5, so every value is divided by sqrt(2.5) before the weight and the gate.
        let unit = 1.0 / 2.5f32.sqrt();
        assert_eq!(gated, [unit, -2.0 * unit * 2.0, 2.0 * unit * 0.5, 0.0]);
    }
}
