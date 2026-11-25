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

/// One Gated DeltaNet layer's recurrent state on the host: the raw fused q/k/v rows before the
/// next (oldest first, at most `taps - 1`, the [causal_conv] history) and each value head's
/// `[key_dim][value_dim]` delta-rule state (empty before the first token).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GdnState {
    /// The conv history.
    pub conv_history: Vec<Vec<f32>>,
    /// Each value head's state.
    pub recurrent: Vec<Vec<f32>>,
}

/// A Gated DeltaNet layer's heads on the host: the key heads q and k come in, the value heads
/// (a whole multiple of them) v and the state, each head's floats, and the conv's taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnHeads {
    /// The key heads.
    pub key_heads: usize,
    /// The value heads.
    pub value_heads: usize,
    /// A key head's floats.
    pub key_dim: usize,
    /// A value head's floats.
    pub value_dim: usize,
    /// The conv's taps.
    pub taps: usize,
}

impl GdnHeads {
    /// A key row's floats (the q part, or the k part, of the fused row).
    #[must_use]
    pub const fn key_width(&self) -> usize {
        self.key_heads * self.key_dim
    }

    /// A value row's floats.
    #[must_use]
    pub const fn value_width(&self) -> usize {
        self.value_heads * self.value_dim
    }

    /// The fused q/k/v row's floats, which the conv runs over.
    #[must_use]
    pub const fn conv_width(&self) -> usize {
        2 * self.key_width() + self.value_width()
    }
}

/// A Gated DeltaNet layer's weights past its projections, as f32: the conv `[conv_width][taps]`,
/// the per-value-head `A_log` and `dt_bias`, and the gated norm's gains (a value head's width).
#[derive(Debug, Clone, Copy)]
pub struct GdnTensors<'a> {
    /// The conv.
    pub conv: &'a [f32],
    /// `A_log`.
    pub a_log: &'a [f32],
    /// `dt_bias`.
    pub dt_bias: &'a [f32],
    /// The gated norm's gains.
    pub norm: &'a [f32],
}

/// The activations a pack's Gated DeltaNet takes, its own numerics: the conv's (SiLU), the output
/// gate's, the softplus of the decay and the sigmoid of beta.
#[derive(Clone, Copy)]
pub struct GdnActivations<'a> {
    /// After the conv.
    pub conv: &'a dyn Fn(f32) -> f32,
    /// On the output gate.
    pub gate: &'a dyn Fn(f32) -> f32,
    /// `softplus`, inside the decay.
    pub softplus: &'a dyn Fn(f32) -> f32,
    /// `sigmoid`, of beta.
    pub sigmoid: &'a dyn Fn(f32) -> f32,
}

/// One token through a Gated DeltaNet layer after its projections: the fused row `raw` convolved
/// over `state`'s history (then appended to it), each value head's q and k from its key head
/// L2-normalised (q scaled by `1 / sqrt(key_dim)`), the decay `exp(-exp(A_log) softplus(a +
/// dt_bias))` and `beta = sigmoid(b)`, the delta-rule step on each head's state, and each head's
/// gated RMSNorm by `z` at `epsilon`. Returns the normed heads, `[value_width]`, which `out_proj`
/// takes.
///
/// # Errors
/// A projection or a tensor whose length is not the heads', by name.
pub fn gdn_token(
    heads: GdnHeads,
    tensors: GdnTensors<'_>,
    (raw, z, b, a): (Vec<f32>, &[f32], &[f32], &[f32]),
    state: &mut GdnState,
    epsilon: f32,
    activations: GdnActivations<'_>,
) -> Result<Vec<f32>, String> {
    let GdnHeads {
        value_heads,
        key_dim,
        value_dim,
        taps,
        ..
    } = heads;
    let (key_width, value_width, width) =
        (heads.key_width(), heads.value_width(), heads.conv_width());
    let lengths = [
        ("the fused q/k/v projection", raw.len(), width),
        ("the output gate's projection", z.len(), value_width),
        ("in_proj_b", b.len(), value_heads),
        ("in_proj_a", a.len(), value_heads),
        ("the conv", tensors.conv.len(), width * taps),
        ("A_log", tensors.a_log.len(), value_heads),
        ("dt_bias", tensors.dt_bias.len(), value_heads),
        ("the gated norm", tensors.norm.len(), value_dim),
    ];
    for (name, got, want) in lengths {
        if got != want {
            return Err(format!(
                "the Gated DeltaNet's {name} holds {got} floats, expected {want}"
            ));
        }
    }
    if heads.key_heads == 0 || value_heads % heads.key_heads != 0 {
        return Err(format!(
            "{value_heads} Gated DeltaNet value heads over {} key heads",
            heads.key_heads
        ));
    }
    let convolved = causal_conv(
        std::slice::from_ref(&raw),
        &state.conv_history,
        tensors.conv,
        width,
        taps,
        activations.conv,
    );
    let mixed = &convolved[0];
    let groups = value_heads / heads.key_heads;
    let scale = (key_dim as f32).sqrt().recip();
    let mut qn = vec![0.0f32; value_heads * key_dim];
    let mut kn = vec![0.0f32; value_heads * key_dim];
    for head in 0..value_heads {
        let source = (head / groups) * key_dim;
        let span = head * key_dim..(head + 1) * key_dim;
        l2_normalise(
            &mixed[source..source + key_dim],
            scale,
            &mut qn[span.clone()],
        );
        l2_normalise(
            &mixed[key_width + source..key_width + source + key_dim],
            1.0,
            &mut kn[span],
        );
    }
    let values = &mixed[2 * key_width..2 * key_width + value_width];
    if state.recurrent.is_empty() {
        state.recurrent = vec![vec![0.0f32; key_dim * value_dim]; value_heads];
    }
    let mut core = vec![0.0f32; value_width];
    for head in 0..value_heads {
        let decay = (-tensors.a_log[head].exp()
            * (activations.softplus)(a[head] + tensors.dt_bias[head]))
        .exp();
        let beta = (activations.sigmoid)(b[head]);
        let (keys, values_at) = (
            head * key_dim..(head + 1) * key_dim,
            head * value_dim..(head + 1) * value_dim,
        );
        delta_rule_step(
            &mut state.recurrent[head],
            Decay::Head(decay),
            &kn[keys.clone()],
            &values[values_at.clone()],
            beta,
            &qn[keys],
            &mut core[values_at],
        );
    }
    let mut normed = vec![0.0f32; value_width];
    for head in 0..value_heads {
        let span = head * value_dim..(head + 1) * value_dim;
        let gated = rms_norm_gated(
            &core[span.clone()],
            tensors.norm,
            &z[span.clone()],
            epsilon,
            activations.gate,
        );
        normed[span].copy_from_slice(&gated);
    }
    advance_conv_history(&mut state.conv_history, vec![raw], taps);
    Ok(normed)
}

/// One Kimi Delta Attention layer's recurrent state on the host: the raw q, k and v rows before
/// the next (oldest first, at most `taps - 1` each, the [causal_conv] histories) and each head's
/// `[head_dim][head_dim]` memory, row-major by key channel (empty before the first token).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KdaState {
    /// The q, k and v conv histories.
    pub conv: [Vec<Vec<f32>>; 3],
    /// Each head's memory.
    pub recurrent: Vec<Vec<f32>>,
}

/// A Kimi Delta Attention layer's heads on the host: `heads` heads (a rank's share, or all of
/// them) of `head_dim` keys and values each, and the conv's taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaHeads {
    /// The heads.
    pub heads: usize,
    /// A head's key and value width.
    pub head_dim: usize,
    /// The conv's taps.
    pub taps: usize,
}

impl KdaHeads {
    /// The q, k or v row's floats, and the gates' and the output's.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// A Kimi Delta Attention layer's weights past its projections, as f32: the q, k and v convs
/// (`[width][taps]` each), `A_log` a head's, `dt_bias` a channel's and the gated norm's gains (a
/// head's width).
#[derive(Debug, Clone, Copy)]
pub struct KdaTensors<'a> {
    /// The q, k and v convs.
    pub conv: [&'a [f32]; 3],
    /// `A_log`.
    pub a_log: &'a [f32],
    /// `dt_bias`.
    pub dt_bias: &'a [f32],
    /// The gated norm's gains.
    pub norm: &'a [f32],
}

/// The activations a pack's Kimi Delta Attention takes, its own numerics: the conv's (SiLU) and
/// the sigmoid of the decay, of beta and of the output gate.
#[derive(Clone, Copy)]
pub struct KdaActivations<'a> {
    /// After the conv.
    pub conv: &'a dyn Fn(f32) -> f32,
    /// `sigmoid`.
    pub sigmoid: &'a dyn Fn(f32) -> f32,
    /// The type `beta` and the delta rule's output are held in before the gated norm: the
    /// identity for an f32 body, a BF16 round where the producer returns BF16 there.
    pub narrow: &'a dyn Fn(f32) -> f32,
}

/// One token through a Kimi Delta Attention layer after its projections (GLM-5.3-Flash's linear
/// attention, `modeling_glm5_next.py` 357-391 and 533): q, k and v each convolved over its own
/// history (then appended to it), each head's q and k L2-normalised (q scaled by `1 /
/// sqrt(head_dim)`), each key channel's decay [kda_channel_decay] from the low-rank forget
/// projection `forget` at `lower_bound`, `beta = sigmoid(write)`, the delta-rule step on each
/// head's memory, and each head's gated RMSNorm by the output gate `gate` with a sigmoid at
/// `epsilon`. Returns the normed heads, `[width]`, which `o_proj` takes.
///
/// # Errors
/// A projection or a tensor whose length is not the heads', by name.
pub fn kda_token(
    heads: KdaHeads,
    tensors: KdaTensors<'_>,
    (qkv, forget, write, gate): ([Vec<f32>; 3], &[f32], &[f32], &[f32]),
    state: &mut KdaState,
    (epsilon, lower_bound): (f32, f32),
    activations: KdaActivations<'_>,
) -> Result<Vec<f32>, String> {
    let KdaHeads {
        heads: count,
        head_dim,
        taps,
    } = heads;
    let width = heads.width();
    let lengths = [
        ("q_proj", qkv[0].len(), width),
        ("k_proj", qkv[1].len(), width),
        ("v_proj", qkv[2].len(), width),
        ("the forget gate's projection", forget.len(), width),
        ("b_proj", write.len(), count),
        ("the output gate's projection", gate.len(), width),
        ("q_conv1d", tensors.conv[0].len(), width * taps),
        ("k_conv1d", tensors.conv[1].len(), width * taps),
        ("v_conv1d", tensors.conv[2].len(), width * taps),
        ("A_log", tensors.a_log.len(), count),
        ("dt_bias", tensors.dt_bias.len(), width),
        ("the gated norm", tensors.norm.len(), head_dim),
    ];
    for (name, got, want) in lengths {
        if got != want {
            return Err(format!(
                "the KDA layer's {name} holds {got} floats, expected {want}"
            ));
        }
    }
    let mut convolved: Vec<Vec<f32>> = Vec::with_capacity(3);
    for ((row, kernel), history) in qkv.into_iter().zip(tensors.conv).zip(&mut state.conv) {
        let mut out = causal_conv(
            std::slice::from_ref(&row),
            history,
            kernel,
            width,
            taps,
            activations.conv,
        );
        advance_conv_history(history, vec![row], taps);
        convolved.push(out.swap_remove(0));
    }
    if state.recurrent.is_empty() {
        state.recurrent = vec![vec![0.0f32; head_dim * head_dim]; count];
    }
    // `1 / sqrt(head_dim)` taken in f64 and rounded once, as GLM's host reference took it.
    let query_scale = (1.0 / (head_dim as f64).sqrt()) as f32;
    let mut out = vec![0.0f32; width];
    let mut query = vec![0.0f32; head_dim];
    let mut key = vec![0.0f32; head_dim];
    for (head, memory) in state.recurrent.iter_mut().enumerate() {
        let channels = head * head_dim..(head + 1) * head_dim;
        l2_normalise(&convolved[0][channels.clone()], query_scale, &mut query);
        l2_normalise(&convolved[1][channels.clone()], 1.0, &mut key);
        let rate = tensors.a_log[head].exp();
        let decay: Vec<f32> = channels
            .clone()
            .map(|channel| {
                kda_channel_decay(
                    lower_bound,
                    rate,
                    forget[channel],
                    tensors.dt_bias[channel],
                    activations.sigmoid,
                )
            })
            .collect();
        delta_rule_step(
            memory,
            Decay::Channel(&decay),
            &key,
            &convolved[2][channels.clone()],
            (activations.narrow)((activations.sigmoid)(write[head])),
            &query,
            &mut out[channels],
        );
    }
    for value in &mut out {
        *value = (activations.narrow)(*value);
    }
    Ok(out
        .chunks(head_dim)
        .zip(gate.chunks(head_dim))
        .flat_map(|(head, gate)| {
            rms_norm_gated(head, tensors.norm, gate, epsilon, activations.sigmoid)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gated_deltanet_token_reads_its_key_head_and_keeps_its_state() {
        let heads = GdnHeads {
            key_heads: 1,
            value_heads: 2,
            key_dim: 2,
            value_dim: 2,
            taps: 2,
        };
        let width = heads.conv_width();
        assert_eq!(width, 8);
        let conv: Vec<f32> = (0..width * 2).map(|i| 0.1 * (i % 3) as f32 + 0.2).collect();
        let tensors = GdnTensors {
            conv: &conv,
            a_log: &[0.1, -0.2],
            dt_bias: &[0.0, 0.5],
            norm: &[1.0, 0.5],
        };
        let identity = |x: f32| x;
        let softplus = |x: f32| (1.0 + x.exp()).ln();
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let activations = GdnActivations {
            conv: &identity,
            gate: &identity,
            softplus: &softplus,
            sigmoid: &sigmoid,
        };
        let raw: Vec<f32> = (0..width).map(|i| i as f32 * 0.25 - 0.5).collect();
        let mut state = GdnState::default();
        let z = [1.0, 2.0, 0.5, 1.5];
        let out = gdn_token(
            heads,
            tensors,
            (raw.clone(), &z, &[0.3, -0.3], &[0.2, 0.1]),
            &mut state,
            1e-6,
            activations,
        )
        .expect("token");
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|value| value.is_finite()));
        assert_eq!(state.conv_history, vec![raw.clone()]);
        assert_eq!(state.recurrent.len(), 2);
        assert!(state.recurrent.iter().all(|head| head.len() == 4));
        let short = gdn_token(
            heads,
            tensors,
            (raw[1..].to_vec(), &z, &[0.3, -0.3], &[0.2, 0.1]),
            &mut state,
            1e-6,
            activations,
        )
        .unwrap_err();
        assert!(short.contains("fused q/k/v"), "{short}");
    }

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
