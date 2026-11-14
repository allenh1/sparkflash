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

//! The arithmetic oracle for Inkling-2-Small's four new kernel families.
//!
//! The host forward calls these, and `tests/kernels.rs` calls the same functions against the
//! sm_121a kernels. One implementation therefore decides what "right" means on both sides: the
//! decode kernels cannot drift from the prefill path, and a kernel cannot pass by agreeing with
//! a second, hand-written reference in a test.

/// Everything one attention call needs beyond its tensors.
#[derive(Debug, Clone, Copy)]
pub struct AttentionParams {
    /// Query heads.
    pub heads: usize,
    /// Key/value heads.
    pub kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Relative dimensions per head.
    pub d_rel: usize,
    /// Distances the learned relative bank covers.
    pub rel_extent: usize,
    /// The L2 clip on queries and keys, or 0 for none.
    pub qk_clip: f32,
    /// The L2 clip on the relative state, or 0 for none.
    pub rel_clip: f32,
    /// The log-scaling factor on full layers, 1 on sliding ones.
    pub tau: f32,
    /// Whether values are RMS-normalised per head before the mix.
    pub value_norm: bool,
    /// The norm epsilon.
    pub eps: f64,
}

/// Relative attention over a cache already trimmed to the window.
///
/// `keys` and `values` are `[positions, kv_heads * head_dim]` newest last; `q`, `gate` and
/// `q_scale` are `[heads * head_dim]`; `rel` and `rel_scale` are `[heads * d_rel]`; `sink` is
/// `[heads]`; and `rel_bank` is `[d_rel, rel_extent]`. The result is `[heads * head_dim]`,
/// after the sigmoid output gate and before the output projection.
///
/// # Errors
/// A tensor whose length does not match the parameters.
pub fn attention(
    q: &[f32],
    gate: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    rel: &[f32],
    rel_bank: &[f32],
    sink: &[f32],
    q_scale: &[f32],
    k_scale: Option<&[f32]>,
    rel_scale: &[f32],
    params: &AttentionParams,
) -> Result<Vec<f32>, String> {
    let AttentionParams {
        heads,
        kv_heads,
        head_dim,
        d_rel,
        rel_extent,
        qk_clip,
        rel_clip,
        tau,
        value_norm,
        eps,
    } = *params;
    let q_width = heads * head_dim;
    if q.len() != q_width || gate.len() != q_width || q_scale.len() != q_width {
        return Err("the query, gate and q scale must be heads * head_dim wide".to_owned());
    }
    if rel.len() != heads * d_rel || rel_scale.len() != heads * d_rel {
        return Err("the relative state and its scale must be heads * d_rel wide".to_owned());
    }
    if sink.len() != heads {
        return Err("the sink must hold one logit a head".to_owned());
    }
    if rel_bank.len() < d_rel * rel_extent {
        return Err("the relative bank is narrower than d_rel * rel_extent".to_owned());
    }
    if keys.len() != values.len() {
        return Err("the key and value caches disagree".to_owned());
    }

    let mut q = q.to_vec();
    let mut rel = rel.to_vec();
    for head in 0..heads {
        let row = &mut q[head * head_dim..(head + 1) * head_dim];
        for (value, scale) in row
            .iter_mut()
            .zip(&q_scale[head * head_dim..(head + 1) * head_dim])
        {
            *value *= *scale;
        }
        clip_l2(row, qk_clip);
        for value in row.iter_mut() {
            *value *= tau;
        }
        let row = &mut rel[head * d_rel..(head + 1) * d_rel];
        for (value, scale) in row.iter_mut().zip(&rel_scale[head * d_rel..(head + 1) * d_rel]) {
            *value *= *scale;
        }
        clip_l2(row, rel_clip);
        for value in row.iter_mut() {
            *value *= tau;
        }
    }

    let groups = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let positions = keys.len();
    // Keys carry the same L2 clip as queries, at their own per-head gain; the released
    // checkpoint has no learned key scale, so `k_scale` is optional.
    let mut clipped_keys = Vec::with_capacity(positions);
    for key in keys {
        let mut row = key.clone();
        for kv_head in 0..kv_heads {
            let run = &mut row[kv_head * head_dim..(kv_head + 1) * head_dim];
            if let Some(scale) = k_scale {
                for (value, gain) in run
                    .iter_mut()
                    .zip(&scale[kv_head * head_dim..(kv_head + 1) * head_dim])
                {
                    *value *= *gain;
                }
            }
            clip_l2(run, qk_clip);
        }
        clipped_keys.push(row);
    }
    let keys = &clipped_keys;
    let mut output = vec![0f32; q_width];
    for head in 0..heads {
        let kv_head = head / groups;
        let mut scores = Vec::with_capacity(positions);
        for (slot, key) in keys.iter().enumerate() {
            let mut dot = 0.0f32;
            for dim in 0..head_dim {
                dot += q[head * head_dim + dim] * key[kv_head * head_dim + dim];
            }
            let distance = positions - 1 - slot;
            let bias = if distance < rel_extent {
                // rel_bank is [d_rel, rel_extent]: one row per relative dimension.
                let mut value = 0.0f32;
                for dim in 0..d_rel {
                    value += rel[head * d_rel + dim] * rel_bank[dim * rel_extent + distance];
                }
                value
            } else {
                0.0
            };
            scores.push(dot * scale + bias);
        }
        let maximum = scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(sink[head]);
        let mut denominator = (sink[head] - maximum).exp();
        for score in &mut scores {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        for (slot, score) in scores.iter().enumerate() {
            let weight = *score / denominator;
            let mut normalised = Vec::new();
            let value = if value_norm {
                normalised.extend_from_slice(&values[slot][kv_head * head_dim..(kv_head + 1) * head_dim]);
                rms_norm_in_place(&mut normalised, eps);
                &normalised
            } else {
                &values[slot][kv_head * head_dim..(kv_head + 1) * head_dim]
            };
            for dim in 0..head_dim {
                output[head * head_dim + dim] += weight * value[dim];
            }
        }
    }
    for (value, gate) in output.iter_mut().zip(gate) {
        *value *= 1.0 / (1.0 + (-gate).exp());
    }
    Ok(output)
}

/// The static short convolution over several rows: `(conv(x) + x) / sqrt(kernel + 1)`.
///
/// `x` is `[rows, channels]`, `taps` is `[channels, kernel]` in conv1d order (index 0 the
/// oldest tap), and a row before the first uses zeros.
#[must_use]
pub fn short_conv(x: &[f32], taps: &[f32], rows: usize, channels: usize, kernel: usize) -> Vec<f32> {
    let scale = 1.0 / ((kernel + 1) as f32).sqrt();
    let mut out = vec![0f32; rows * channels];
    for row in 0..rows {
        for channel in 0..channels {
            let mut convolution = 0.0f32;
            for tap in 0..kernel {
                if row < tap {
                    continue;
                }
                convolution += taps[channel * kernel + (kernel - 1 - tap)]
                    * x[(row - tap) * channels + channel];
            }
            out[row * channels + channel] =
                (convolution + x[row * channels + channel]) * scale;
        }
    }
    out
}

/// One decode step of the static form. `state` is `[channels, kernel - 1]` oldest first and is
/// advanced in place.
pub fn short_conv_decode(
    x: &[f32],
    taps: &[f32],
    state: &mut [f32],
    channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let scale = 1.0 / ((kernel + 1) as f32).sqrt();
    let history = kernel - 1;
    let mut out = vec![0f32; channels];
    for channel in 0..channels {
        let current = x[channel];
        let mut convolution = 0.0f32;
        for tap in 0..history {
            convolution += taps[channel * kernel + tap] * state[channel * history + tap];
        }
        convolution += taps[channel * kernel + history] * current;
        out[channel] = (convolution + current) * scale;
        for tap in 0..history.saturating_sub(1) {
            state[channel * history + tap] = state[channel * history + tap + 1];
        }
        if history > 0 {
            state[channel * history + history - 1] = current;
        }
    }
    out
}

/// The dynamic short convolution over several rows.
///
/// `static_taps` is `[kernel, channels]` and `dynamic_taps` is `[rows, heads, kernel]`, both in
/// conv1d order; the scale is `1 / sqrt(2 * kernel + 1)`.
#[must_use]
pub fn short_conv_dynamic(
    x: &[f32],
    static_taps: &[f32],
    dynamic_taps: &[f32],
    rows: usize,
    channels: usize,
    kernel: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / ((2 * kernel + 1) as f32).sqrt();
    let heads = channels / head_dim;
    let mut out = vec![0f32; rows * channels];
    for row in 0..rows {
        for channel in 0..channels {
            let head = channel / head_dim;
            let mut convolution = 0.0f32;
            for tap in 0..kernel {
                if row < tap {
                    continue;
                }
                // Distance `tap` back is conv1d index kernel - 1 - tap in both tables.
                let index = kernel - 1 - tap;
                let weight = static_taps[index * channels + channel]
                    + dynamic_taps[(row * heads + head) * kernel + index];
                convolution += weight * x[(row - tap) * channels + channel];
            }
            out[row * channels + channel] = convolution * scale;
        }
    }
    out
}

/// One decode step of the dynamic form.
pub fn short_conv_dynamic_decode(
    x: &[f32],
    static_taps: &[f32],
    dynamic_taps: &[f32],
    state: &mut [f32],
    channels: usize,
    kernel: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / ((2 * kernel + 1) as f32).sqrt();
    let history = kernel - 1;
    let mut out = vec![0f32; channels];
    for channel in 0..channels {
        let head = channel / head_dim;
        let current = x[channel];
        let mut convolution = 0.0f32;
        for tap in 0..history {
            let weight = static_taps[tap * channels + channel]
                + dynamic_taps[head * kernel + tap];
            convolution += weight * state[channel * history + tap];
        }
        let last = static_taps[history * channels + channel] + dynamic_taps[head * kernel + history];
        convolution += last * current;
        out[channel] = convolution * scale;
        for tap in 0..history.saturating_sub(1) {
            state[channel * history + tap] = state[channel * history + tap + 1];
        }
        if history > 0 {
            state[channel * history + history - 1] = current;
        }
    }
    out
}

/// The softcapped SwiGLU: `softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)`.
///
/// The two halves of `gate_up` are `[width]` each; a cap of 0 means no cap.
#[must_use]
pub fn softcapped_swiglu(gate_up: &[f32], width: usize, gate_cap: f32, up_cap: f32) -> Vec<f32> {
    let mut out = vec![0f32; width];
    for column in 0..width {
        let gate = gate_up[column];
        let up = gate_up[width + column];
        let gated = if gate_cap > 0.0 {
            (gate / gate_cap).tanh() * gate_cap
        } else {
            gate
        };
        let capped = if up_cap > 0.0 {
            (up / up_cap).tanh() * up_cap
        } else {
            up
        };
        out[column] = gated * (1.0 / (1.0 + (-gate).exp())) * capped;
    }
    out
}

/// The softcap and unpadded slice of the final logits.
#[must_use]
pub fn logit_softcap(logits: &[f32], rows: usize, padded: usize, unpadded: usize, cap: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * unpadded];
    for row in 0..rows {
        for column in 0..unpadded.min(padded) {
            let value = logits[row * padded + column];
            out[row * unpadded + column] = if cap > 0.0 {
                (value / cap).tanh() * cap
            } else {
                value
            };
        }
    }
    out
}

fn clip_l2(input: &mut [f32], threshold: f32) {
    if threshold <= 0.0 {
        return;
    }
    let norm = input.iter().map(|value| value * value).sum::<f32>().sqrt();
    let scale = (threshold / (norm + 1e-6)).min(1.0);
    for value in input {
        *value *= scale;
    }
}

fn rms_norm_in_place(input: &mut [f32], epsilon: f64) {
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean + epsilon as f32).sqrt();
    for value in input {
        *value *= scale;
    }
}

/// One NVFP4 expert's matrix-vector product: `y = W x`, with `W` dequantised on the fly.
///
/// `codes` is `[rows, columns / 2]` with the **low** nibble the even column, `scales` is
/// `[rows, columns / 16]` of E4M3 multipliers, and `scale_2` is the expert's own
/// `amax / (6 * outer_scale_denominator)`. The release stores one of those per expert, so the
/// factor rides on the expert's output rather than on a shared tensor constant.
///
/// # Errors
/// A tensor whose length does not match the shape.
pub fn nvfp4_matvec(
    codes: &[u8],
    scales: &[u8],
    scale_2: f32,
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, String> {
    let packed = columns.div_ceil(2);
    let scale_columns = columns.div_ceil(16);
    if codes.len() != rows * packed {
        return Err(format!(
            "the NVFP4 codes hold {} bytes, expected {rows}x{packed}",
            codes.len()
        ));
    }
    if scales.len() != rows * scale_columns {
        return Err(format!(
            "the NVFP4 scales hold {} bytes, expected {rows}x{scale_columns}",
            scales.len()
        ));
    }
    if input.len() != columns {
        return Err(format!(
            "the input is {} wide, expected {columns}",
            input.len()
        ));
    }
    let mut out = vec![0f32; rows];
    for row in 0..rows {
        let codes = &codes[row * packed..(row + 1) * packed];
        let scales = &scales[row * scale_columns..(row + 1) * scale_columns];
        let mut value = 0.0f32;
        for (column, weight) in input.iter().enumerate() {
            let byte = codes[column / 2];
            let code = if column % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let scale = sparkflash_weights::quant::f8e4m3_to_f32(scales[column / 16]);
            value += sparkflash_weights::quant::f4e2m1_to_f32(code) * scale * weight;
        }
        out[row] = value * scale_2;
    }
    Ok(out)
}

/// One MXFP8 expert's matrix-vector product.
///
/// `codes` is `[rows, columns]` of E4M3, `scales` is `[rows, columns / block]` of ue8m0
/// codes (a power of two each), and the layer has no per-expert factor.
///
/// # Errors
/// A tensor whose length does not match the shape.
pub fn mxfp8_matvec(
    codes: &[u8],
    scales: &[u8],
    input: &[f32],
    rows: usize,
    columns: usize,
    block: usize,
) -> Result<Vec<f32>, String> {
    let scale_columns = columns.div_ceil(block);
    if codes.len() != rows * columns || scales.len() != rows * scale_columns {
        return Err(format!(
            "the MXFP8 tensors do not match a {rows}x{columns} weight at block {block}"
        ));
    }
    if input.len() != columns {
        return Err(format!(
            "the input is {} wide, expected {columns}",
            input.len()
        ));
    }
    let mut out = vec![0f32; rows];
    for row in 0..rows {
        let codes = &codes[row * columns..(row + 1) * columns];
        let scales = &scales[row * scale_columns..(row + 1) * scale_columns];
        let mut value = 0.0f32;
        for (column, weight) in input.iter().enumerate() {
            let code = sparkflash_weights::quant::f8e4m3_to_f32(codes[column]);
            value += code * ue8m0(scales[column / block]) * weight;
        }
        out[row] = value;
    }
    Ok(out)
}

/// A ue8m0 scale code: the power of two `2^(code - 127)`.
#[must_use]
pub fn ue8m0(code: u8) -> f32 {
    f32::from_bits(u32::from(code) << 23)
}

/// The sigmoid router: a score a expert, a learned bias on the selection, and the selected
/// experts' shares.
///
/// The released router scores with a sigmoid (not a softmax), ranks `score + bias`, and scales
/// the selected scores by `route_scale` over their sum. Ties keep the lower expert index, so the
/// schedule is a function of the scores alone.
#[must_use]
pub fn router_topk(
    logits: &[f32],
    bias: &[f32],
    experts: usize,
    top_k: usize,
    route_scale: f32,
) -> (Vec<usize>, Vec<f32>) {
    let scores: Vec<f32> = logits[..experts]
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect();
    let mut order: Vec<usize> = (0..experts).collect();
    order.sort_by(|left, right| {
        let left_score = scores[*left] + bias.get(*left).copied().unwrap_or(0.0);
        let right_score = scores[*right] + bias.get(*right).copied().unwrap_or(0.0);
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(right))
    });
    order.truncate(top_k);
    let sum: f32 = order.iter().map(|expert| scores[*expert]).sum();
    let shares = order
        .iter()
        .map(|expert| scores[*expert] / (sum + 1e-20) * route_scale)
        .collect();
    (order, shares)
}

/// `out = residual + delta`, elementwise: the layer's additive residual.
///
/// The block-attention topology has two residual forms. At a block boundary the sublayer output
/// *is* the stream, so this is not called; at every other layer the stream is updated here. The
/// addition is one f32 add a element, which is why the kernel and this function can be compared
/// exactly rather than within a tolerance.
#[must_use]
pub fn residual_add(residual: &[f32], delta: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let index = row * columns + column;
            out[index] = residual[index] + delta[index];
        }
    }
    out
}

/// The routed experts' mix: `out = sum over slots of scales[slot] * partials[slot]`.
///
/// One body serves the three expert formats because they differ only in where the per-expert
/// factor comes from: an NVFP4 expert stores `amax / (6 * outer_scale_denominator)`, an MXFP8
/// expert stores nothing, and a BF16 expert is the fixture's dense tensor. The arm multiplies that
/// factor by the router's share and hands the product here, so the mix is a plain scaled sum in
/// slot order - the same sequential f32 order the host forward's combine used, which is why the
/// kernel and this function can be compared exactly.
#[must_use]
pub fn expert_mix(partials: &[f32], scales: &[f32], slots: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0f32; width];
    for slot in 0..slots {
        let scale = scales[slot];
        for column in 0..width {
            out[column] += scale * partials[slot * width + column];
        }
    }
    out
}

/// The BF16 value the key/value cache stores for one f32 projection.
///
/// The attention body reads its cache as BF16, which is the released model's stored precision,
/// so the narrowing is part of the model's arithmetic and not a transport detail: the oracle
/// returns the same rounded value the kernel writes.
#[must_use]
pub fn cache_bf16(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            sparkflash_weights::quant::bf16_to_f32(sparkflash_weights::quant::f32_to_bf16_bits(
                *value,
            ))
        })
        .collect()
}


