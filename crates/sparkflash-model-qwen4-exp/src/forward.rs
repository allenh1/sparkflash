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

//! The f32 host reference forward.
//!
//! This is the oracle the device arm is compared against, not a serving path. It runs the exact
//! arithmetic of the released implementation at any geometry, so the tiny fixture in the tests
//! exercises every branch: the gated residual read mixer, the Gated DeltaNet recurrence, the QSA
//! block indexer, the NVFP4 expert body, the gated output of linear attention and the n-gram PLE
//! with its dilated convolution.
//!
//! # Layer order (from Qwen4ExpTextDecoderLayer.forward)
//!
//! 1. PLE injection, if this layer carries it: hidden += PLE(hidden, token window)
//! 2. attention hyper-connection: (mixed, hyper_input, injection) = attn_gate(hidden)
//! 3. the branch (Gated DeltaNet or QSA) runs on mixed
//! 4. hidden = hyper_input + branch_out (x) injection, per residual stream
//! 5. the same three steps with the MLP hyper-connection and the MoE block
//!
//! After the last layer the final hyper-connection mixer folds the four streams into one and the
//! LM head scores it. There is deliberately no final RMSNorm: the released model has none, which
//! is why the tensor plan has no model.norm weight.

use sparkflash_weights::linear_attention::{self, Decay};
use sparkflash_weights::topk;

use crate::config::{LayerKind, Qwen4ExpConfig};
use crate::ngram::PleHasher;

/// The error a forward or a weight load produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardError {
    /// What went wrong.
    pub detail: String,
}

impl ForwardError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for ForwardError {}

/// A row-major f32 matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub columns: usize,
    /// Values, row-major.
    pub data: Vec<f32>,
}

impl Mat {
    /// A matrix from row-major data.
    ///
    /// # Panics
    /// When the data length is not rows * columns.
    #[must_use]
    pub fn new(rows: usize, columns: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * columns, "matrix shape and data disagree");
        Self {
            rows,
            columns,
            data,
        }
    }

    /// An all-zero matrix.
    #[must_use]
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            data: vec![0.0; rows * columns],
        }
    }

    /// y = W x.
    #[must_use]
    pub fn matvec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.columns, "matvec input width");
        let mut out = vec![0.0; self.rows];
        for (row, slot) in out.iter_mut().enumerate() {
            let base = row * self.columns;
            let mut accumulator = 0.0f32;
            for (value, activation) in self.data[base..base + self.columns].iter().zip(input) {
                accumulator += value * activation;
            }
            *slot = accumulator;
        }
        out
    }
}

/// One hyper-connection (gated residual) block.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperBlock {
    /// The per-stream norm weight, [hc * hidden].
    pub norm: Vec<f32>,
    /// The bottleneck down projection, [lowrank, hc * hidden].
    pub down: Mat,
    /// The bottleneck up projection, [hc * hidden, lowrank].
    pub up: Mat,
    /// The per-stream injection weight, [hc, hc * hidden]; absent on the final mixer.
    pub inject: Option<Mat>,
}

/// One routed expert's dequantised body.
#[derive(Debug, Clone, PartialEq)]
pub struct Expert {
    /// Gate projection [moe_intermediate, hidden].
    pub gate: Mat,
    /// Up projection [moe_intermediate, hidden].
    pub up: Mat,
    /// Down projection [hidden, moe_intermediate].
    pub down: Mat,
}

/// The MoE block of one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeWeights {
    /// The router [num_experts, hidden].
    pub router: Mat,
    /// The shared expert.
    pub shared: Expert,
    /// The shared-expert gate [1, hidden].
    pub shared_gate: Mat,
    /// The routed experts.
    pub experts: Vec<Expert>,
}

/// The weights of one attention branch.
#[derive(Debug, Clone, PartialEq)]
pub enum BranchWeights {
    /// Gated DeltaNet.
    Linear {
        /// Mixed QKV projection [conv_width, hidden].
        in_qkv: Mat,
        /// Value gate projection [value_width, hidden].
        in_z: Mat,
        /// Beta projection [value_heads, hidden].
        in_b: Mat,
        /// Decay projection [value_heads, hidden].
        in_a: Mat,
        /// Output projection [hidden, value_width].
        out_proj: Mat,
        /// Depthwise convolution weight [conv_width, kernel].
        conv: Vec<f32>,
        /// log(A), [value_heads].
        a_log: Vec<f32>,
        /// dt bias, [value_heads].
        dt_bias: Vec<f32>,
        /// Output-gate norm weight, [value_head_dim].
        norm: Vec<f32>,
    },
    /// Qwen Sparse Attention.
    Sparse {
        /// Query and gate projection [2 * heads * head_dim, hidden].
        q_proj: Mat,
        /// Key projection [kv_heads * head_dim, hidden].
        k_proj: Mat,
        /// Value projection [kv_heads * head_dim, hidden].
        v_proj: Mat,
        /// Output projection [hidden, heads * head_dim].
        o_proj: Mat,
        /// Query norm [head_dim].
        q_norm: Vec<f32>,
        /// Key norm [head_dim].
        k_norm: Vec<f32>,
        /// Indexer projection [(index_q + index_kv) * index_dim, hidden].
        index_qk: Mat,
        /// Indexer query norm [index_dim].
        index_q_norm: Vec<f32>,
        /// Indexer key norm [index_dim].
        index_k_norm: Vec<f32>,
    },
}

/// One decoder layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerWeights {
    /// The attention branch.
    pub branch: BranchWeights,
    /// The attention hyper-connection.
    pub attn_hyper: HyperBlock,
    /// The MLP hyper-connection.
    pub mlp_hyper: HyperBlock,
    /// The MoE block.
    pub moe: MoeWeights,
    /// The PLE block, when this layer carries one.
    pub ple: Option<PleWeights>,
}

/// The PLE projections. The 51.2 GiB n-gram table is not here: it stays behind
/// [crate::ple::PleTable].
#[derive(Debug, Clone, PartialEq)]
pub struct PleWeights {
    /// [hc * hidden, ple_embed_dim].
    pub key_proj: Mat,
    /// [hidden, ple_embed_dim].
    pub value_proj: Mat,
    /// Key norm [hc * hidden].
    pub norm_key: Vec<f32>,
    /// Query norm [hc * hidden].
    pub norm_query: Vec<f32>,
    /// Convolution norm [hc * hidden].
    pub norm_conv: Vec<f32>,
    /// Dilated depthwise convolution [hc * hidden, kernel].
    pub conv: Vec<f32>,
}

/// The complete host reference weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpWeights {
    /// The configuration these weights were built for.
    pub config: Qwen4ExpConfig,
    /// The token embedding [vocab, hidden].
    pub embed: Mat,
    /// The LM head [vocab, hidden].
    pub lm_head: Mat,
    /// The final mixer (no injection).
    pub mixer: HyperBlock,
    /// The decoder layers.
    pub layers: Vec<LayerWeights>,
}

/// The f32 state one sequence carries between steps.
#[derive(Debug, Clone, Default)]
pub struct ReferenceState {
    /// The next absolute position.
    pub position: usize,
    /// Sparse-attention caches, one per sparse layer (None for linear layers).
    pub sparse: Vec<Option<SparseState>>,
    /// Gated DeltaNet convolution and recurrent state, one per linear layer.
    pub gdn: Vec<Option<GdnState>>,
    /// PLE convolution and token window, when the layer carries PLE.
    pub ple: Option<PleState>,
}

/// The key/value and indexer-key history of one sparse layer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseState {
    /// Rotated keys, one row of kv_heads * head_dim per position, rounded to the BF16 cache dtype.
    pub keys: Vec<Vec<f32>>,
    /// Values, one row of kv_heads * head_dim per position, rounded to the BF16 cache dtype.
    pub values: Vec<Vec<f32>>,
    /// Raw indexer keys before the block pooling norm, one idx_head_dim row per position, rounded
    /// to the BF16 cache dtype.
    pub index_keys: Vec<Vec<f32>>,
}

/// The recurrent state of one Gated DeltaNet layer.
#[derive(Debug, Clone, PartialEq)]
pub struct GdnState {
    /// The convolution history [kernel - 1][conv_width].
    pub conv: Vec<Vec<f32>>,
    /// The recurrent state [value_heads][key_dim][value_dim].
    pub recurrent: Vec<Vec<f32>>,
}

/// The state of the PLE convolution and its token window.
#[derive(Debug, Clone)]
pub struct PleState {
    /// The dilated convolution history, [kernel - 1][hc * hidden].
    pub conv: Vec<Vec<f32>>,
    /// The rolling n-gram hasher (it owns the token window and the segment rule).
    pub hasher: PleHasher,
}

/// The cache keeps the model dtype, as the device store does: round one f32 row to BF16 and
/// widen it back, so the host oracle reads the same keys and values the device cache holds.
fn narrow_to_bf16(row: &mut [f32]) {
    for value in row {
        *value = half::bf16::from_f32(*value).to_f32();
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

/// The zero-centred grouped RMSNorm: the released weight is a delta from 1, and group_size
/// splits the row into independent blocks (Qwen4ExpTextRMSNorm with group_size).
fn rms_norm(input: &[f32], weight: &[f32], group_size: usize, eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), weight.len(), "norm width");
    assert!(group_size > 0 && input.len().is_multiple_of(group_size));
    let mut out = vec![0.0; input.len()];
    for start in (0..input.len()).step_by(group_size) {
        let block = &input[start..start + group_size];
        let mean_square: f32 =
            block.iter().map(|value| value * value).sum::<f32>() / group_size as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for offset in 0..group_size {
            out[start + offset] = block[offset] * scale * (1.0 + weight[start + offset]);
        }
    }
    out
}

/// The output gate activation of Gated DeltaNet's gated RMSNorm
/// ([linear_attention::rms_norm_gated]).
///
/// `sigmoid_gate` is `output_gate_type`, **not** `hidden_act`: the reference constructs this norm
/// as `Qwen4ExpTextRMSNormGated(head_v_dim, eps, activation=config.output_gate_type or
/// config.hidden_act)` and the released checkpoint names `"sigmoid"`. Applying `silu` here - which
/// this did, and which the device kernel did too - puts every GDN layer's output at the wrong
/// scale and sign. A device/host parity gate cannot see it, because both arms shared the mistake.
fn output_gate(sigmoid_gate: bool) -> fn(f32) -> f32 {
    if sigmoid_gate { sigmoid } else { silu }
}

/// Partial RoPE: rotate the first rotary_dim entries of every head.
fn apply_rope(heads: &mut [f32], head_dim: usize, rotary_dim: usize, position: usize, theta: f32) {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for index in 0..half {
        let frequency = 1.0 / theta.powf((2 * index) as f32 / rotary_dim as f32);
        let angle = position as f32 * frequency;
        cos[index] = angle.cos();
        sin[index] = angle.sin();
    }
    for head in heads.chunks_mut(head_dim) {
        let mut rotated = vec![0.0f32; rotary_dim];
        for index in 0..half {
            let a = head[index];
            let b = head[index + half];
            rotated[index] = a * cos[index] - b * sin[index];
            rotated[index + half] = a * sin[index] + b * cos[index];
        }
        head[..rotary_dim].copy_from_slice(&rotated);
    }
}

fn hyper_mix(
    block: &HyperBlock,
    hyper_input: &[f32],
    config: &Qwen4ExpConfig,
    inject: bool,
) -> (Vec<f32>, Vec<f32>) {
    let hc = config.hc_count;
    let hidden = config.hidden_size;
    let normed = rms_norm(hyper_input, &block.norm, hidden, config.rms_norm_eps as f32);
    let down = block.down.matvec(&normed);
    let down: Vec<f32> = down.iter().map(|value| silu(value / hc as f32)).collect();
    let up = block.up.matvec(&down);
    let mut mixed = vec![0.0f32; hidden];
    for stream in 0..hc {
        for column in 0..hidden {
            mixed[column] +=
                sigmoid(up[stream * hidden + column]) * normed[stream * hidden + column];
        }
    }
    for value in &mut mixed {
        *value /= hc as f32;
    }
    let injection = if inject {
        let weights = block
            .inject
            .as_ref()
            .expect("an injecting hyper-connection has a block_inject weight");
        weights
            .matvec(&normed)
            .into_iter()
            .map(|value| 2.0 * sigmoid(value / hc as f32))
            .collect()
    } else {
        Vec::new()
    };
    (mixed, injection)
}

fn inject_branch(
    hyper_input: &[f32],
    branch: &[f32],
    injection: &[f32],
    hidden: usize,
) -> Vec<f32> {
    let mut out = hyper_input.to_vec();
    for (stream, weight) in injection.iter().enumerate() {
        for column in 0..hidden {
            out[stream * hidden + column] += branch[column] * weight;
        }
    }
    out
}

fn moe_forward(weights: &MoeWeights, config: &Qwen4ExpConfig, input: &[f32]) -> Vec<f32> {
    let hidden = config.hidden_size;
    let shared = expert_forward(&weights.shared, input);
    let gate = sigmoid(weights.shared_gate.matvec(input)[0]);
    let logits = weights.router.matvec(input);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let total: f32 = probabilities.iter().sum();
    for value in &mut probabilities {
        *value /= total;
    }
    // The Qwen release is a softmax router with no correction bias and no output scale; that is
    // the shared noaux selection with a zero bias and a scale of one. Ties then break toward the
    // lower expert index, as the reference's topk does, instead of following a sort's order.
    let bias = vec![0.0f32; probabilities.len()];
    let mut selected = Vec::with_capacity(config.num_experts_per_tok);
    topk::noaux_top_k(
        &probabilities,
        &bias,
        config.num_experts_per_tok,
        config.norm_topk_prob,
        1.0,
        &mut selected,
    );
    let mut out = vec![0.0f32; hidden];
    for (expert, weight) in selected {
        let body = expert_forward(&weights.experts[expert], input);
        for column in 0..hidden {
            out[column] += weight * body[column];
        }
    }
    for column in 0..hidden {
        out[column] += gate * shared[column];
    }
    out
}

fn expert_forward(expert: &Expert, input: &[f32]) -> Vec<f32> {
    let gate = expert.gate.matvec(input);
    let up = expert.up.matvec(input);
    let activated: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| silu(*gate) * up)
        .collect();
    expert.down.matvec(&activated)
}

fn gated_delta_net(
    weights: &BranchWeights,
    config: &Qwen4ExpConfig,
    rows: &[Vec<f32>],
    state: &mut GdnState,
) -> Vec<Vec<f32>> {
    let BranchWeights::Linear {
        in_qkv,
        in_z,
        in_b,
        in_a,
        out_proj,
        conv,
        a_log,
        dt_bias,
        norm,
    } = weights
    else {
        unreachable!("gated_delta_net called with a sparse branch");
    };
    let hidden = config.hidden_size;
    let key_width = config.linear_key_width();
    let value_width = config.linear_value_width();
    let width = config.linear_conv_width();
    let value_heads = config.linear_num_value_heads;
    let key_dim = config.linear_key_head_dim;
    let value_dim = config.linear_value_head_dim;
    let groups = config.value_groups();
    let taps = config.linear_conv_kernel_dim;
    let raw: Vec<Vec<f32>> = rows.iter().map(|row| in_qkv.matvec(row)).collect();
    let convolved = linear_attention::causal_conv(&raw, &state.conv, conv, width, taps, silu);
    let mut out = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let mixed = &convolved[index];
        let z = in_z.matvec(row);
        let b = in_b.matvec(row);
        let a = in_a.matvec(row);
        let mut queries = vec![0.0f32; value_heads * key_dim];
        let mut keys = vec![0.0f32; value_heads * key_dim];
        for head in 0..value_heads {
            let source = head / groups;
            for dim in 0..key_dim {
                queries[head * key_dim + dim] = mixed[source * key_dim + dim];
                keys[head * key_dim + dim] = mixed[key_width + source * key_dim + dim];
            }
        }
        let values = &mixed[2 * key_width..2 * key_width + value_width];
        // L2-normalise every query and key head, then scale the queries by 1/sqrt(key_dim).
        let mut qn = vec![0.0f32; queries.len()];
        let mut kn = vec![0.0f32; keys.len()];
        let scale = (key_dim as f32).sqrt().recip();
        for ((q, k), (q_out, k_out)) in queries
            .chunks(key_dim)
            .zip(keys.chunks(key_dim))
            .zip(qn.chunks_mut(key_dim).zip(kn.chunks_mut(key_dim)))
        {
            linear_attention::l2_normalise(q, scale, q_out);
            linear_attention::l2_normalise(k, 1.0, k_out);
        }
        let mut core = vec![0.0f32; value_heads * value_dim];
        for head in 0..value_heads {
            let decay = (-(a_log[head].exp()) * softplus(a[head] + dt_bias[head])).exp();
            linear_attention::delta_rule_step(
                &mut state.recurrent[head],
                Decay::Head(decay),
                &kn[head * key_dim..(head + 1) * key_dim],
                &values[head * value_dim..(head + 1) * value_dim],
                sigmoid(b[head]),
                &qn[head * key_dim..(head + 1) * key_dim],
                &mut core[head * value_dim..(head + 1) * value_dim],
            );
        }
        let mut normed = vec![0.0f32; value_heads * value_dim];
        for head in 0..value_heads {
            let gated = linear_attention::rms_norm_gated(
                &core[head * value_dim..(head + 1) * value_dim],
                norm,
                &z[head * value_dim..(head + 1) * value_dim],
                config.rms_norm_eps as f32,
                output_gate(config.output_gate_is_sigmoid()),
            );
            normed[head * value_dim..(head + 1) * value_dim].copy_from_slice(&gated);
        }
        out.push(out_proj.matvec(&normed));
    }
    // Keep the last taps - 1 raw rows for the next step.
    linear_attention::advance_conv_history(&mut state.conv, raw, taps);
    let _ = hidden;
    out
}

fn sparse_attention(
    weights: &BranchWeights,
    config: &Qwen4ExpConfig,
    rows: &[Vec<f32>],
    state: &mut SparseState,
    start_position: usize,
) -> Vec<Vec<f32>> {
    let BranchWeights::Sparse {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        index_qk,
        index_q_norm,
        index_k_norm,
    } = weights
    else {
        unreachable!("sparse_attention called with a linear branch");
    };
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let groups = config.query_groups();
    let rotary = config.rotary_dim();
    let theta = config.rope_theta as f32;
    let eps = config.rms_norm_eps as f32;
    let index_heads = config.indexer_n_heads;
    let index_dim = config.indexer_head_dim;
    let ratio = config.indexer_compress_ratio;
    let block_budget = config.indexer_block_budget();
    let mut out = Vec::with_capacity(rows.len());
    for (offset, row) in rows.iter().enumerate() {
        let position = start_position + offset;
        let qg = q_proj.matvec(row);
        let mut queries = vec![0.0f32; heads * head_dim];
        let mut gates = vec![0.0f32; heads * head_dim];
        for head in 0..heads {
            for dim in 0..head_dim {
                queries[head * head_dim + dim] = qg[head * head_dim * 2 + dim];
                gates[head * head_dim + dim] = qg[head * head_dim * 2 + head_dim + dim];
            }
        }
        // The query norm runs per head before RoPE, which is what the reference does.
        for head in 0..heads {
            let normalized = rms_norm(
                &queries[head * head_dim..(head + 1) * head_dim],
                q_norm,
                head_dim,
                eps,
            );
            queries[head * head_dim..(head + 1) * head_dim].copy_from_slice(&normalized);
        }
        apply_rope(&mut queries, head_dim, rotary, position, theta);
        let mut keys = k_proj.matvec(row);
        for head in 0..kv_heads {
            let normalized = rms_norm(
                &keys[head * head_dim..(head + 1) * head_dim],
                k_norm,
                head_dim,
                eps,
            );
            keys[head * head_dim..(head + 1) * head_dim].copy_from_slice(&normalized);
        }
        apply_rope(&mut keys, head_dim, rotary, position, theta);
        let mut values = v_proj.matvec(row);
        narrow_to_bf16(&mut keys);
        narrow_to_bf16(&mut values);
        state.keys.push(keys.clone());
        state.values.push(values.clone());

        // The indexer: project, normalise each query head, rope the query at its own position,
        // store the raw key, then pool complete 4-token blocks and rope the pooled keys at the
        // block start.
        let index = index_qk.matvec(row);
        let mut index_queries = index[..index_heads * index_dim].to_vec();
        let mut raw_key = index[index_heads * index_dim..].to_vec();
        narrow_to_bf16(&mut raw_key);
        for head in 0..index_heads {
            let normalized = rms_norm(
                &index_queries[head * index_dim..(head + 1) * index_dim],
                index_q_norm,
                index_dim,
                eps,
            );
            index_queries[head * index_dim..(head + 1) * index_dim].copy_from_slice(&normalized);
        }
        apply_rope(
            &mut index_queries,
            index_dim,
            rotary.min(index_dim),
            position,
            theta,
        );
        state.index_keys.push(raw_key);
        let blocks = (position + 1) / ratio;
        let mut scores = vec![0.0f32; blocks];
        if blocks > 0 {
            let mut pooled: Vec<Vec<f32>> = Vec::with_capacity(blocks);
            for block in 0..blocks {
                let start = block * ratio;
                let mut mean = vec![0.0f32; index_dim];
                for key in &state.index_keys[start..start + ratio] {
                    for dim in 0..index_dim {
                        mean[dim] += key[dim];
                    }
                }
                for value in &mut mean {
                    *value /= ratio as f32;
                }
                let normalized = rms_norm(&mean, index_k_norm, index_dim, eps);
                let mut rotated = normalized;
                apply_rope(&mut rotated, index_dim, rotary.min(index_dim), start, theta);
                pooled.push(rotated);
            }
            for (block, key) in pooled.iter().enumerate() {
                let mut score = 0.0f32;
                for head in 0..index_heads {
                    let query = &index_queries[head * index_dim..(head + 1) * index_dim];
                    let mut dot = 0.0f32;
                    for dim in 0..index_dim {
                        dot += query[dim] * key[dim];
                    }
                    score += dot.max(0.0);
                }
                scores[block] = score / (index_dim as f32).sqrt();
            }
        }
        // The best `block_budget` blocks (score descending, then block index ascending) and the
        // incomplete block the query sits in.
        let mut selected: Vec<usize> = Vec::new();
        topk::top_pools(&scores, ratio, block_budget, position, true, &mut selected);
        // The sparse attention itself: GQA over the selected keys, softmax in f32, then the
        // sigmoid output gate.
        let mut attended = vec![0.0f32; heads * head_dim];
        for head in 0..heads {
            let kv_head = head / groups;
            let mut logits = Vec::with_capacity(selected.len());
            for token in &selected {
                let key = &state.keys[*token][kv_head * head_dim..(kv_head + 1) * head_dim];
                let query = &queries[head * head_dim..(head + 1) * head_dim];
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += query[dim] * key[dim];
                }
                logits.push(dot / (head_dim as f32).sqrt());
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut weights: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for value in &mut weights {
                *value /= total;
            }
            for (slot, token) in selected.iter().enumerate() {
                let value = &state.values[*token][kv_head * head_dim..(kv_head + 1) * head_dim];
                for dim in 0..head_dim {
                    attended[head * head_dim + dim] += weights[slot] * value[dim];
                }
            }
            for dim in 0..head_dim {
                let index = head * head_dim + dim;
                attended[index] *= sigmoid(gates[index]);
            }
        }
        out.push(o_proj.matvec(&attended));
    }
    out
}

/// Dilated depthwise causal convolution with SiLU (the PLE short convolution).
///
/// Like the GDN convolution, PyTorch's dilated F.conv1d pairs tap taps - 1 with the current
/// frame and tap k with the frame (taps - 1 - k) * dilation steps back.
fn causal_conv_dilated(
    rows: &[Vec<f32>],
    history: &[Vec<f32>],
    kernel: &[f32],
    width: usize,
    taps: usize,
    dilation: usize,
) -> Vec<Vec<f32>> {
    let state_len = (taps - 1) * dilation;
    let mut frames: Vec<&[f32]> = history.iter().map(|row| row.as_slice()).collect();
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        frames.push(row.as_slice());
        let mut convolved = vec![0.0f32; width];
        for channel in 0..width {
            let mut accumulator = 0.0f32;
            for tap in 0..taps {
                let back = (taps - 1 - tap) * dilation;
                let index = frames.len() - 1 - back;
                let source = frames[index];
                accumulator += kernel[channel * taps + tap] * source[channel];
            }
            convolved[channel] = silu(accumulator);
        }
        out.push(convolved);
        if frames.len() > state_len + 1 {
            frames.remove(0);
        }
    }
    out
}

/// The PLE branch of one layer, including its per-sentence convolution state.
fn ple_forward(
    weights: &PleWeights,
    config: &Qwen4ExpConfig,
    hidden: &[Vec<f32>],
    embeddings: &[Vec<f32>],
    state: &mut PleState,
) -> Vec<Vec<f32>> {
    let hc = config.hc_count;
    let width = config.hyper_width();
    let dim = config.hidden_size;
    let eps = config.rms_norm_eps as f32;
    let taps = config.ple_conv_kernel_size;
    let dilation = config.ngram_size;
    let mut gated = Vec::with_capacity(hidden.len());
    for (row, embedding) in hidden.iter().zip(embeddings) {
        let key = weights.key_proj.matvec(embedding);
        let key = rms_norm(&key, &weights.norm_key, dim, eps);
        let value = weights.value_proj.matvec(embedding);
        let query = rms_norm(row, &weights.norm_query, dim, eps);
        let mut stream_gates = vec![0.0f32; hc];
        for stream in 0..hc {
            let mut dot = 0.0f32;
            for column in 0..dim {
                dot += key[stream * dim + column] * query[stream * dim + column];
            }
            let gate = dot / (dim as f32).sqrt();
            let magnitude = gate.abs().max(1e-6).sqrt();
            stream_gates[stream] = magnitude.copysign(gate);
        }
        let mut flattened = vec![0.0f32; width];
        for stream in 0..hc {
            let gate = sigmoid(stream_gates[stream]);
            for column in 0..dim {
                flattened[stream * dim + column] = gate * value[column];
            }
        }
        let normed = rms_norm(&flattened, &weights.norm_conv, dim, eps);
        gated.push((flattened, normed));
    }
    let normed_rows: Vec<Vec<f32>> = gated.iter().map(|(_, normed)| normed.clone()).collect();
    let convolved = causal_conv_dilated(
        &normed_rows,
        &state.conv,
        &weights.conv,
        width,
        taps,
        dilation,
    );
    let state_len = (taps - 1) * dilation;
    let mut frames: Vec<Vec<f32>> = std::mem::take(&mut state.conv);
    frames.extend(normed_rows);
    if frames.len() > state_len {
        frames.drain(..frames.len() - state_len);
    }
    state.conv = frames;
    gated
        .into_iter()
        .zip(convolved)
        .map(|((flattened, _), conv)| {
            flattened
                .iter()
                .zip(conv)
                .map(|(value, conv)| value + conv)
                .collect()
        })
        .collect()
}

impl Qwen4ExpWeights {
    /// A fresh sequence state for this model.
    #[must_use]
    pub fn new_state(&self) -> ReferenceState {
        let mut state = ReferenceState {
            position: 0,
            sparse: Vec::with_capacity(self.config.num_hidden_layers),
            gdn: Vec::with_capacity(self.config.num_hidden_layers),
            ple: None,
        };
        for layer in &self.layers {
            match layer.branch {
                BranchWeights::Sparse { .. } => {
                    state.sparse.push(Some(SparseState::default()));
                    state.gdn.push(None);
                }
                BranchWeights::Linear { .. } => {
                    state.sparse.push(None);
                    state.gdn.push(Some(GdnState {
                        conv: Vec::new(),
                        recurrent: vec![
                            vec![
                                0.0;
                                self.config.linear_key_head_dim
                                    * self.config.linear_value_head_dim
                            ];
                            self.config.linear_num_value_heads
                        ],
                    }));
                }
            }
        }
        if self.layers.iter().any(|layer| layer.ple.is_some()) {
            let index = self
                .config
                .ple_layer_indices()
                .first()
                .copied()
                .unwrap_or(0);
            let ple_layer_index = self
                .config
                .ple_layer_ids
                .iter()
                .position(|id| *id == index + 1)
                .expect("index is in ple_layer_ids");
            state.ple = Some(PleState {
                conv: vec![
                    vec![0.0; self.config.hyper_width()];
                    (self.config.ple_conv_kernel_size.saturating_sub(1))
                        * self.config.ngram_size
                ],
                hasher: PleHasher::new(&self.config, ple_layer_index).expect("PLE needs an eos"),
            });
        }
        state
    }

    /// Run the reference forward over a chunk of tokens and return one logit row per token.
    ///
    /// ple_embeddings, when the model has PLE, is the decoded n-gram embedding for each token
    /// (the [crate::ple::PleTable] product); None runs the layer with zero PLE contribution, which
    /// is only meaningful for tests.
    ///
    /// # Errors
    /// When a token is outside the vocabulary, the chunk runs past the context window, or the
    /// PLE embedding rows do not match the tokens.
    pub fn forward(
        &self,
        state: &mut ReferenceState,
        tokens: &[u32],
        mask: Option<&[bool]>,
        ple_embeddings: Option<&[Vec<f32>]>,
    ) -> Result<Vec<Vec<f32>>, ForwardError> {
        let config = &self.config;
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if state.position + tokens.len() > config.max_position_embeddings {
            return Err(ForwardError::new(format!(
                "the chunk would end at {} past the context window {}",
                state.position + tokens.len(),
                config.max_position_embeddings
            )));
        }
        for token in tokens {
            if *token as usize >= config.vocab_size {
                return Err(ForwardError::new(format!(
                    "token {token} is outside the vocabulary {}",
                    config.vocab_size
                )));
            }
        }
        let has_ple = self.layers.iter().any(|layer| layer.ple.is_some());
        if has_ple {
            match ple_embeddings {
                Some(rows) if rows.len() == tokens.len() => {
                    for row in rows {
                        if row.len() != config.ple_embed_dim {
                            return Err(ForwardError::new(format!(
                                "a PLE embedding row has {} values, expected {}",
                                row.len(),
                                config.ple_embed_dim
                            )));
                        }
                    }
                }
                Some(_) => {
                    return Err(ForwardError::new(format!(
                        "the PLE needs one embedding per token ({}), got {}",
                        tokens.len(),
                        ple_embeddings.map_or(0, <[Vec<f32>]>::len)
                    )));
                }
                None => {}
            }
        }
        if let Some(ple_state) = &mut state.ple {
            ple_state
                .hasher
                .push(tokens, mask)
                .map_err(ForwardError::new)?;
        }
        let start_position = state.position;
        let mut hyper: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
        for token in tokens {
            let embedding = &self.embed.data
                [*token as usize * config.hidden_size..(*token as usize + 1) * config.hidden_size];
            let mut row = Vec::with_capacity(config.hyper_width());
            for _ in 0..config.hc_count {
                row.extend_from_slice(embedding);
            }
            hyper.push(row);
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if let Some(ple) = &layer.ple {
                let embeddings = ple_embeddings.ok_or_else(|| {
                    ForwardError::new("the model has PLE but no embedding rows were supplied")
                })?;
                let ple_out = ple_forward(
                    ple,
                    config,
                    &hyper,
                    embeddings,
                    state.ple.as_mut().expect("a PLE layer has PLE state"),
                );
                for (row, addition) in hyper.iter_mut().zip(ple_out) {
                    for (value, addition) in row.iter_mut().zip(addition) {
                        *value += addition;
                    }
                }
            }
            let mut mixed_rows = Vec::with_capacity(tokens.len());
            for row in &hyper {
                let (mixed, _) = hyper_mix(&layer.attn_hyper, row, config, true);
                mixed_rows.push(mixed);
            }
            let branch = match &layer.branch {
                BranchWeights::Linear { .. } => gated_delta_net(
                    &layer.branch,
                    config,
                    &mixed_rows,
                    state.gdn[index]
                        .as_mut()
                        .expect("a linear layer has GDN state"),
                ),
                BranchWeights::Sparse { .. } => sparse_attention(
                    &layer.branch,
                    config,
                    &mixed_rows,
                    state.sparse[index]
                        .as_mut()
                        .expect("a sparse layer has a KV state"),
                    start_position,
                ),
            };
            let mut injections = Vec::with_capacity(tokens.len());
            for row in &hyper {
                let (_, injection) = hyper_mix(&layer.attn_hyper, row, config, true);
                injections.push(injection);
            }
            for (row, (branch_row, injection)) in
                hyper.iter_mut().zip(branch.into_iter().zip(injections))
            {
                *row = inject_branch(row, &branch_row, &injection, config.hidden_size);
            }
            let mut mixed_rows: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
            let mut injections: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
            for row in &hyper {
                let (mixed, injection) = hyper_mix(&layer.mlp_hyper, row, config, true);
                mixed_rows.push(mixed);
                injections.push(injection);
            }
            let moe_rows: Vec<Vec<f32>> = mixed_rows
                .iter()
                .map(|row| moe_forward(&layer.moe, config, row))
                .collect();
            for (row, (moe_row, injection)) in
                hyper.iter_mut().zip(moe_rows.into_iter().zip(injections))
            {
                let replaced = inject_branch(row, &moe_row, &injection, config.hidden_size);
                *row = replaced;
            }
        }
        let mut logits = Vec::with_capacity(tokens.len());
        for row in &hyper {
            let (mixed, _) = hyper_mix(&self.mixer, row, config, false);
            logits.push(self.lm_head.matvec(&mixed));
        }
        state.position += tokens.len();
        Ok(logits)
    }

    /// Build deterministic synthetic weights for a configuration.
    ///
    /// Values are small and fixed by the seed, so a test can compare two runs without a
    /// checkpoint. Norm weights are zero (a factor of one, the released zero-centred convention)
    /// except the plain GDN output-gate norm, whose released weight is one.
    #[must_use]
    pub fn synthetic(config: &Qwen4ExpConfig, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let hidden = config.hidden_size;
        let hyper_width = config.hyper_width();
        let mut layer_weights = Vec::with_capacity(config.num_hidden_layers);
        for layer_type in &config.layer_types {
            let branch = match layer_type {
                LayerKind::Linear => {
                    let key_dim = config.linear_key_head_dim;
                    let value_dim = config.linear_value_head_dim;
                    BranchWeights::Linear {
                        in_qkv: rng.mat(config.linear_conv_width(), hidden, 0.08),
                        in_z: rng.mat(config.linear_value_width(), hidden, 0.08),
                        in_b: rng.mat(config.linear_num_value_heads, hidden, 0.02),
                        in_a: rng.mat(config.linear_num_value_heads, hidden, 0.02),
                        out_proj: rng.mat(hidden, config.linear_value_width(), 0.05),
                        conv: rng.vec(
                            config.linear_conv_width() * config.linear_conv_kernel_dim,
                            0.3,
                        ),
                        a_log: (0..config.linear_num_value_heads)
                            .map(|head| (head % 4) as f32 * 0.1)
                            .collect(),
                        dt_bias: vec![0.0; config.linear_num_value_heads],
                        norm: vec![
                            1.0;
                            value_dim.max(key_dim) / value_dim.max(key_dim) * value_dim
                        ],
                    }
                }
                LayerKind::Sparse => {
                    let head_dim = config.head_dim;
                    let index_dim = config.indexer_head_dim;
                    BranchWeights::Sparse {
                        q_proj: rng.mat(config.q_proj_width(), hidden, 0.05),
                        k_proj: rng.mat(config.kv_proj_width(), hidden, 0.05),
                        v_proj: rng.mat(config.kv_proj_width(), hidden, 0.05),
                        o_proj: rng.mat(hidden, config.attn_out_width(), 0.05),
                        q_norm: vec![0.0; head_dim],
                        k_norm: vec![0.0; head_dim],
                        index_qk: rng.mat(
                            (config.indexer_n_heads + config.indexer_kv_heads) * index_dim,
                            hidden,
                            0.05,
                        ),
                        index_q_norm: vec![0.0; index_dim],
                        index_k_norm: vec![0.0; index_dim],
                    }
                }
            };
            let ple = if config.ple_layer_ids.contains(&(layer_weights.len() + 1)) {
                Some(PleWeights {
                    key_proj: rng.mat(hyper_width, config.ple_embed_dim, 0.05),
                    value_proj: rng.mat(hidden, config.ple_embed_dim, 0.05),
                    norm_key: vec![0.0; hyper_width],
                    norm_query: vec![0.0; hyper_width],
                    norm_conv: vec![0.0; hyper_width],
                    conv: rng.vec(hyper_width * config.ple_conv_kernel_size, 0.2),
                })
            } else {
                None
            };
            layer_weights.push(LayerWeights {
                branch,
                attn_hyper: rng.hyper(config, true),
                mlp_hyper: rng.hyper(config, true),
                moe: rng.moe(config),
                ple,
            });
        }
        Self {
            embed: rng.mat(config.vocab_size, hidden, 0.05),
            lm_head: rng.mat(config.vocab_size, hidden, 0.05),
            mixer: rng.hyper(config, false),
            layers: layer_weights,
            config: config.clone(),
        }
    }
}

/// The host reference weight set. A name kept for parity with the other packs.
pub type ReferenceWeights = Qwen4ExpWeights;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    fn next(&mut self) -> f32 {
        self.state = crate::ngram::splitmix64(self.state);
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|_| (self.next() * 2.0 - 1.0) * scale)
            .collect()
    }

    fn mat(&mut self, rows: usize, columns: usize, scale: f32) -> Mat {
        Mat::new(rows, columns, self.vec(rows * columns, scale))
    }

    fn hyper(&mut self, config: &Qwen4ExpConfig, inject: bool) -> HyperBlock {
        let width = config.hyper_width();
        HyperBlock {
            norm: vec![0.0; width],
            down: self.mat(config.hc_lowrank, width, 0.05),
            up: self.mat(width, config.hc_lowrank, 0.05),
            inject: inject.then(|| self.mat(config.hc_count, width, 0.05)),
        }
    }

    fn expert(&mut self, config: &Qwen4ExpConfig) -> Expert {
        Expert {
            gate: self.mat(config.moe_intermediate_size, config.hidden_size, 0.05),
            up: self.mat(config.moe_intermediate_size, config.hidden_size, 0.05),
            down: self.mat(config.hidden_size, config.moe_intermediate_size, 0.05),
        }
    }

    fn moe(&mut self, config: &Qwen4ExpConfig) -> MoeWeights {
        MoeWeights {
            router: self.mat(config.num_experts, config.hidden_size, 0.05),
            shared: self.expert(config),
            shared_gate: self.mat(1, config.hidden_size, 0.05),
            experts: (0..config.num_experts)
                .map(|_| self.expert(config))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> (Qwen4ExpConfig, Qwen4ExpWeights, Vec<Vec<f32>>) {
        let config = Qwen4ExpConfig::tiny();
        let weights = Qwen4ExpWeights::synthetic(&config, 7);
        let mut rng = Rng::new(11);
        let ple = (0..8).map(|_| rng.vec(config.ple_embed_dim, 0.2)).collect();
        (config, weights, ple)
    }

    /// The Gated DeltaNet output gate is `output_gate_type` ("sigmoid"), not `hidden_act` ("silu").
    ///
    /// A golden-value gate on purpose. Device/host parity cannot catch this class of bug - both
    /// arms shared the `silu` assumption - and on the released checkpoint it was worth a flat
    /// logit row and token soup across the 36 linear-attention layers, so the activation itself is
    /// pinned here rather than only exercised through a parity comparison.
    #[test]
    fn the_gdn_output_gate_is_sigmoid_not_silu() {
        let input = [1.0f32, -2.0, 0.5, -0.25, 3.0, -0.75];
        let weight = [1.0f32; 6];
        let gate = [0.0f32, 1.0, -1.0, 4.0, -6.0, 0.25];
        let gated =
            linear_attention::rms_norm_gated(&input, &weight, &gate, 1e-6, output_gate(true));
        let mean_square: f32 = input.iter().map(|value| value * value).sum::<f32>() / 6.0;
        let scale = 1.0 / (mean_square + 1e-6).sqrt();
        for (index, value) in input.iter().enumerate() {
            let expected = value * scale * sigmoid(gate[index]);
            assert!(
                (gated[index] - expected).abs() < 1e-6,
                "index {index}: {} vs {expected}",
                gated[index]
            );
        }
        // ... and the two activations must actually differ here, or this gate proves nothing.
        let silu_gated =
            linear_attention::rms_norm_gated(&input, &weight, &gate, 1e-6, output_gate(false));
        assert!(
            gated
                .iter()
                .zip(&silu_gated)
                .any(|(sigmoid_value, silu_value)| (sigmoid_value - silu_value).abs() > 1e-3),
            "sigmoid and silu agree on this input, so the gate is not discriminating"
        );
    }

    #[test]
    fn the_tiny_reference_forwards_to_finite_logits() {
        let (config, weights, ple) = tiny_model();
        let mut state = weights.new_state();
        let logits = weights
            .forward(&mut state, &[1, 2, 3, 4], None, Some(&ple[..4]))
            .expect("the tiny forward runs");
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), config.vocab_size);
        assert_eq!(state.position, 4);
        assert!(logits.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn a_prefix_produces_the_same_logits_as_its_head() {
        let (_config, weights, ple) = tiny_model();
        let mut full_state = weights.new_state();
        let full = weights
            .forward(&mut full_state, &[1, 2, 3, 4, 5], None, Some(&ple[..5]))
            .expect("the full forward runs");
        let mut prefix_state = weights.new_state();
        let prefix = weights
            .forward(&mut prefix_state, &[1, 2, 3], None, Some(&ple[..3]))
            .expect("the prefix forward runs");
        for (left, right) in prefix.iter().zip(&full) {
            for (a, b) in left.iter().zip(right) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn the_ple_contribution_changes_the_logits() {
        let (config, weights, ple) = tiny_model();
        let mut with = weights.new_state();
        let a = weights
            .forward(&mut with, &[1, 2, 3], None, Some(&ple[..3]))
            .expect("runs with PLE");
        let mut without = weights.new_state();
        let zeros = vec![vec![0.0f32; config.ple_embed_dim]; 3];
        let b = weights
            .forward(&mut without, &[1, 2, 3], None, Some(&zeros))
            .expect("runs with a zero PLE");
        assert_ne!(a, b);
    }

    #[test]
    fn a_chunk_past_the_context_window_is_refused() {
        let (config, weights, ple) = tiny_model();
        let mut state = weights.new_state();
        state.position = config.max_position_embeddings - 1;
        let error = weights
            .forward(&mut state, &[1, 2], None, Some(&ple[..2]))
            .expect_err("the chunk runs past the window");
        assert!(error.to_string().contains("context window"));
    }

    #[test]
    fn an_out_of_vocabulary_token_is_refused() {
        let (config, weights, ple) = tiny_model();
        let mut state = weights.new_state();
        let error = weights
            .forward(
                &mut state,
                &[config.vocab_size as u32],
                None,
                Some(&ple[..1]),
            )
            .expect_err("the token is out of range");
        assert!(error.to_string().contains("vocabulary"));
    }

    #[test]
    fn the_reference_is_deterministic() {
        let (_config, weights, ple) = tiny_model();
        let mut first = weights.new_state();
        let a = weights
            .forward(&mut first, &[4, 5, 6], None, Some(&ple[..3]))
            .expect("runs");
        let mut second = weights.new_state();
        let b = weights
            .forward(&mut second, &[4, 5, 6], None, Some(&ple[..3]))
            .expect("runs");
        assert_eq!(a, b);
    }
}
