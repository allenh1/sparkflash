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

//! A Gated DeltaNet block inside the DeepSeek-shaped stack, on the host.
//!
//! The block's mHC entry, norms and FFN are the stack's; only its token mixer differs: the
//! normalised rows project through `attn.in_proj_qkv`/`z`/`b`/`a` in the dense class, and each
//! row takes the shared host Gated DeltaNet step ([sparkflash_weights::linear_attention::gdn_token]):
//! the causal conv over the block's history, the delta-rule step on its recurrent state and the
//! gated RMSNorm by `z`. `attn.out_proj` projects the normed heads back into BF16 rows. The
//! activations are the ones the Qwen3-Next release and the device's kernels take: SiLU after the
//! conv and on the gate (the config refuses another gate), softplus inside the decay and sigmoid
//! of beta, each the stack's own exact form.

use super::weight_view::WeightView;
use crate::config::{GdnGeometry, ModelConfig};
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::quant;
use crate::weights::WeightSource;
use sparkflash_weights::linear_attention::{
    GdnActivations, GdnHeads, GdnState, GdnTensors, gdn_token,
};

/// One Gated DeltaNet block's state: the conv history and the delta-rule state the shared step
/// keeps, and the positions they cover.
#[derive(Debug, Clone, Default)]
pub struct RecurrentState {
    state: GdnState,
    consumed: usize,
}

impl RecurrentState {
    /// The positions the state covers.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.consumed
    }

    /// Whether the state covers nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.consumed == 0
    }
}

fn shape(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}

const fn heads(geometry: &GdnGeometry) -> GdnHeads {
    GdnHeads {
        key_heads: geometry.key_heads,
        value_heads: geometry.value_heads,
        key_dim: geometry.key_dim,
        value_dim: geometry.value_dim,
        taps: geometry.conv,
    }
}

/// Runs block `layer`'s Gated DeltaNet over `x` (`[rows][dim]`, the normalised block input) at
/// `start_pos`, advancing `state` past every row, and writes the projected rows into `out`
/// (`[rows][dim]`).
///
/// # Errors
///
/// [EngineError::Shape] when the stack states no Gated DeltaNet geometry, the state does not end
/// at `start_pos`, or a weight is not the geometry's shape; a read error from `source`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recurrent_attention_forward(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    state: &mut RecurrentState,
    source: &dyn WeightSource,
    out: &mut Vec<f32>,
    limit: usize,
) -> Result<(), EngineError> {
    let what = format!("layer {layer} Gated DeltaNet");
    let geometry = config.gdn.as_ref().ok_or_else(|| {
        shape(
            &what,
            "the block is linear_attention and the stack states no linear_* geometry".to_owned(),
        )
    })?;
    let dim = config.dim;
    if dim == 0 || !x.len().is_multiple_of(dim) {
        return Err(shape(
            &what,
            format!("{} input values are not whole {dim}-wide rows", x.len()),
        ));
    }
    let rows = x.len() / dim;
    if state.consumed != start_pos {
        return Err(shape(
            &what,
            format!(
                "the state covers {} positions and the chunk starts at {start_pos}",
                state.consumed
            ),
        ));
    }
    let heads = heads(geometry);
    let (width, value_width) = (heads.conv_width(), heads.value_width());
    let grid = config.dense_grid().unwrap_or(ScaleGrid::Block32);
    let project = |name: &str, outputs: usize, inputs: usize, input: &[f32]| {
        let view = WeightView::read_optional_scale(
            source,
            &format!("{prefix}.attn.{name}"),
            outputs,
            inputs,
            grid,
            limit,
        )?;
        let count = input.len() / inputs;
        let mut projected = vec![0.0f32; count * outputs];
        view.matmul(&mut projected, input, outputs, inputs, count)?;
        Ok::<_, EngineError>(projected)
    };
    let mixed = project("in_proj_qkv", width, dim, x)?;
    let gate = project("in_proj_z", value_width, dim, x)?;
    let beta = project("in_proj_b", geometry.value_heads, dim, x)?;
    let decay = project("in_proj_a", geometry.value_heads, dim, x)?;
    let read = |name: &str| linalg::decode_by_name(source, &format!("{prefix}.attn.{name}"), limit);
    let (conv, a_log, dt_bias, norm) = (
        read("conv1d.weight")?,
        read("A_log")?,
        read("dt_bias")?,
        read("norm.weight")?,
    );
    let tensors = GdnTensors {
        conv: &conv,
        a_log: &a_log,
        dt_bias: &dt_bias,
        norm: &norm,
    };
    let activations = GdnActivations {
        conv: &linalg::silu,
        gate: &linalg::silu,
        softplus: &super::moe::softplus,
        sigmoid: &linalg::sigmoid,
    };
    let heads_out = geometry.value_heads;
    let mut attended = Vec::with_capacity(rows * value_width);
    for row in 0..rows {
        let normed = gdn_token(
            heads,
            tensors,
            (
                mixed[row * width..(row + 1) * width].to_vec(),
                &gate[row * value_width..(row + 1) * value_width],
                &beta[row * heads_out..(row + 1) * heads_out],
                &decay[row * heads_out..(row + 1) * heads_out],
            ),
            &mut state.state,
            config.norm_eps as f32,
            activations,
        )
        .map_err(|detail| shape(&what, detail))?;
        attended.extend_from_slice(&normed);
        state.consumed += 1;
    }
    let projected = project("out_proj", dim, value_width, &attended)?;
    // The output narrows to BF16 once, as the device's output projection writes it for the
    // attention all-reduce's BF16 wire.
    out.clear();
    out.extend(projected.iter().map(|value| quant::bf16_round(*value)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::SyntheticWeights;
    use crate::tensor::TensorPlan;

    fn recurrent_tiny() -> ModelConfig {
        ModelConfig {
            gdn_layers: vec![1],
            gdn: Some(GdnGeometry {
                key_heads: 2,
                value_heads: 4,
                key_dim: 16,
                value_dim: 8,
                conv: 4,
            }),
            ..ModelConfig::tiny()
        }
    }

    /// The writer plans the Gated DeltaNet's tensors and none of MLA's at the recurrent block.
    #[test]
    fn the_plan_holds_the_recurrent_block_s_tensors_only() {
        let config = recurrent_tiny();
        let plan = TensorPlan::language_model(&config);
        for name in [
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_b.weight",
            "in_proj_a.weight",
            "out_proj.weight",
            "conv1d.weight",
            "A_log",
            "dt_bias",
            "norm.weight",
        ] {
            assert!(plan.contains(&format!("layers.1.attn.{name}")), "{name}");
            assert!(!plan.contains(&format!("layers.0.attn.{name}")), "{name}");
        }
        for name in ["wq_a.weight", "wkv.weight", "attn_sink", "q_norm.weight"] {
            assert!(!plan.contains(&format!("layers.1.attn.{name}")));
            assert!(plan.contains(&format!("layers.0.attn.{name}")));
        }
        assert!(plan.contains("layers.1.attn_norm.weight"));
    }

    /// One chunk and the same rows in other chunks leave the same bits and the same state: the
    /// scan is a row at a time, so a cut anywhere resumes exactly.
    #[test]
    fn chunks_split_anywhere_and_resume_from_the_state() {
        let config = recurrent_tiny();
        let source = SyntheticWeights::from_plan(&TensorPlan::language_model(&config), 3);
        let limit = 1 << 22;
        let x: Vec<f32> = (0..5 * config.dim)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) * 0.01)
            .collect();
        let run = |cuts: &[(usize, usize)]| {
            let mut state = RecurrentState::default();
            let mut all = Vec::new();
            for &(start, end) in cuts {
                let mut out = Vec::new();
                recurrent_attention_forward(
                    &config,
                    1,
                    "layers.1",
                    &x[start * config.dim..end * config.dim],
                    start,
                    &mut state,
                    &source,
                    &mut out,
                    limit,
                )
                .unwrap();
                all.extend(out.iter().map(|value| value.to_bits()));
            }
            (all, state)
        };
        let (whole, whole_state) = run(&[(0, 5)]);
        let (split, split_state) = run(&[(0, 2), (2, 3), (3, 5)]);
        // The synthetic source's bytes are format-agnostic, so the rows may hold NaN: compare bits.
        let state_bits = |state: &RecurrentState| {
            let GdnState {
                conv_history,
                recurrent,
            } = &state.state;
            conv_history
                .iter()
                .chain(recurrent)
                .flatten()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(whole, split);
        assert_eq!(state_bits(&whole_state), state_bits(&split_state));
        assert!(!state_bits(&whole_state).is_empty());
        assert_eq!(whole_state.len(), 5);
        assert!(whole.iter().any(|bits| *bits != 0));
        // A chunk that does not start where the state ends is refused by name.
        let mut state = RecurrentState::default();
        let error = recurrent_attention_forward(
            &config,
            1,
            "layers.1",
            &x[..config.dim],
            3,
            &mut state,
            &source,
            &mut Vec::new(),
            limit,
        )
        .unwrap_err();
        assert!(error.to_string().contains("chunk starts at 3"), "{error}");
    }

    /// A stack holding a Gated DeltaNet block prefills in chunks with the bits of one whole pass.
    #[test]
    fn a_stack_with_a_recurrent_block_prefills_in_any_chunks() {
        use crate::forward::{InferenceEngine, SequenceState};
        let config = recurrent_tiny();
        let engine = || {
            let source = SyntheticWeights::from_plan(&TensorPlan::language_model(&config), 9);
            InferenceEngine::new(config.clone(), Box::new(source), 1 << 22).unwrap()
        };
        let prompt = [3u32, 17, 42, 5, 99, 1];
        let whole_engine = engine();
        let mut whole = SequenceState::new(&config);
        let one = whole_engine.prefill(&mut whole, &prompt).unwrap();
        let split_engine = engine();
        let mut split = SequenceState::new(&config);
        split_engine.prefill(&mut split, &prompt[..2]).unwrap();
        let two = split_engine.prefill(&mut split, &prompt[2..]).unwrap();
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&one.logits), bits(&two.logits));
        assert_eq!(whole.recurrent[1].len(), prompt.len());
    }
}
