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

//! Per-layer boundary capture: the Phase 1 boundary table the device path is diffed against.
//!
//! Spec: `docs/archive/build/phase1-boundary-table.md` (the format, the tensor order, the byte order and the
//! comparison classes) and `docs/archive/build/roadmap-to-serving.md` section 4.5 criteria 1 and 2 (the
//! per-layer table is the Phase 1 deliverable and the perturbation test is what makes it a test).
//! The points captured are the ones `docs/engine/01-forward-order.md` section 3.2 defines - the
//! block order of `Block.forward` (model.py:968-994) - read between the operations rather than
//! inside them.
//!
//! # Asking the forward to record
//!
//! [crate::forward::InferenceEngine::prefill_observed] and
//! [crate::forward::InferenceEngine::decode_observed] run the same forward as
//! [crate::forward::InferenceEngine::prefill] and [crate::forward::InferenceEngine::decode] and
//! return the same [crate::forward::StepOutput]; the only difference is the [BoundaryTable] they
//! return with it. The default entry points pass no table, so nothing in this module runs on them
//! and their behaviour and determinism are untouched: the capture is `None` in the one code path
//! both forms share.
//!
//! One row per layer. A backbone layer takes the layer id `0..n_layers-1`; an MTP stage takes
//! `n_layers + stage`. The MTP rows exist only when the engine runs the stages. This engine skips
//! the MTP pass at `start_pos == 0` (`forward/mod.rs::run`), so a first step's table holds the
//! backbone rows only. The reference differs in exactly that respect: it still runs each stage's
//! attention at `start_pos == 0` to seed the stage's window KV and only then returns `None`
//! (model.py:1044-1052, 1275-1282), so a first-step table records the engine's row contract, not
//! the reference's cache-seeding work.
//!
//! # The step header
//!
//! A table identifies the step it recorded by the token ids the sequence had already consumed
//! ([BoundaryTable::consumed_prefix]), the absolute [BoundaryTable::start_position], the chunk's
//! [BoundaryTable::tokens], and - when an MTP row ran - the sampled [BoundaryTable::mtp_anchor] the
//! stages consumed as their first draft token. The first three are redundant for a sequence (the
//! prefix's length is the start position), and the anchor is not derivable from the rows at all:
//! it is the backbone's sampled output, and the MTP block consumes it before the stage's own
//! boundaries exist. [BoundaryTable::diff] compares all four, so two decode steps that consume the
//! same token at the same position but over different histories are refused instead of compared.
//!
//! # Layout
//!
//! Every tensor is token-major: position 0's values, then position 1's, and so on. Inside a token
//! the mHC stream is copy-major (`hc_mult` copies of `dim` values each), exactly the layout
//! [crate::forward::mhc] uses and `docs/engine/01-forward-order.md` section 3.2 describes. The
//! attention and FFN output rows are one `dim`-long vector per token - the sublayer outputs before
//! the mHC post-mix. The quantised block input is the reference's activation quantiser over the
//! normalised block input: one E4M3 code per element and one E8M0 scale byte per 32-element block
//! (kernel.py:98-124, section 6 point 1 of the forward-order doc), with the row dividing into whole
//! blocks because the reference asserts it (kernel.py:108).
//!
//! [TENSOR_ORDER] is the order a row records its tensors in, and the order the format document lists
//! them in; it is not an implementation detail to be inferred from the code.
//!
//! # What a diff proves, and what it does not
//!
//! [BoundaryTable::diff] compares two tables row by row and tensor by tensor. It reports a row as
//! differing when any of its boundary values differs, so a weight change in a middle layer shows up
//! in that layer's row *and in the rows after it*: the later layers received a different residual
//! stream, and that is a real difference, not noise. The Phase 1 perturbation criterion's "flags
//! layer 17 and no other layer" therefore holds for the injected layer when it is the last layer
//! whose borders the table covers (the shipped test uses eighteen backbone layers, so layer 17 is
//! the last row), or through a replay-based protocol that feeds each layer the same input on both
//! sides. Section 7 of the format document states what the shipped test proves and what a composed
//! diff cannot localise without that replay; the engine deliberately does not pretend a plain
//! end-to-end diff can.

use crate::config::ModelConfig;
use crate::dtype::SCALE_BLOCK;
use crate::linalg::EngineError;
use crate::quant;
use std::io::Write;

/// The block-entry residual stream: `hidden` as the block receives it, `[positions, hc_mult, dim]`.
///
/// The last boundary of the previous layer and this layer's input are the same value; recording it
/// twice keeps one row self-contained, so a comparison of a device row does not depend on whether
/// the previous row was compared. (The perturbation test asserts the two agree.)
pub const RESIDUAL_IN: &str = "residual_in";

/// The input mix consumed by the attention sublayer's `hc_pre`: `[positions, hc_mult]`.
///
/// This is the previous sublayer's `pre` coefficient set - the FFN predictor of the previous block,
/// or the identity one-hot before the first block (model.py:983, 1260; section 5 of
/// `docs/architecture/02-mhc.md`).
pub const HC_IN_PRE: &str = "hc_in_pre";

/// The attention sublayer's input coefficient set A: `[positions, hc_mult]` (kernel.py:427).
pub const HC_ATTN_PRE: &str = "hc_attn_pre";

/// The attention sublayer's post coefficient set B: `[positions, hc_mult]` (kernel.py:429).
pub const HC_ATTN_POST: &str = "hc_attn_post";

/// The attention sublayer's residual coefficient set C: `[positions, hc_mult, hc_mult]`
/// (kernel.py:430-431, 436-458), row-major with the source copy as the row and the destination as
/// the column - the released `comb[j][k]` orientation, which [crate::forward::mhc::hc_post] applies
/// as `sum_j comb[j][k] * residual[j]` (`docs/architecture/02-mhc.md:91-99`;
/// `crates/sparkflash-kernels/src/mega_mhc_ref.rs:58-67`).
pub const HC_ATTN_COMB: &str = "hc_attn_comb";

/// The normalised attention block input, `[positions, dim]`: the `dim`-long value per token that
/// the attention kernel consumes (model.py:984-985).
pub const PRE_ATTENTION_HIDDEN: &str = "pre_attention_hidden";

/// The quantised attention block input: one E4M3 code per element, `[positions, dim]`.
pub const PRE_ATTENTION_XHAT: &str = "pre_attention_xhat";

/// The attention block input's scales: one E8M0 code per 32-element block, `[positions, dim / 32]`.
pub const PRE_ATTENTION_SCALES: &str = "pre_attention_scales";

/// The attention sublayer's output, `[positions, dim]`, before its mHC post-mix (model.py:985).
pub const ATTENTION_OUT: &str = "attention_out";

/// The stream after the attention sublayer's mHC post-mix, `[positions, hc_mult, dim]`
/// (model.py:986).
pub const POST_ATTENTION_HIDDEN: &str = "post_attention_hidden";

/// The FFN sublayer's input coefficient set A: `[positions, hc_mult]` (kernel.py:427).
pub const HC_FFN_PRE: &str = "hc_ffn_pre";

/// The FFN sublayer's post coefficient set B: `[positions, hc_mult]` (kernel.py:429).
pub const HC_FFN_POST: &str = "hc_ffn_post";

/// The FFN sublayer's residual coefficient set C: `[positions, hc_mult, hc_mult]`, laid out as
/// [HC_ATTN_COMB] is (kernel.py:430-431, 436-458).
pub const HC_FFN_COMB: &str = "hc_ffn_comb";

/// The normalised FFN block input, `[positions, dim]` (model.py:990-992).
pub const PRE_FFN_HIDDEN: &str = "pre_ffn_hidden";

/// The quantised FFN block input: one E4M3 code per element, `[positions, dim]`.
pub const PRE_FFN_XHAT: &str = "pre_ffn_xhat";

/// The FFN block input's scales: one E8M0 code per 32-element block, `[positions, dim / 32]`.
pub const PRE_FFN_SCALES: &str = "pre_ffn_scales";

/// The FFN sublayer's output, `[positions, dim]`, before its mHC post-mix (model.py:992).
pub const FFN_OUT: &str = "ffn_out";

/// The stream after the FFN sublayer's mHC post-mix, `[positions, hc_mult, dim]` (model.py:993):
/// the block's output and the next block's [RESIDUAL_IN].
pub const RESIDUAL_OUT: &str = "residual_out";

/// Every tensor of one row, in the order the forward records them and the format document lists
/// them.
pub const TENSOR_ORDER: [&str; 18] = [
    RESIDUAL_IN,
    HC_IN_PRE,
    HC_ATTN_PRE,
    HC_ATTN_POST,
    HC_ATTN_COMB,
    PRE_ATTENTION_HIDDEN,
    PRE_ATTENTION_XHAT,
    PRE_ATTENTION_SCALES,
    ATTENTION_OUT,
    POST_ATTENTION_HIDDEN,
    HC_FFN_PRE,
    HC_FFN_POST,
    HC_FFN_COMB,
    PRE_FFN_HIDDEN,
    PRE_FFN_XHAT,
    PRE_FFN_SCALES,
    FFN_OUT,
    RESIDUAL_OUT,
];

/// The element format of one boundary tensor.
///
/// The host reference records `f32` values and the quantised block input's raw code bytes; the
/// device harness emits the same two kinds. A tensor's format is part of the format document, not a
/// per-file decision: the name says which one it is.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryData {
    /// Host-side f32 values, in element order.
    F32(Vec<f32>),
    /// Raw storage bytes - the E4M3 codes and the E8M0 scales of a quantised block input.
    U8(Vec<u8>),
}

impl BoundaryData {
    /// The values when this is an f32 tensor.
    #[must_use]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::U8(_) => None,
        }
    }

    /// The bytes when this is a code or scale tensor.
    #[must_use]
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(bytes) => Some(bytes),
            Self::F32(_) => None,
        }
    }

    /// How many elements this tensor holds.
    #[must_use]
    pub fn elements(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U8(bytes) => bytes.len(),
        }
    }

    /// Whether two series are byte-for-byte identical: f32 arrays compare their bits, so `-0.0`
    /// and `+0.0` are different, and code arrays compare their bytes. This is the identity the
    /// format document's "byte for byte" wording means; the derived `PartialEq` is numeric for
    /// f32 and must not be used where that wording is tested.
    #[must_use]
    pub fn bit_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(left), Self::F32(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            }
            (Self::U8(left), Self::U8(right)) => left == right,
            (Self::F32(_), Self::U8(_)) | (Self::U8(_), Self::F32(_)) => false,
        }
    }
}

/// One boundary tensor: its format name, its shape, and its values.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryTensor {
    name: &'static str,
    shape: Vec<usize>,
    data: BoundaryData,
}

impl BoundaryTensor {
    /// The tensor's format name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The tensor's shape, in element order: `[positions, hc_mult, dim]` and so on.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The tensor's values or storage bytes.
    #[must_use]
    pub fn data(&self) -> &BoundaryData {
        &self.data
    }
}

/// One layer's boundary row: the tensors of [TENSOR_ORDER] as the reference recorded them.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerBoundary {
    layer: usize,
    tensors: Vec<BoundaryTensor>,
}

impl LayerBoundary {
    /// An empty row for `layer`; the forward fills it in the [TENSOR_ORDER] order.
    pub fn new(layer: usize) -> Self {
        Self {
            layer,
            tensors: Vec::with_capacity(TENSOR_ORDER.len()),
        }
    }

    /// The layer id: `0..n_layers-1` is a backbone block, `n_layers + stage` an MTP stage.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// The row's tensors, in [TENSOR_ORDER] order.
    #[must_use]
    pub fn tensors(&self) -> &[BoundaryTensor] {
        &self.tensors
    }

    /// One tensor of the row by its format name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&BoundaryTensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Appends a tensor, refusing a shape whose element count is not the value count. The shape
    /// check keeps the format document and the captured bytes from drifting apart silently.
    pub fn push(
        &mut self,
        name: &'static str,
        shape: Vec<usize>,
        data: BoundaryData,
    ) -> Result<(), EngineError> {
        let declared: usize = shape.iter().try_fold(1usize, |acc, &side| {
            acc.checked_mul(side).ok_or(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!("{name}: the shape {shape:?} overflows usize"),
            })
        })?;
        if declared != data.elements() {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!(
                    "{name}: the shape {shape:?} declares {declared} elements, but {} values were recorded",
                    data.elements()
                ),
            });
        }
        self.tensors.push(BoundaryTensor { name, shape, data });
        Ok(())
    }

    /// Appends an mHC stream tensor: `positions` tokens of `hc_mult` copies of `dim` values,
    /// `[positions, hc_mult, dim]`.
    pub fn push_stream(
        &mut self,
        config: &ModelConfig,
        name: &'static str,
        values: Vec<f32>,
    ) -> Result<(), EngineError> {
        let width = config.hc_width();
        if width == 0 || !values.len().is_multiple_of(width) {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!(
                    "{name}: an mHC stream is whole {} value tokens, got {}",
                    width,
                    values.len()
                ),
            });
        }
        let positions = values.len() / width;
        self.push(
            name,
            vec![positions, config.hc_mult, config.dim],
            BoundaryData::F32(values),
        )
    }

    /// Appends one pre or post coefficient vector, `[positions, hc_mult]`.
    ///
    /// The comb sets have their own helper ([Self::push_comb]) rather than a `per_token` argument
    /// here: with `hc_mult == 1` the per-token pre/post size and the per-token comb size are both
    /// 1, so a size-chosen shape silently records a 2-D comb.
    pub fn push_mixes(
        &mut self,
        config: &ModelConfig,
        name: &'static str,
        values: Vec<f32>,
    ) -> Result<(), EngineError> {
        let hc = config.hc_mult;
        if hc == 0 || !values.len().is_multiple_of(hc) {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!(
                    "{name}: a coefficient set is whole {hc}-value tokens, got {}",
                    values.len()
                ),
            });
        }
        let positions = values.len() / hc;
        self.push(name, vec![positions, hc], BoundaryData::F32(values))
    }

    /// Appends one comb coefficient set: always `[positions, hc_mult, hc_mult]`, even when
    /// `hc_mult == 1`, because the released comb is a matrix per token
    /// (`docs/archive/build/phase1-boundary-table.md` section 3).
    pub fn push_comb(
        &mut self,
        config: &ModelConfig,
        name: &'static str,
        values: Vec<f32>,
    ) -> Result<(), EngineError> {
        let hc = config.hc_mult;
        let per_token = hc * hc;
        if hc == 0 || per_token == 0 || !values.len().is_multiple_of(per_token) {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!(
                    "{name}: a comb set is whole {per_token}-value tokens, got {}",
                    values.len()
                ),
            });
        }
        let positions = values.len() / per_token;
        self.push(name, vec![positions, hc, hc], BoundaryData::F32(values))
    }

    /// Appends a dense `[positions, dim]` f32 tensor: a sublayer input or output before the mHC
    /// post-mix.
    pub fn push_dense(
        &mut self,
        config: &ModelConfig,
        name: &'static str,
        values: Vec<f32>,
    ) -> Result<(), EngineError> {
        if config.dim == 0 || !values.len().is_multiple_of(config.dim) {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!(
                    "{name}: a dense row is whole {} value tokens, got {}",
                    config.dim,
                    values.len()
                ),
            });
        }
        let positions = values.len() / config.dim;
        self.push(name, vec![positions, config.dim], BoundaryData::F32(values))
    }

    /// Appends the normalised block input as three tensors: the f32 values the reference computes,
    /// the E4M3 codes the device kernel consumes, and the E8M0 scale byte of each 32-element block.
    ///
    /// The quantiser is the reference's activation quantiser, [quant::quantize_activation_fp8] over
    /// each token's row - the same rule the forward applies before an FP8 weight (kernel.py:98-124,
    /// section 6 point 1 of `docs/engine/01-forward-order.md`). The row must divide into whole
    /// 32-element blocks, which is the reference's own assertion (kernel.py:108); the observed entry
    /// points refuse a configuration whose `dim` does not, before a weight is read.
    pub fn push_block_input(
        &mut self,
        config: &ModelConfig,
        hidden_name: &'static str,
        xhat_name: &'static str,
        scales_name: &'static str,
        values: &[f32],
    ) -> Result<(), EngineError> {
        let dim = config.dim;
        if dim == 0 || !dim.is_multiple_of(SCALE_BLOCK) {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!(
                    "{hidden_name}: the quantised block input needs whole {SCALE_BLOCK}-element blocks, dim is {dim}"
                ),
            });
        }
        if !values.len().is_multiple_of(dim) {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!(
                    "{hidden_name}: a block input row is {dim} values, got {}",
                    values.len()
                ),
            });
        }
        let positions = values.len() / dim;
        let mut codes = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(positions * (dim / SCALE_BLOCK));
        for row in values.chunks_exact(dim) {
            let quantised = quant::quantize_activation_fp8(row)?;
            codes.extend_from_slice(quantised.codes());
            scales.extend_from_slice(quantised.scales());
        }
        self.push(
            hidden_name,
            vec![positions, dim],
            BoundaryData::F32(values.to_vec()),
        )?;
        self.push(xhat_name, vec![positions, dim], BoundaryData::U8(codes))?;
        self.push(
            scales_name,
            vec![positions, dim / SCALE_BLOCK],
            BoundaryData::U8(scales),
        )
    }
}

/// One step's boundary table: the step's header (the tokens already consumed, the first position,
/// the chunk's tokens and the MTP anchor when the stages ran) and one row per layer that ran.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryTable {
    start_position: usize,
    tokens: Vec<u32>,
    consumed_prefix: Vec<u32>,
    mtp_anchor: Option<u32>,
    rows: Vec<LayerBoundary>,
}

impl BoundaryTable {
    /// An empty table for the step that consumes `tokens` at `start_position`, after a sequence
    /// that had already consumed `consumed_prefix`.
    ///
    /// The constructor does not require `consumed_prefix.len() == start_position`: the engine always
    /// passes matching values, and [Self::diff] compares the two independently so a capture bug in
    /// either field is caught. The unit tests construct a deliberately inconsistent pair to prove
    /// that the start-position check is not shadowed by the prefix check.
    pub fn new(start_position: usize, tokens: &[u32], consumed_prefix: &[u32]) -> Self {
        Self {
            start_position,
            tokens: tokens.to_vec(),
            consumed_prefix: consumed_prefix.to_vec(),
            mtp_anchor: None,
            rows: Vec::new(),
        }
    }

    /// The first position the step consumed.
    #[must_use]
    pub fn start_position(&self) -> usize {
        self.start_position
    }

    /// The tokens the step consumed.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The token ids the sequence had already consumed before this step, in order.
    ///
    /// A decode table depends on this prefix through the window ring and the compressed/index
    /// caches, which is why [Self::diff] compares it even when the start position and chunk tokens
    /// agree.
    #[must_use]
    pub fn consumed_prefix(&self) -> &[u32] {
        &self.consumed_prefix
    }

    /// The backbone-sampled token the MTP stages consumed as their first draft token, or `None`
    /// when no MTP row ran.
    ///
    /// The stages put this token at draft position 0 (model.py:1131-1132; `mtp.rs` assigns
    /// `ids[0]`), so an MTP row is a function of it; it is not a boundary tensor and cannot be
    /// recovered from the row.
    #[must_use]
    pub fn mtp_anchor(&self) -> Option<u32> {
        self.mtp_anchor
    }

    /// Records the anchor the MTP stages consume. `forward::run` calls this immediately before the
    /// stages run, so a table with no MTP rows keeps `None`.
    pub(crate) fn set_mtp_anchor(&mut self, anchor: u32) {
        self.mtp_anchor = Some(anchor);
    }

    /// The rows, in the order the forward ran them: backbone layers ascending, then MTP stages.
    #[must_use]
    pub fn rows(&self) -> &[LayerBoundary] {
        &self.rows
    }

    /// The row of one layer id.
    #[must_use]
    pub fn row(&self, layer: usize) -> Option<&LayerBoundary> {
        self.rows.iter().find(|row| row.layer == layer)
    }

    /// How many rows the table holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the forward recorded nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Appends one row. The forward appends in its own order; a duplicate layer id would make a
    /// diff ambiguous, so it is refused here rather than compared arbitrarily later.
    pub fn push_row(&mut self, row: LayerBoundary) -> Result<(), EngineError> {
        if self.rows.iter().any(|held| held.layer == row.layer) {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!("layer {} was recorded twice in one step", row.layer),
            });
        }
        self.rows.push(row);
        Ok(())
    }

    /// Compares this table (the reference, or the left side) with `other` row by row.
    ///
    /// The two tables must describe the same step - same consumed prefix, same start position, same
    /// chunk tokens, same MTP anchor, same layer ids, same tensor names in the same order, same
    /// shapes - because comparing different steps is a wiring error rather than a difference. The
    /// prefix and the anchor are the two header fields a start/token comparison cannot see: a decode
    /// step's attention reads the window ring and the compressed/index caches built by the prefix,
    /// and an MTP stage consumes the anchor as its first draft token, so the same token at the same
    /// position over two different histories is two different steps.
    ///
    /// Within a tensor, two values differ when their bits differ: the host reference is
    /// deterministic, so identical weights and input are bit-identical (the determinism the
    /// perturbation test relies on), and a device comparison uses the comparison classes of the
    /// format document against [BoundaryTensorDiff::max_abs_delta] instead.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the two tables do not describe the same step, layer or tensor
    /// layout.
    pub fn diff(&self, other: &Self) -> Result<BoundaryDiff, EngineError> {
        if self.start_position != other.start_position {
            return Err(EngineError::Shape {
                what: "boundary diff".to_owned(),
                detail: format!(
                    "the tables describe different steps: position {} against position {}",
                    self.start_position, other.start_position
                ),
            });
        }
        if self.tokens != other.tokens {
            return Err(EngineError::Shape {
                what: "boundary diff".to_owned(),
                detail: format!(
                    "the tables describe different steps: tokens {:?} against tokens {:?}",
                    self.tokens, other.tokens
                ),
            });
        }
        if self.consumed_prefix != other.consumed_prefix {
            return Err(EngineError::Shape {
                what: "boundary diff".to_owned(),
                detail: format!(
                    "the tables describe different steps: consumed prefix {:?} against prefix {:?}",
                    self.consumed_prefix, other.consumed_prefix
                ),
            });
        }
        if self.mtp_anchor != other.mtp_anchor {
            return Err(EngineError::Shape {
                what: "boundary diff".to_owned(),
                detail: format!(
                    "the tables describe different steps: MTP anchor {:?} against anchor {:?}",
                    self.mtp_anchor, other.mtp_anchor
                ),
            });
        }
        if self.rows.len() != other.rows.len() {
            return Err(EngineError::Shape {
                what: "boundary diff".to_owned(),
                detail: format!("{} rows against {} rows", self.rows.len(), other.rows.len()),
            });
        }
        let mut rows = Vec::with_capacity(self.rows.len());
        for (left, right) in self.rows.iter().zip(other.rows.iter()) {
            if left.layer != right.layer {
                return Err(EngineError::Shape {
                    what: "boundary diff".to_owned(),
                    detail: format!(
                        "row {} against row {}: the rows are in a different layer order",
                        left.layer, right.layer
                    ),
                });
            }
            if left.tensors.len() != right.tensors.len() {
                return Err(EngineError::Shape {
                    what: "boundary diff".to_owned(),
                    detail: format!(
                        "layer {}: {} tensors against {} tensors",
                        left.layer,
                        left.tensors.len(),
                        right.tensors.len()
                    ),
                });
            }
            let mut tensors = Vec::with_capacity(left.tensors.len());
            for (a, b) in left.tensors.iter().zip(right.tensors.iter()) {
                if a.name != b.name || a.shape != b.shape {
                    return Err(EngineError::Shape {
                        what: "boundary diff".to_owned(),
                        detail: format!(
                            "layer {}: tensor {} {:?} against tensor {} {:?}",
                            left.layer, a.name, a.shape, b.name, b.shape
                        ),
                    });
                }
                let (differing, max_abs_delta) =
                    compare_data(&a.data, &b.data).ok_or(EngineError::Shape {
                        what: "boundary diff".to_owned(),
                        detail: format!(
                            "layer {}: tensor {} was recorded in two different element formats",
                            left.layer, a.name
                        ),
                    })?;
                tensors.push(BoundaryTensorDiff {
                    name: a.name,
                    differing,
                    max_abs_delta,
                });
            }
            rows.push(BoundaryRowDiff {
                layer: left.layer,
                tensors,
            });
        }
        Ok(BoundaryDiff { rows })
    }
}

/// Compares two equal-shaped element series, bit for bit. The element count is reported for
/// reporting and bars; the maximum absolute delta is `f32::INFINITY` when the differing pair is not
/// ordered (a NaN on one side), so a NaN difference cannot pass a numeric bar by looking small.
fn compare_data(left: &BoundaryData, right: &BoundaryData) -> Option<(usize, f32)> {
    match (left, right) {
        (BoundaryData::F32(a), BoundaryData::F32(b)) => {
            let mut differing = 0usize;
            let mut max_abs_delta = 0.0f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                if x.to_bits() != y.to_bits() {
                    differing += 1;
                    let delta = (x - y).abs();
                    if delta.is_nan() {
                        max_abs_delta = f32::INFINITY;
                    } else if delta > max_abs_delta {
                        max_abs_delta = delta;
                    }
                }
            }
            Some((differing, max_abs_delta))
        }
        (BoundaryData::U8(a), BoundaryData::U8(b)) => {
            let mut differing = 0usize;
            let mut max_abs_delta = 0.0f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                if x != y {
                    differing += 1;
                    let delta = f32::from(x.abs_diff(y));
                    if delta > max_abs_delta {
                        max_abs_delta = delta;
                    }
                }
            }
            Some((differing, max_abs_delta))
        }
        (BoundaryData::F32(_), BoundaryData::U8(_))
        | (BoundaryData::U8(_), BoundaryData::F32(_)) => None,
    }
}

/// The comparison of two boundary tables, row by row.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryDiff {
    rows: Vec<BoundaryRowDiff>,
}

impl BoundaryDiff {
    /// The row comparisons, in the tables' row order.
    #[must_use]
    pub fn rows(&self) -> &[BoundaryRowDiff] {
        &self.rows
    }

    /// Whether every boundary of every row is bit-identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.rows.iter().all(BoundaryRowDiff::is_identical)
    }

    /// The layer ids whose rows hold at least one differing value, in row order.
    #[must_use]
    pub fn differing_layers(&self) -> Vec<usize> {
        self.rows
            .iter()
            .filter(|row| !row.is_identical())
            .map(|row| row.layer)
            .collect()
    }

    /// The first layer whose row differs, or `None` when the tables are identical: the layer a
    /// composed run attributes the divergence to, before any replay narrows it further.
    #[must_use]
    pub fn first_differing(&self) -> Option<usize> {
        self.rows
            .iter()
            .find(|row| !row.is_identical())
            .map(|row| row.layer)
    }
}

/// The comparison of one layer row.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryRowDiff {
    layer: usize,
    tensors: Vec<BoundaryTensorDiff>,
}

impl BoundaryRowDiff {
    /// The layer id of the compared row.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// The tensor comparisons, in [TENSOR_ORDER] order.
    #[must_use]
    pub fn tensors(&self) -> &[BoundaryTensorDiff] {
        &self.tensors
    }

    /// One tensor's comparison by its format name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&BoundaryTensorDiff> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Whether every tensor of the row is bit-identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.tensors.iter().all(|tensor| tensor.differing == 0)
    }

    /// How many values in the whole row differ.
    #[must_use]
    pub fn differing(&self) -> usize {
        self.tensors.iter().map(|tensor| tensor.differing).sum()
    }

    /// The largest delta of any differing value in the row, or 0 when the row is identical.
    ///
    /// This is the number a tolerance comparison class checks against its bar; the format document
    /// records which classes are bit-exact and which are tolerance classes.
    ///
    /// **The unit is per tensor, and it is not always a value delta.** For an f32 tensor it is the
    /// absolute difference `|left - right|`; for a u8 code or scale tensor it is the absolute
    /// integer distance between the two codes ([BoundaryTensorDiff::max_abs_delta] states this too).
    /// A row-level maximum is therefore meaningful within one comparison class, and the per-tensor
    /// values are what a bar is checked against; do not read a code distance as a numeric delta.
    #[must_use]
    pub fn max_abs_delta(&self) -> f32 {
        self.tensors
            .iter()
            .map(|tensor| tensor.max_abs_delta)
            .fold(0.0f32, f32::max)
    }
}

/// The comparison of one tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryTensorDiff {
    name: &'static str,
    differing: usize,
    max_abs_delta: f32,
}

impl BoundaryTensorDiff {
    /// The tensor's format name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// How many elements differ (by bits for f32, by code for u8).
    #[must_use]
    pub fn differing(&self) -> usize {
        self.differing
    }

    /// The largest delta of a differing element, or 0 when no element differs.
    ///
    /// For an f32 tensor this is the absolute value difference, and it is infinite for a pair that
    /// includes a NaN. For a u8 code or scale tensor it is the largest absolute integer code
    /// distance (for example 2 for codes `0x10` and `0x12`): a unitless count, not a value delta.
    #[must_use]
    pub fn max_abs_delta(&self) -> f32 {
        self.max_abs_delta
    }
}
/// The v1 format token and element width of one boundary tensor.
///
/// The name decides the format (section 2 and section 4 of docs/archive/build/phase1-boundary-table.md): the
/// two quantised block inputs are E4M3 code bytes, their companions are E8M0 scale bytes, and every
/// other tensor is f32. A name outside TENSOR_ORDER has no token.
fn v1_format(name: &str) -> Option<(&'static str, usize)> {
    match name {
        PRE_ATTENTION_XHAT | PRE_FFN_XHAT => Some(("u8-e4m3", 1)),
        PRE_ATTENTION_SCALES | PRE_FFN_SCALES => Some(("u8-e8m0", 1)),
        _ if TENSOR_ORDER.contains(&name) => Some(("f32", 4)),
        _ => None,
    }
}

/// The bytes one boundary tensor contributes to the v1 blob: f32 values little-endian, code bytes
/// as they are.
fn v1_bytes(data: &BoundaryData) -> Vec<u8> {
    match data {
        BoundaryData::F32(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        BoundaryData::U8(codes) => codes.clone(),
    }
}

/// A typed refusal for a manifest that does not follow the v1 format.
fn v1_bad(detail: impl Into<String>) -> EngineError {
    EngineError::Shape {
        what: "boundary v1".to_owned(),
        detail: detail.into(),
    }
}

fn v1_number<T: std::str::FromStr>(token: Option<&str>, field: &str) -> Result<T, EngineError> {
    let token = token.ok_or_else(|| v1_bad(format!("a {field} field is missing")))?;
    token
        .parse()
        .map_err(|_| v1_bad(format!("{token:?} is not a valid {field}")))
}

impl BoundaryTable {
    /// Writes this step in the version 1 manifest-plus-blob format of
    /// docs/archive/build/phase1-boundary-table.md section 4.
    ///
    /// The manifest is text: one header line per field (the version, the step, the prefix, the MTP
    /// anchor and the backbone layer count), one row line per row and one tensor line per tensor in
    /// TENSOR_ORDER order. The blob holds the tensors in the same order with no padding, which is the
    /// natural order the format recommends for a byte-identical pair. The backbone count is the
    /// model's n_layers: a first-step table has exactly that many rows and no anchor, a decode table
    /// has the backbone rows plus one row per MTP stage and an anchor. The table has already checked
    /// every tensor's shape when it was captured, so the errors here are the writer's own.
    ///
    /// # Errors
    ///
    /// [std::io::Error] from either writer, and an invalid-data error if the table cannot be a valid
    /// v1 step: a zero backbone count, non-contiguous rows, a first-step table whose row count is not
    /// the backbone count, a table with an anchor but no stage rows, or a row whose tensor order or
    /// count disagrees with TENSOR_ORDER.
    pub fn write_v1(
        &self,
        backbone_layers: usize,
        manifest: &mut dyn Write,
        blob: &mut dyn Write,
    ) -> std::io::Result<()> {
        if backbone_layers == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "a v1 header needs a non-zero backbone layer count",
            ));
        }
        let first_step = self.mtp_anchor.is_none();
        if first_step && self.rows.len() != backbone_layers {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "a table with no MTP anchor has exactly the {backbone_layers} backbone rows, not {}",
                    self.rows.len()
                ),
            ));
        }
        if !first_step && self.rows.len() <= backbone_layers {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "a table with an MTP anchor has stage rows past the {backbone_layers} backbone rows, but it has {} rows",
                    self.rows.len()
                ),
            ));
        }
        for (index, row) in self.rows.iter().enumerate() {
            if row.layer() != index {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "row {index} is layer {}, so the table is not one contiguous step from layer 0",
                        row.layer()
                    ),
                ));
            }
        }
        writeln!(manifest, "# phase1-boundary v1")?;
        write!(
            manifest,
            "step {} {}",
            self.start_position,
            self.tokens.len()
        )?;
        for token in &self.tokens {
            write!(manifest, " {token}")?;
        }
        writeln!(manifest)?;
        write!(manifest, "prefix {}", self.consumed_prefix.len())?;
        for token in &self.consumed_prefix {
            write!(manifest, " {token}")?;
        }
        writeln!(manifest)?;
        match self.mtp_anchor {
            Some(anchor) => writeln!(manifest, "mtp-anchor {anchor}")?,
            None => writeln!(manifest, "mtp-anchor none")?,
        }
        writeln!(manifest, "backbone {backbone_layers}")?;
        let mut offset = 0u64;
        for row in &self.rows {
            writeln!(manifest, "row {}", row.layer())?;
            if row.tensors().len() != TENSOR_ORDER.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "layer {} has {} tensors, not the {} of TENSOR_ORDER",
                        row.layer(),
                        row.tensors().len(),
                        TENSOR_ORDER.len()
                    ),
                ));
            }
            for (index, name) in TENSOR_ORDER.iter().enumerate() {
                let tensor = &row.tensors()[index];
                if tensor.name() != *name {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "layer {} tensor {} is {}, not the {name} TENSOR_ORDER requires",
                            row.layer(),
                            index,
                            tensor.name()
                        ),
                    ));
                }
                let (dtype, _) = v1_format(name).expect("a TENSOR_ORDER name has a format");
                let shape: Vec<String> = tensor.shape().iter().map(usize::to_string).collect();
                let bytes = v1_bytes(tensor.data());
                writeln!(
                    manifest,
                    "tensor {} {} {} {} {offset} {}",
                    row.layer(),
                    tensor.name(),
                    dtype,
                    shape.join(","),
                    bytes.len()
                )?;
                blob.write_all(&bytes)?;
                offset += bytes.len() as u64;
            }
        }
        writeln!(manifest, "end")?;
        Ok(())
    }
}

/// Reads one version 1 manifest and its blob back into a [BoundaryTable], keeping the backbone
/// count the manifest declares.
///
/// This is the other half of [BoundaryTable::write_v1] and the surface a device diff uses: a device
/// run writes the same two files, the host reads both and diffs the tables. The backbone count is
/// returned beside the table rather than stored in it because the host engine's own captures are
/// made without a file and carry no layout; [diff_v1] is the pair-level entry point that requires
/// both halves of a comparison to declare the same backbone count before it diffs the tables.
///
/// The reader enforces the format's structural rules and refuses a manifest that breaks any of
/// them:
///
/// * the version line, and every header field exactly once (step, prefix, mtp-anchor, backbone);
/// * the step's start position is the consumed prefix's length, and the token counts match;
/// * rows are exactly layers 0, 1, 2, ... with no gap and no repeat, each with the eighteen tensors
///   of TENSOR_ORDER once each, in order;
/// * a table with no MTP anchor has exactly the backbone count of rows, and a table with an anchor
///   has stage rows past the backbone count;
/// * the dtype token each name requires, shapes whose element count matches the byte length, and
///   byte ranges inside the blob that do not overlap another tensor's;
/// * no trailing fields on any line, and an explicit missing line is a named refusal rather than a
///   shorter table, because criterion 1 fails if any row misses its bar.
///
/// # Errors
///
/// [EngineError::Shape] with the boundary v1 context when the files do not follow the format.
pub fn read_v1_with_layout(
    manifest: &str,
    blob: &[u8],
) -> Result<(BoundaryTable, usize), EngineError> {
    let mut lines = manifest.lines().enumerate();
    let (_, first) = lines
        .next()
        .ok_or_else(|| v1_bad("the manifest is empty"))?;
    if first.trim() != "# phase1-boundary v1" {
        return Err(v1_bad(format!(
            "the first line is {first:?}, not the version 1 header"
        )));
    }
    let mut start_position: Option<usize> = None;
    let mut tokens: Vec<u32> = Vec::new();
    let mut consumed_prefix: Vec<u32> = Vec::new();
    let mut prefix_seen = false;
    let mut anchor: Option<Option<u32>> = None;
    let mut backbone: Option<usize> = None;
    let mut rows: Vec<LayerBoundary> = Vec::new();
    let mut current: Option<LayerBoundary> = None;
    let mut used_ranges: Vec<(u64, u64)> = Vec::new();
    let mut ended = false;
    for (number, line) in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if ended {
            return Err(v1_bad(format!("line {} follows the end line", number + 1)));
        }
        let mut fields = line.split_whitespace();
        let keyword = fields.next().expect("a non-empty line has a keyword");
        match keyword {
            "step" => {
                if start_position.is_some() {
                    return Err(v1_bad("the step line appears twice"));
                }
                let position = v1_number(fields.next(), "start position")?;
                let count: usize = v1_number(fields.next(), "token count")?;
                tokens = fields
                    .map(|token| v1_number(Some(token), "token id"))
                    .collect::<Result<Vec<u32>, EngineError>>()?;
                if tokens.len() != count {
                    return Err(v1_bad(format!(
                        "the step line declares {count} tokens and gives {}",
                        tokens.len()
                    )));
                }
                start_position = Some(position);
            }
            "prefix" => {
                if prefix_seen {
                    return Err(v1_bad("the prefix line appears twice"));
                }
                let count: usize = v1_number(fields.next(), "prefix length")?;
                consumed_prefix = fields
                    .map(|token| v1_number(Some(token), "prefix token"))
                    .collect::<Result<Vec<u32>, EngineError>>()?;
                if consumed_prefix.len() != count {
                    return Err(v1_bad(format!(
                        "the prefix line declares {count} tokens and gives {}",
                        consumed_prefix.len()
                    )));
                }
                prefix_seen = true;
            }
            "mtp-anchor" => {
                if anchor.is_some() {
                    return Err(v1_bad("the mtp-anchor line appears twice"));
                }
                let token = fields
                    .next()
                    .ok_or_else(|| v1_bad("the mtp-anchor line has no value"))?;
                if fields.next().is_some() {
                    return Err(v1_bad("the mtp-anchor line has extra fields"));
                }
                anchor = Some(if token == "none" {
                    None
                } else {
                    Some(v1_number(Some(token), "MTP anchor")?)
                });
            }
            "backbone" => {
                if backbone.is_some() {
                    return Err(v1_bad("the backbone line appears twice"));
                }
                let count: usize = v1_number(fields.next(), "backbone layer count")?;
                if count == 0 {
                    return Err(v1_bad("the backbone count is zero"));
                }
                if fields.next().is_some() {
                    return Err(v1_bad("the backbone line has extra fields"));
                }
                backbone = Some(count);
            }
            "row" => {
                if start_position.is_none()
                    || !prefix_seen
                    || anchor.is_none()
                    || backbone.is_none()
                {
                    return Err(v1_bad(format!(
                        "line {} opens a row before the step, prefix, mtp-anchor and backbone lines",
                        number + 1
                    )));
                }
                if let Some(row) = current.take() {
                    v1_close_row(&mut rows, row)?;
                }
                let layer: usize = v1_number(fields.next(), "row layer")?;
                if fields.next().is_some() {
                    return Err(v1_bad("a row line has extra fields"));
                }
                current = Some(LayerBoundary::new(layer));
            }
            "tensor" => {
                let row = current.as_mut().ok_or_else(|| {
                    v1_bad(format!("line {} has a tensor before any row", number + 1))
                })?;
                let layer: usize = v1_number(fields.next(), "tensor layer")?;
                if layer != row.layer() {
                    return Err(v1_bad(format!(
                        "a tensor line names layer {layer} inside row {}",
                        row.layer()
                    )));
                }
                let name = fields
                    .next()
                    .ok_or_else(|| v1_bad("a tensor line has no name"))?;
                let index = row.tensors().len();
                let expected = TENSOR_ORDER.get(index).ok_or_else(|| {
                    v1_bad(format!(
                        "layer {layer} has more than {} tensors",
                        TENSOR_ORDER.len()
                    ))
                })?;
                if name != *expected {
                    return Err(v1_bad(format!(
                        "layer {layer} tensor {index} is {name:?}, not the {expected:?} TENSOR_ORDER requires"
                    )));
                }
                let (expected_dtype, width) = v1_format(expected)
                    .ok_or_else(|| v1_bad(format!("{expected:?} has no v1 format")))?;
                let dtype = fields
                    .next()
                    .ok_or_else(|| v1_bad("a tensor line has no dtype"))?;
                if dtype != expected_dtype {
                    return Err(v1_bad(format!(
                        "{name} has dtype {dtype:?}, but its format is {expected_dtype:?}"
                    )));
                }
                let shape_token = fields
                    .next()
                    .ok_or_else(|| v1_bad("a tensor line has no shape"))?;
                let shape: Vec<usize> = shape_token
                    .split(',')
                    .map(|side| v1_number(Some(side), "shape dimension"))
                    .collect::<Result<Vec<usize>, EngineError>>()?;
                let offset: u64 = v1_number(fields.next(), "byte offset")?;
                let length: usize = v1_number(fields.next(), "byte length")?;
                if fields.next().is_some() {
                    return Err(v1_bad(format!("the {name} line has extra fields")));
                }
                let elements = shape.iter().try_fold(1usize, |product, &side| {
                    product.checked_mul(side).ok_or_else(|| {
                        v1_bad(format!("{name}: the shape {shape:?} overflows usize"))
                    })
                })?;
                let expected_length = elements.checked_mul(width).ok_or_else(|| {
                    v1_bad(format!(
                        "{name}: the shape {shape:?} at width {width} overflows usize"
                    ))
                })?;
                if length != expected_length {
                    return Err(v1_bad(format!(
                        "{name}: shape {shape:?} at width {width} needs {expected_length} bytes, the line gives {length}"
                    )));
                }
                let end = offset.checked_add(length as u64).ok_or_else(|| {
                    v1_bad(format!("{name}: byte range {offset}+{length} overflows"))
                })?;
                for &(begin, other_end) in &used_ranges {
                    if offset < other_end && begin < end {
                        return Err(v1_bad(format!(
                            "{name}: byte range {offset}..{end} overlaps another tensor's range {begin}..{other_end}"
                        )));
                    }
                }
                used_ranges.push((offset, end));
                let range = usize::try_from(offset)
                    .ok()
                    .and_then(|begin| usize::try_from(end).ok().map(|end| begin..end))
                    .ok_or_else(|| {
                        v1_bad(format!(
                            "{name}: byte range {offset}..{end} does not fit this host"
                        ))
                    })?;
                let slice = blob.get(range.clone()).ok_or_else(|| {
                    v1_bad(format!(
                        "{name}: byte range {range:?} is outside the {}-byte blob",
                        blob.len()
                    ))
                })?;
                let data = if width == 4 {
                    BoundaryData::F32(
                        slice
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|bytes| f32::from_le_bytes(*bytes))
                            .collect(),
                    )
                } else {
                    BoundaryData::U8(slice.to_vec())
                };
                row.push(expected, shape, data)?;
            }
            "missing" => {
                let layer = fields.next().unwrap_or("?");
                let name = fields.next().unwrap_or("?");
                let reason = fields.collect::<Vec<&str>>().join(" ");
                return Err(v1_bad(format!(
                    "layer {layer} tensor {name} is missing: {reason}"
                )));
            }
            "end" => {
                if fields.next().is_some() {
                    return Err(v1_bad("the end line has extra fields"));
                }
                ended = true;
            }
            other => return Err(v1_bad(format!("unknown manifest keyword {other:?}"))),
        }
    }
    if !ended {
        return Err(v1_bad("the manifest has no end line"));
    }
    if let Some(row) = current.take() {
        v1_close_row(&mut rows, row)?;
    }
    let start_position = start_position.ok_or_else(|| v1_bad("the manifest has no step line"))?;
    if !prefix_seen {
        return Err(v1_bad("the manifest has no prefix line"));
    }
    let anchor = anchor.ok_or_else(|| v1_bad("the manifest has no mtp-anchor line"))?;
    let backbone = backbone.ok_or_else(|| v1_bad("the manifest has no backbone line"))?;
    if start_position != consumed_prefix.len() {
        return Err(v1_bad(format!(
            "the step starts at position {start_position} but the consumed prefix has {} tokens",
            consumed_prefix.len()
        )));
    }
    if rows.is_empty() {
        return Err(v1_bad(format!(
            "the manifest records no rows, but the backbone count is {backbone}"
        )));
    }
    if anchor.is_none() && rows.len() != backbone {
        return Err(v1_bad(format!(
            "a table with no MTP anchor has exactly {backbone} rows, not {}",
            rows.len()
        )));
    }
    if anchor.is_some() && rows.len() <= backbone {
        return Err(v1_bad(format!(
            "a table with an MTP anchor has stage rows past the {backbone} backbone rows, but it has {} rows",
            rows.len()
        )));
    }
    let mut table = BoundaryTable::new(start_position, &tokens, &consumed_prefix);
    if let Some(anchor) = anchor {
        table.set_mtp_anchor(anchor);
    }
    for row in rows {
        table.push_row(row)?;
    }
    Ok((table, backbone))
}

/// Reads one version 1 manifest and its blob back into a [BoundaryTable].
///
/// The backbone count is checked by the anchor rule but dropped here; use [read_v1_with_layout] to
/// keep it, and [diff_v1] to compare two dumps as a pair.
///
/// # Errors
///
/// [EngineError::Shape] with the boundary v1 context when the files do not follow the format.
pub fn read_v1(manifest: &str, blob: &[u8]) -> Result<BoundaryTable, EngineError> {
    read_v1_with_layout(manifest, blob).map(|(table, _)| table)
}

/// Reads two version 1 pairs and diffs them, refusing one that does not declare the same backbone
/// count.
///
/// This is the pair gate a host-versus-device comparison uses: both halves must agree on the step
/// identity that [BoundaryTable::diff] compares (start position, tokens, consumed prefix, MTP anchor)
/// and on the backbone count the manifest declares, which the table itself does not carry. A pair
/// whose backbone lines differ is refused by name rather than diffed as if the layouts matched; the
/// prefill case would be caught by the row count anyway, but a decode pair (backbone rows plus stage
/// rows) would not.
///
/// # Errors
///
/// [EngineError::Shape] with the boundary v1 context when either file does not follow the format,
/// when the two declare different backbone counts, or when [BoundaryTable::diff] refuses the two
/// tables as different steps.
pub fn diff_v1(
    host_manifest: &str,
    host_blob: &[u8],
    device_manifest: &str,
    device_blob: &[u8],
) -> Result<BoundaryDiff, EngineError> {
    let (host, host_backbone) = read_v1_with_layout(host_manifest, host_blob)?;
    let (device, device_backbone) = read_v1_with_layout(device_manifest, device_blob)?;
    if host_backbone != device_backbone {
        return Err(v1_bad(format!(
            "the two dumps declare different backbone counts: host {host_backbone}, device {device_backbone}"
        )));
    }
    host.diff(&device)
}

/// Closes one parsed row: rows must be layers 0, 1, 2, ... with no gap and no repeat, and a row
/// must hold every tensor of TENSOR_ORDER in order. A short or missing row is refused here rather
/// than read as a shorter, silently different table.
fn v1_close_row(rows: &mut Vec<LayerBoundary>, row: LayerBoundary) -> Result<(), EngineError> {
    let expected = rows.len();
    if row.layer() != expected {
        return Err(v1_bad(format!(
            "row {} is out of order: the next row after {} is row {expected}",
            row.layer(),
            expected.saturating_sub(1)
        )));
    }
    if row.tensors().len() != TENSOR_ORDER.len() {
        return Err(v1_bad(format!(
            "layer {} has {} tensors, not the {} a row must carry",
            row.layer(),
            row.tensors().len(),
            TENSOR_ORDER.len()
        )));
    }
    rows.push(row);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(start_position: usize, tokens: &[u32], consumed_prefix: &[u32]) -> BoundaryTable {
        BoundaryTable::new(start_position, tokens, consumed_prefix)
    }

    fn row_with(
        layer: usize,
        name: &'static str,
        shape: Vec<usize>,
        data: BoundaryData,
    ) -> LayerBoundary {
        let mut row = LayerBoundary::new(layer);
        row.push(name, shape, data).expect("a valid test tensor");
        row
    }

    fn empty_row(layer: usize) -> LayerBoundary {
        LayerBoundary::new(layer)
    }

    fn diff_error(left: &BoundaryTable, right: &BoundaryTable) -> String {
        left.diff(right)
            .expect_err("the tables must be refused")
            .to_string()
    }

    /// A different start position is a different step even when the chunk token and the consumed
    /// prefix are the same. The prefix is deliberately empty in both tables: if the start-position
    /// check were dropped, the prefix check would not cover for it and this test would pass with
    /// `Ok`, which is the mutation the review asked to pin.
    #[test]
    fn a_different_start_position_is_refused_by_name() {
        assert_eq!(
            diff_error(&table(0, &[7], &[]), &table(1, &[7], &[])),
            "boundary diff: the tables describe different steps: position 0 against position 1"
        );
    }

    /// A different chunk token is a different step even at the same start position and prefix.
    #[test]
    fn a_different_token_is_refused_by_name() {
        assert_eq!(
            diff_error(&table(1, &[7], &[6]), &table(1, &[8], &[6])),
            "boundary diff: the tables describe different steps: tokens [7] against tokens [8]"
        );
    }

    /// The consumed prefix is part of the step identity: a decode at position 1 over prefix [6] is
    /// not the decode at position 1 over prefix [5], because the attention reads different caches.
    #[test]
    fn a_different_consumed_prefix_is_refused_by_name() {
        assert_eq!(
            diff_error(&table(1, &[7], &[6]), &table(1, &[7], &[5])),
            "boundary diff: the tables describe different steps: consumed prefix [6] against prefix [5]"
        );
    }

    /// The MTP anchor is the sampled token the stages consume; the same header over a different
    /// anchor is still a different MTP step.
    #[test]
    fn a_different_mtp_anchor_is_refused_by_name() {
        let mut left = table(1, &[7], &[6]);
        let mut right = table(1, &[7], &[6]);
        left.set_mtp_anchor(3);
        right.set_mtp_anchor(4);
        assert_eq!(left.mtp_anchor(), Some(3));
        assert_eq!(right.mtp_anchor(), Some(4));
        assert_eq!(
            diff_error(&left, &right),
            "boundary diff: the tables describe different steps: MTP anchor Some(3) against anchor Some(4)"
        );
    }

    /// A table with no MTP row has no anchor, and that is not the same header as one with an
    /// anchor even before any row is compared.
    #[test]
    fn an_anchor_and_no_anchor_are_different_headers() {
        let mut anchored = table(1, &[7], &[6]);
        anchored.set_mtp_anchor(3);
        assert_eq!(
            diff_error(&anchored, &table(1, &[7], &[6])),
            "boundary diff: the tables describe different steps: MTP anchor Some(3) against anchor None"
        );
    }

    /// Row count is checked before any layer id or tensor, and the refusal names both counts.
    #[test]
    fn a_different_row_count_is_refused_by_name() {
        let mut one = table(0, &[7], &[]);
        one.push_row(empty_row(0)).expect("one row");
        let mut two = table(0, &[7], &[]);
        two.push_row(empty_row(0)).expect("row 0");
        two.push_row(empty_row(1)).expect("row 1");
        assert_eq!(
            diff_error(&one, &two),
            "boundary diff: 1 rows against 2 rows"
        );
    }

    /// The same number of rows with a different layer id is a different layer order, not a value
    /// difference: a device table that recorded layer 1 where the reference recorded layer 0 must
    /// be refused rather than compared tensor by tensor.
    #[test]
    fn a_different_layer_id_at_the_same_row_count_is_refused_by_name() {
        let mut left = table(0, &[7], &[]);
        left.push_row(empty_row(0)).expect("row 0");
        let mut right = table(0, &[7], &[]);
        right.push_row(empty_row(1)).expect("row 1");
        assert_eq!(
            diff_error(&left, &right),
            "boundary diff: row 0 against row 1: the rows are in a different layer order"
        );
    }

    /// A row with a different number of tensors is refused; the tensor-name and shape checks below
    /// would otherwise index past the shorter row.
    #[test]
    fn a_different_tensor_count_is_refused_by_name() {
        let mut left = table(0, &[7], &[]);
        left.push_row(empty_row(0)).expect("empty row");
        let mut right = table(0, &[7], &[]);
        right
            .push_row(row_with(
                0,
                RESIDUAL_IN,
                vec![1],
                BoundaryData::F32(vec![1.0]),
            ))
            .expect("one-tensor row");
        assert_eq!(
            diff_error(&left, &right),
            "boundary diff: layer 0: 0 tensors against 1 tensors"
        );
    }

    /// Same tensor count, different format name: the rows describe different capture points.
    #[test]
    fn a_different_tensor_name_is_refused_by_name() {
        let mut left = table(0, &[7], &[]);
        left.push_row(row_with(
            0,
            RESIDUAL_IN,
            vec![1],
            BoundaryData::F32(vec![1.0]),
        ))
        .expect("row");
        let mut right = table(0, &[7], &[]);
        right
            .push_row(row_with(0, FFN_OUT, vec![1], BoundaryData::F32(vec![1.0])))
            .expect("row");
        assert_eq!(
            diff_error(&left, &right),
            "boundary diff: layer 0: tensor residual_in [1] against tensor ffn_out [1]"
        );
    }

    /// Same tensor name, different shape: a table cannot be compared against a row of another
    /// width even when the name and format agree.
    #[test]
    fn a_different_tensor_shape_is_refused_by_name() {
        let mut left = table(0, &[7], &[]);
        left.push_row(row_with(
            0,
            RESIDUAL_IN,
            vec![1],
            BoundaryData::F32(vec![1.0]),
        ))
        .expect("row");
        let mut right = table(0, &[7], &[]);
        right
            .push_row(row_with(
                0,
                RESIDUAL_IN,
                vec![2],
                BoundaryData::F32(vec![1.0, 2.0]),
            ))
            .expect("row");
        assert_eq!(
            diff_error(&left, &right),
            "boundary diff: layer 0: tensor residual_in [1] against tensor residual_in [2]"
        );
    }

    /// The same name and shape recorded as f32 on one side and as raw code bytes on the other is a
    /// format disagreement, not a numeric difference.
    #[test]
    fn a_tensor_recorded_in_two_element_formats_is_refused_by_name() {
        let mut left = table(0, &[7], &[]);
        left.push_row(row_with(
            0,
            RESIDUAL_IN,
            vec![1],
            BoundaryData::F32(vec![1.0]),
        ))
        .expect("row");
        let mut right = table(0, &[7], &[]);
        right
            .push_row(row_with(
                0,
                RESIDUAL_IN,
                vec![1],
                BoundaryData::U8(vec![0x38]),
            ))
            .expect("row");
        assert_eq!(
            diff_error(&left, &right),
            "boundary diff: layer 0: tensor residual_in was recorded in two different element formats"
        );
    }

    /// Byte identity is bitwise, not numeric: signed zeros differ, NaN payloads compare by bits, and
    /// an f32 series never equals a code series. The derived `PartialEq` would answer `true` for
    /// `-0.0 == 0.0`, which is why the row-wiring test uses this method instead.
    #[test]
    fn bit_equality_is_bitwise_not_numeric() {
        assert!(!BoundaryData::F32(vec![0.0]).bit_eq(&BoundaryData::F32(vec![-0.0])));
        assert!(BoundaryData::F32(vec![0.0]).bit_eq(&BoundaryData::F32(vec![0.0])));
        assert!(BoundaryData::F32(vec![f32::NAN]).bit_eq(&BoundaryData::F32(vec![f32::NAN])));
        assert!(BoundaryData::U8(vec![1, 2]).bit_eq(&BoundaryData::U8(vec![1, 2])));
        assert!(!BoundaryData::U8(vec![1, 2]).bit_eq(&BoundaryData::U8(vec![1, 3])));
        assert!(!BoundaryData::F32(vec![1.0]).bit_eq(&BoundaryData::U8(vec![1])));
        assert!(!BoundaryData::F32(vec![1.0]).bit_eq(&BoundaryData::F32(vec![1.0, 2.0])));
    }

    /// `max_abs_delta` is a value delta for f32 and a unitless code distance for u8 - the two are
    /// different quantities and the row-level maximum mixes them, which the type's documentation
    /// now says explicitly. A NaN pair is infinite so a numeric bar cannot pass by looking small.
    #[test]
    fn max_abs_delta_is_a_value_delta_or_a_code_distance() {
        let (differing, delta) = compare_data(
            &BoundaryData::U8(vec![0x10, 0x20]),
            &BoundaryData::U8(vec![0x12, 0x20]),
        )
        .expect("two code series");
        assert_eq!(differing, 1);
        assert_eq!(delta, 2.0, "the code distance is 0x12 - 0x10");

        let (differing, delta) =
            compare_data(&BoundaryData::F32(vec![1.0]), &BoundaryData::F32(vec![1.5]))
                .expect("two f32 series");
        assert_eq!(differing, 1);
        assert_eq!(delta, 0.5);

        let (_, delta) = compare_data(
            &BoundaryData::F32(vec![f32::NAN]),
            &BoundaryData::F32(vec![1.0]),
        )
        .expect("two f32 series");
        assert!(delta.is_infinite(), "a NaN difference is infinite");
    }

    /// A `dim` that does not divide into 32-element quantisation blocks is refused by the tensor
    /// helper itself, before a byte is captured; the observed entry points check the same rule one
    /// level up, before a weight is read.
    #[test]
    fn push_block_input_refuses_a_dim_that_does_not_divide_32() {
        let config = ModelConfig {
            dim: 48,
            ..ModelConfig::tiny()
        };
        let mut row = LayerBoundary::new(0);
        let error = row
            .push_block_input(
                &config,
                RESIDUAL_IN,
                PRE_ATTENTION_XHAT,
                PRE_ATTENTION_SCALES,
                &[0.1; 48],
            )
            .expect_err("48 is not a multiple of 32");
        assert_eq!(
            error.to_string(),
            "boundary capture: residual_in: the quantised block input needs whole 32-element blocks, dim is 48"
        );
    }

    /// One step cannot record the same layer twice: a diff of duplicate rows would be ambiguous.
    #[test]
    fn a_duplicate_layer_row_is_refused() {
        let mut table = table(0, &[7], &[]);
        table.push_row(empty_row(0)).expect("first row");
        let error = table.push_row(empty_row(0)).expect_err("duplicate layer");
        assert_eq!(
            error.to_string(),
            "boundary capture: layer 0 was recorded twice in one step"
        );
    }

    /// One full row of eighteen tensors, so a v1 dump has the shape the forward records.
    fn full_row(layer: usize, scale: f32) -> LayerBoundary {
        full_row_shaped(layer, scale, 32, 2)
    }

    fn full_row_shaped(layer: usize, scale: f32, dim: usize, hc: usize) -> LayerBoundary {
        let positions = 1usize;
        let mut row = LayerBoundary::new(layer);
        for name in TENSOR_ORDER {
            let (shape, data) = match name {
                RESIDUAL_IN | POST_ATTENTION_HIDDEN | RESIDUAL_OUT => (
                    vec![positions, hc, dim],
                    BoundaryData::F32(vec![0.5 * scale; positions * hc * dim]),
                ),
                HC_IN_PRE | HC_ATTN_PRE | HC_ATTN_POST | HC_FFN_PRE | HC_FFN_POST => (
                    vec![positions, hc],
                    BoundaryData::F32(vec![0.25 * scale; positions * hc]),
                ),
                HC_ATTN_COMB | HC_FFN_COMB => (
                    vec![positions, hc, hc],
                    BoundaryData::F32(vec![0.125 * scale; positions * hc * hc]),
                ),
                PRE_ATTENTION_HIDDEN | PRE_FFN_HIDDEN | ATTENTION_OUT | FFN_OUT => (
                    vec![positions, dim],
                    BoundaryData::F32(vec![1.0 * scale; positions * dim]),
                ),
                PRE_ATTENTION_XHAT | PRE_FFN_XHAT => (
                    vec![positions, dim],
                    BoundaryData::U8(vec![0x38; positions * dim]),
                ),
                PRE_ATTENTION_SCALES | PRE_FFN_SCALES => (
                    vec![positions, dim / SCALE_BLOCK],
                    BoundaryData::U8(vec![0x80; positions * (dim / SCALE_BLOCK)]),
                ),
                _ => unreachable!("TENSOR_ORDER is the list above"),
            };
            row.push(name, shape, data).expect("a valid full row");
        }
        row
    }

    #[test]
    fn a_v1_dump_round_trips_to_the_same_table() {
        let mut table = table(2, &[7, 8], &[5, 6]);
        table.push_row(full_row(0, 1.0)).expect("row 0");
        table.push_row(full_row(1, 1.0)).expect("row 1");
        let mut manifest = Vec::new();
        let mut blob = Vec::new();
        table
            .write_v1(2, &mut manifest, &mut blob)
            .expect("the table writes");
        let text = String::from_utf8(manifest).expect("the manifest is text");
        assert!(text.starts_with(
            "# phase1-boundary v1
step 2 2 7 8
prefix 2 5 6
mtp-anchor none
backbone 2
"
        ));
        assert!(text.ends_with(
            "end
"
        ));
        assert!(!blob.is_empty());
        let read = read_v1(&text, &blob).expect("the dump reads back");
        assert_eq!(read.len(), 2);
        assert_eq!(read.start_position(), 2);
        assert_eq!(read.tokens(), &[7, 8]);
        assert_eq!(read.consumed_prefix(), &[5, 6]);
        assert_eq!(read.mtp_anchor(), None);
        assert!(
            read.diff(&table).expect("the same step").is_identical(),
            "the round trip must not change any tensor"
        );
        assert_eq!(read, table, "the whole table survives the round trip");
    }

    /// A decode table has the backbone rows plus one row per stage and an anchor; the reader must
    /// accept that valid shape, not only a first-step table.
    #[test]
    fn a_v1_decode_dump_round_trips_with_its_stage_rows() {
        let mut table = table(2, &[7], &[5, 6]);
        table.set_mtp_anchor(5);
        for layer in 0..3 {
            table.push_row(full_row(layer, 1.0)).expect("a decode row");
        }
        let mut manifest = Vec::new();
        let mut blob = Vec::new();
        table
            .write_v1(2, &mut manifest, &mut blob)
            .expect("the decode table writes");
        let text = String::from_utf8(manifest).expect("the manifest is text");
        assert!(text.contains(
            "mtp-anchor 5
backbone 2
"
        ));
        let read = read_v1(&text, &blob).expect("the decode dump reads back");
        assert_eq!(read.len(), 3);
        assert_eq!(read.mtp_anchor(), Some(5));
        assert_eq!(read, table);
    }

    /// The format document's released-size table is a contract, not an estimate: forty one-token
    /// rows are 13,537,920 bytes, the size the released-checkpoint dump on the node produced.
    #[test]
    fn a_released_one_token_dump_is_the_documented_thirteen_megabytes() {
        let mut table = table(0, &[0], &[]);
        for layer in 0..40 {
            table
                .push_row(full_row_shaped(layer, 1.0, 5120, 4))
                .expect("a released row");
        }
        let mut manifest = Vec::new();
        let mut blob = Vec::new();
        table.write_v1(40, &mut manifest, &mut blob).expect("write");
        assert_eq!(blob.len(), 40 * 338_448);
        assert_eq!(blob.len(), 13_537_920);
        let text = String::from_utf8(manifest).expect("the manifest is text");
        assert!(text.contains(
            "mtp-anchor none
backbone 40
"
        ));
        let read = read_v1(&text, &blob).expect("the dump reads back");
        assert_eq!(read.len(), 40);
        assert!(read.diff(&table).expect("the same step").is_identical());
    }

    /// Rewrites one line of a manifest: drops it, replaces it or appends a field to it. The refusals
    /// below are all one edit away from a valid pair, so each test names the rule it breaks.
    fn edit_line(text: &str, prefix: &str, replacement: Option<&str>) -> String {
        let mut lines: Vec<String> = Vec::new();
        for line in text.lines() {
            if line.starts_with(prefix) {
                match replacement {
                    Some(replacement) => lines.push(replacement.to_owned()),
                    None => continue,
                }
            } else {
                lines.push(line.to_owned());
            }
        }
        lines.join(
            "
",
        )
    }

    fn valid_manifest() -> (String, Vec<u8>) {
        let mut table = table(0, &[7], &[]);
        table.push_row(full_row(0, 1.0)).expect("row 0");
        let mut manifest = Vec::new();
        let mut blob = Vec::new();
        table
            .write_v1(1, &mut manifest, &mut blob)
            .expect("the table writes");
        (String::from_utf8(manifest).expect("text"), blob)
    }

    #[test]
    fn a_v1_dump_with_a_short_or_missing_tensor_is_refused() {
        let (text, blob) = valid_manifest();

        // Dropping one tensor line leaves a short row, which must be refused rather than read as a
        // table with a smaller tensor list.
        let dropped = edit_line(&text, "tensor 0 pre_ffn_scales", None);
        let error = read_v1(&dropped, &blob).expect_err("a short row is refused");
        assert!(
            error.to_string().contains("TENSOR_ORDER requires")
                || error.to_string().contains("a row must carry"),
            "{error}"
        );

        // The format has an explicit missing line for a tensor the run could not emit; a reader
        // must refuse it, because criterion 1 fails when a row misses its bar.
        let missing = edit_line(
            &text,
            "tensor 0 pre_ffn_scales",
            Some("missing 0 pre_ffn_scales the kernel did not record it"),
        );
        let error = read_v1(&missing, &blob).expect_err("a missing tensor is refused");
        assert!(error.to_string().contains("is missing"), "{error}");

        // A dtype token that disagrees with the name's format is refused by name.
        let wrong_dtype = text.replace(
            "tensor 0 pre_attention_xhat u8-e4m3",
            "tensor 0 pre_attention_xhat f32",
        );
        let error = read_v1(&wrong_dtype, &blob).expect_err("a wrong dtype is refused");
        assert!(error.to_string().contains("dtype"), "{error}");

        // A range outside the blob is refused, not silently short.
        let error = read_v1(&text, &blob[..blob.len() - 1]).expect_err("a short blob is refused");
        assert!(error.to_string().contains("outside"), "{error}");

        // A trailing field on a tensor line is refused rather than ignored.
        let extra = text.replacen(
            "tensor 0 residual_in f32 1,2,32 0 256",
            "tensor 0 residual_in f32 1,2,32 0 256 extra",
            1,
        );
        let error = read_v1(&extra, &blob).expect_err("an extra field is refused");
        assert!(error.to_string().contains("extra fields"), "{error}");
    }

    /// Two tensors may not name the same bytes: the format says ranges must not overlap, and an
    /// overlapping pair would let one tensor's record hide another's.
    #[test]
    fn a_v1_dump_with_overlapping_tensor_ranges_is_refused() {
        let (text, blob) = valid_manifest();
        let mut lines = Vec::new();
        let mut first_offset = None;
        for line in text.lines() {
            if line.starts_with("tensor 0 residual_in ") {
                let fields: Vec<&str> = line.split_whitespace().collect();
                first_offset = Some(fields[5].to_owned());
            }
            if line.starts_with("tensor 0 hc_in_pre ") {
                let fields: Vec<&str> = line.split_whitespace().collect();
                let overlap = format!(
                    "tensor 0 hc_in_pre {} {} {} {}",
                    fields[3],
                    fields[4],
                    first_offset.as_deref().expect("residual first"),
                    fields[6]
                );
                lines.push(overlap);
            } else {
                lines.push(line.to_owned());
            }
        }
        let error = read_v1(
            &lines.join(
                "
",
            ),
            &blob,
        )
        .expect_err("an overlap is refused");
        assert!(error.to_string().contains("overlaps"), "{error}");
    }

    /// A missing row is a shorter table, not a valid table: the rows must be layers 0, 1, 2, ...
    #[test]
    fn a_v1_dump_with_a_missing_row_is_refused() {
        let (text, blob) = valid_manifest();
        let rows: String = text
            .lines()
            .filter(|line| !line.starts_with("row 0") && !line.starts_with("tensor 0 "))
            .collect::<Vec<&str>>()
            .join(
                "
",
            );
        // Dropping the only row leaves no rows at all; the reader names the backbone mismatch.
        let error = read_v1(&rows, &blob).expect_err("a rowless table is refused");
        assert!(error.to_string().contains("no rows"), "{error}");
    }

    #[test]
    fn a_v1_dump_with_an_empty_or_inconsistent_header_is_refused() {
        let empty = "# phase1-boundary v1
step 0 0
prefix 0
mtp-anchor none
backbone 1
end
";
        let error = read_v1(empty, &[]).expect_err("an empty table is refused");
        assert!(error.to_string().contains("no rows"), "{error}");

        let (text, blob) = valid_manifest();
        let anchor = text.replace("mtp-anchor none", "mtp-anchor 7");
        let error = read_v1(&anchor, &blob).expect_err("an anchor with no stage rows is refused");
        assert!(
            error.to_string().contains("stage rows past the 1 backbone"),
            "{error}"
        );

        let duplicated = text.replace(
            "mtp-anchor none
",
            "mtp-anchor none
mtp-anchor 7
",
        );
        let error = read_v1(&duplicated, &blob).expect_err("a duplicate anchor is refused");
        assert!(error.to_string().contains("appears twice"), "{error}");

        let shifted = text.replace("step 0 1 7", "step 5 1 7");
        let error = read_v1(&shifted, &blob).expect_err("an inconsistent step is refused");
        assert!(error.to_string().contains("starts at position"), "{error}");

        let no_backbone = text.replace(
            "backbone 1
",
            "",
        );
        let error = read_v1(&no_backbone, &blob).expect_err("a missing backbone line is refused");
        assert!(error.to_string().contains("backbone"), "{error}");
    }

    #[test]
    fn a_v1_dump_of_another_version_is_refused() {
        let error = read_v1(
            "# phase1-boundary v2
",
            &[],
        )
        .expect_err("wrong version");
        assert!(error.to_string().contains("version 1 header"), "{error}");
    }

    /// The writer refuses a table that cannot be one v1 step, so the dump never produces a file the
    /// reader would then have to refuse.
    #[test]
    fn the_writer_refuses_a_table_that_is_not_one_contiguous_step() {
        let mut unanchored = table(0, &[7], &[]);
        unanchored.push_row(full_row(0, 1.0)).expect("row 0");
        let mut manifest = Vec::new();
        let mut blob = Vec::new();
        let error = unanchored
            .write_v1(0, &mut manifest, &mut blob)
            .expect_err("zero backbone");
        assert!(error.to_string().contains("non-zero backbone"), "{error}");

        let error = unanchored
            .write_v1(2, &mut manifest, &mut blob)
            .expect_err("row count differs from backbone");
        assert!(
            error.to_string().contains("exactly the 2 backbone rows"),
            "{error}"
        );

        let mut anchored = table(0, &[7], &[]);
        anchored.set_mtp_anchor(3);
        anchored.push_row(full_row(0, 1.0)).expect("row 0");
        let error = anchored
            .write_v1(1, &mut manifest, &mut blob)
            .expect_err("an anchor without stage rows");
        assert!(
            error.to_string().contains("stage rows past the 1 backbone"),
            "{error}"
        );
    }

    /// A row missing from the middle of a table is a different layout, not a shorter table. The gap
    /// check is what catches it; the earlier rowless test only exercises the empty-table rule.
    #[test]
    fn a_v1_dump_with_a_gap_in_the_middle_of_the_rows_is_refused() {
        let mut table = table(0, &[7], &[]);
        for layer in 0..3 {
            table.push_row(full_row(layer, 1.0)).expect("a row");
        }
        let mut manifest = Vec::new();
        let mut blob = Vec::new();
        table
            .write_v1(3, &mut manifest, &mut blob)
            .expect("the table writes");
        let text = String::from_utf8(manifest).expect("the manifest is text");
        let gapped: String = text
            .lines()
            .filter(|line| !line.starts_with("row 1") && !line.starts_with("tensor 1 "))
            .collect::<Vec<&str>>()
            .join("\n");
        let error = read_v1(&gapped, &blob).expect_err("a middle-row gap is refused");
        assert!(error.to_string().contains("out of order"), "{error}");
    }

    /// The backbone line is a layout declaration beside the step identity. Two decode halves can
    /// carry identical tables and still declare different backbone counts; the pair gate refuses
    /// that, because the same rows would then mean different layers.
    #[test]
    fn a_pair_that_declares_different_backbone_counts_is_refused() {
        let mut table = table(2, &[7], &[5, 6]);
        table.set_mtp_anchor(5);
        for layer in 0..3 {
            table.push_row(full_row(layer, 1.0)).expect("a decode row");
        }
        let mut host_manifest = Vec::new();
        let mut host_blob = Vec::new();
        table
            .write_v1(2, &mut host_manifest, &mut host_blob)
            .expect("the host pair writes");
        let mut device_manifest = Vec::new();
        let mut device_blob = Vec::new();
        table
            .write_v1(1, &mut device_manifest, &mut device_blob)
            .expect("the device pair writes");
        let identical = diff_v1(
            &String::from_utf8(host_manifest.clone()).expect("text"),
            &host_blob,
            &String::from_utf8(host_manifest.clone()).expect("text"),
            &host_blob,
        )
        .expect("the same pair diffs");
        assert!(identical.is_identical());
        let error = diff_v1(
            &String::from_utf8(host_manifest).expect("text"),
            &host_blob,
            &String::from_utf8(device_manifest).expect("text"),
            &device_blob,
        )
        .expect_err("a pair with different backbones is refused");
        assert!(
            error
                .to_string()
                .contains("different backbone counts: host 2, device 1"),
            "{error}"
        );
    }
}
