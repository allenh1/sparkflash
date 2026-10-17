//! Per-layer boundary capture: the Phase 1 boundary table the device path is diffed against.
//!
//! Spec: `docs/build/phase1-boundary-table.md` (the format, the tensor order, the byte order and the
//! comparison classes) and `docs/build/roadmap-to-serving.md` section 4.5 criteria 1 and 2 (the
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
    pub(crate) fn new(layer: usize) -> Self {
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
    fn push(
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
    pub(crate) fn push_stream(
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
    pub(crate) fn push_mixes(
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
    /// (`docs/build/phase1-boundary-table.md` section 3).
    pub(crate) fn push_comb(
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
    pub(crate) fn push_dense(
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
    pub(crate) fn push_block_input(
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
    pub(crate) fn new(start_position: usize, tokens: &[u32], consumed_prefix: &[u32]) -> Self {
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
    pub(crate) fn push_row(&mut self, row: LayerBoundary) -> Result<(), EngineError> {
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
}
