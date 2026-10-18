//! CSA2 attention, the compressor, the HSI indexer and the window ring.
//!
//! The reference pass for one attention layer: low-rank queries, the sliding-window KV with its RoPE
//! tail, the compressed KV the compressor pools out of `compress_ratio` tokens at a time, the
//! two-level HSI indexer that picks which compressed positions a query sees, and the grouped output
//! projection. It is the slice of docs/engine/01-forward-order.md sections 3.3-3.5 and 4 that
//! model.py:622-789 describes, plus the DSpark stage pass (model.py:1032-1074) that mtp.rs drives.
//! The kernels followed are kernel.py:311-389 (sparse attention) and kernel.py:98-181 (the two
//! activation quantisers).
//!
//! # The cross-layer channel
//!
//! Attention layers hand work down the stack instead of recomputing it (spec section 4): a layer in
//! `kv_source_layers` compresses the KV and owns the index keys, `candidate_source_layer`
//! publishes the HSI block mask, and the layers in between read what the source published. The
//! reference keeps one module-level singleton for this, `shared_attn` (model.py:1166-1181),
//! written before its consumers read and never reset between forwards. The entry points carry one
//! layer's state and nothing else, so the channel is a thread-local here with the same order of
//! writes and one slot each, per thread so that tests running in parallel do not share it.
//! [`reset_shared_attention`] clears it between sequences.
//!
//! A source layer keeps its own rows in [`AttnState::compressed`] and [`AttnState::indexer`] and
//! publishes the same rows to the channel; a consumer reads the channel. That is one extra copy of
//! the sequence-length caches, which the reference avoids by pointing the channel at the same
//! storage. It is the price of a channel with no borrowed lifetime and it is the only duplication
//! in this module.
//!
//! # What is deliberately not reproduced
//!
//! 1. The reference quantises a GEMM's activation before every fp8 weight (model.py:187, 197,
//!    kernel.py:98-124). This module multiplies decoded f32 weights by f32 activations, because
//!    [`crate::linalg`] exposes plain matmuls and nothing else. The round trips that change what a
//!    later step *reads* are reproduced: the window KV through fp8 (model.py:707), the compressed
//!    latent through fp4 with E4M3 scales (model.py:760), and the index keys and queries through fp4
//!    (model.py:546, 552).
//! 2. There is no bf16 arithmetic in this crate, so the reference's bf16 roundings of module outputs
//!    are not reproduced either. The clearest case is the compressor: model.py:446 keeps its weights
//!    bf16 at ratio 1 and model.py:485 casts the pooled result back, where this is f32 throughout.
//! 3. The kernels assert `N % block_size == 0` (kernel.py:108). The released sizes divide -
//!    head_dim 512 by both 32 and 16, index_head_dim 128 by 32 - but [`ModelConfig::tiny`] does not
//!    (head_dim 16, index_head_dim 8), so a row shorter than its block is quantised as one block
//!    here. That is the same max-abs, floor and round scheme over one block, which is the only value
//!    the rule can produce for such a row.
//! 4. `torch.topk` leaves ties undefined. This module breaks them towards the lower index, in the
//!    HSI block ranking and in the per-query top-k alike.
//!
//! A consumer refuses rather than guesses: a layer that reads a published cache checks that it is
//! there and long enough, and returns [`EngineError::Shape`] instead of attending a zero-filled
//! buffer, which is what the reference would read from a registered buffer it never wrote. Run the
//! source layers first, which spec section 4 already requires.

use super::attention_case;
use crate::config::ModelConfig;
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::quant;
use crate::weights::WeightSource;

use std::cell::RefCell;

/// One fp8 scale per this many elements along a window KV row (model.py:27, kernel.py:98-124).
const WINDOW_QUANT_BLOCK: usize = 32;
/// One fp4 scale per this many elements along an index key or query row (model.py:28).
const INDEX_QUANT_BLOCK: usize = 32;
/// The compressed KV pools 16 elements per scale, not 32 (model.py:759-760).
const COMPRESS_QUANT_BLOCK: usize = 16;
/// The amax floor of the fp4 quantiser that uses E4M3 scales (kernel.py:160-163).
const FP4_E4M3_AMAX_FLOOR: f32 = 6.0 / 512.0;
/// Where the sparse kernel starts its running score maximum (kernel.py:320-324), which is what makes
/// a query with nothing to attend to produce zero instead of NaN.
const SCORE_FLOOR: f32 = -1e30;
/// The marker for an index that attends nothing: the reference's -1 (model.py:420, 580).
const NO_INDEX: usize = usize::MAX;

/// What a layer with a compressor of its own hands down the stack (model.py:1166-1181).
///
/// One slot each: every field is written whole, before any consumer of the same step reads it.
#[derive(Debug, Default)]
struct SharedAttn {
    /// The compressed KV cache last published, `head_dim` values per row (model.py:748, 761).
    compressed: Vec<f32>,
    /// How many rows of `compressed` hold a value.
    compressed_rows: usize,
    /// Which layer published it: the reference's channel points at that layer's cache, so a
    /// different owner replaces the slot rather than extending it (model.py:748, 1173).
    compressed_owner: Option<usize>,
    /// The index keys last published, `index_head_dim` values per row (model.py:547-548).
    index_keys: Vec<f32>,
    /// How many rows of `index_keys` hold a value.
    index_rows: usize,
    /// Which layer published the keys, with the same replacement rule as `compressed_owner`.
    index_owner: Option<usize>,
    /// Which compressed positions each query of the last index source attends to, window offset
    /// already added, row-major per query (model.py:577-580).
    topk: Vec<usize>,
    /// Query rows in `topk`.
    topk_rows: usize,
    /// Positions per query row in `topk`.
    topk_cols: usize,
    /// Whether an index source has published `topk` at all: an empty publication is a real answer
    /// (model.py:729-730), a missing one is a wiring error.
    topk_ready: bool,
    /// The chunk `topk` was published for, so a consumer can tell a fresh publication from the
    /// previous step's.
    topk_step: Option<(usize, usize)>,
    /// The HSI block mask last published by the candidate source, one bool per compressed position,
    /// row-major per query (model.py:569-572).
    candidates: Vec<bool>,
    /// Query rows in `candidates`.
    candidate_rows: usize,
    /// Compressed positions per query row in `candidates`.
    candidate_cols: usize,
    /// The chunk `candidates` was published for.
    candidate_step: Option<(usize, usize)>,
}

thread_local! {
    /// The channel itself. `RefCell` because the writers and readers are all closures; a reentrant
    /// call would panic on the borrow, and the reference's single-slot shape has the same property.
    static SHARED: RefCell<SharedAttn> = RefCell::new(SharedAttn::default());
}

/// Runs `body` against this thread's channel.
fn with_shared<R>(body: impl FnOnce(&mut SharedAttn) -> R) -> R {
    SHARED.with(|cell| body(&mut cell.borrow_mut()))
}

/// Clears everything the layers published (model.py:1172-1176 has the four slots; the reference never
/// resets them because one process runs one sequence).
pub fn reset_shared_attention() {
    with_shared(|shared| *shared = SharedAttn::default());
}

/// The HSI block mask the last candidate source published: its query rows, its compressed positions
/// per row, and the mask itself row-major (model.py:569-572). All zero before any source has run.
#[must_use]
pub fn published_candidates() -> (usize, usize, Vec<bool>) {
    with_shared(|shared| {
        (
            shared.candidate_rows,
            shared.candidate_cols,
            shared.candidates.clone(),
        )
    })
}

/// A [EngineError::Shape] naming the item that rejected a geometry.
fn shape(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}

/// The element format an in-place round trip stores its values in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Element {
    /// FP8 E4M3, the window KV format (model.py:707).
    Fp8,
    /// FP4 E2M1, the compressed latent and index format (model.py:546, 552, 760).
    Fp4,
}

/// How a round trip picks its scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScaleRule {
    /// The smallest power of two that keeps the block inside the element range (kernel.py:76, 166).
    PowerOfTwo,
    /// `amax / 6` rounded to E4M3, the compressed-KV rule (kernel.py:162-163).
    E4M3Six,
}

/// Round-trips every `block`-long block of every `width`-long row of `values` through `element`,
/// in place.
///
/// The rule is the reference's quantiser - max-abs, floor, scale, clamp, round, scale back - in the
/// reference's order: the power-of-two scales multiply the max-abs by the reciprocal of the element
/// maximum (quant.rs's block_scale_code does the same), the E4M3 scale divides by 6 and is the E4M3
/// value it rounds to, and the value is divided by the decoded scale, clamped and rounded
/// (kernel.py:41-95, 160-172). A row shorter than `block` is one block: see the module notes on
/// kernel.py:108.
fn round_trip_rows(
    values: &mut [f32],
    width: usize,
    block: usize,
    element: Element,
    rule: ScaleRule,
) {
    if width == 0 || block == 0 {
        return;
    }
    for row in values.chunks_mut(width) {
        for chunk in row.chunks_mut(block) {
            let amax = chunk.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
            let scale = match (element, rule) {
                (Element::Fp8, ScaleRule::PowerOfTwo) => {
                    let amax = amax.max(quant::ACTIVATION_AMAX_FLOOR);
                    quant::f8e8m0_to_f32(quant::e8m0_encode_ceil(
                        amax * (1.0 / quant::FP8_E4M3_MAX),
                    ))
                }
                (Element::Fp4, ScaleRule::PowerOfTwo) => {
                    let amax = amax.max(quant::FP4_AMAX_FLOOR);
                    quant::f8e8m0_to_f32(quant::e8m0_encode_ceil(
                        amax * (1.0 / quant::FP4_E2M1_MAX),
                    ))
                }
                (Element::Fp4, ScaleRule::E4M3Six) => {
                    let amax = amax.max(FP4_E4M3_AMAX_FLOOR);
                    quant::f8e4m3_to_f32(quant::e4m3_encode(amax / quant::FP4_E2M1_MAX))
                }
                // No fp8 tensor uses an E4M3 scale: model.py:760 is the only such rule and it is fp4.
                (Element::Fp8, ScaleRule::E4M3Six) => amax,
            };
            for value in chunk.iter_mut() {
                let scaled = *value / scale;
                *value = match element {
                    Element::Fp8 => {
                        let code = quant::e4m3_encode(
                            scaled.clamp(-quant::FP8_E4M3_MAX, quant::FP8_E4M3_MAX),
                        );
                        quant::f8e4m3_to_f32(code) * scale
                    }
                    Element::Fp4 => {
                        let nibble = quant::f4e2m1_encode(
                            scaled.clamp(-quant::FP4_E2M1_MAX, quant::FP4_E2M1_MAX),
                        );
                        quant::f4e2m1_to_f32(nibble) * scale
                    }
                };
            }
        }
    }
}

/// The RoPE tables for one position each: `rope_head_dim / 2` cosines and the same sines.
pub struct RopeTables {
    /// The cosines, `positions * rope_head_dim / 2` long.
    pub cos: Vec<f32>,
    /// The sines, the same length.
    pub sin: Vec<f32>,
}

/// Builds the tables for `positions` with the base geometry: `rope_theta` and no YaRN correction
/// (model.py:368-389 as model.py:639-640 selects it for a layer with no compression).
///
/// This is the layer-independent table. A layer whose `compress_ratio` is positive rotates with
/// `compress_rope_theta` and the YaRN correction instead, which needs the layer id and is what
/// [`rope_tables_for_layer`] builds.
#[must_use]
pub fn rope_tables(config: &ModelConfig, positions: &[usize]) -> RopeTables {
    tables_for(
        config,
        config.rope_theta,
        0,
        config.rope_head_dim,
        positions,
    )
}

/// Builds the tables for `positions` as `layer` rotates them (model.py:626-698).
///
/// A layer with a positive `compress_ratio` takes `compress_rope_theta` and the YaRN correction
/// that `original_seq_len` turns on (model.py:639-640); every other layer takes `rope_theta` with
/// no correction.
#[must_use]
pub fn rope_tables_for_layer(
    config: &ModelConfig,
    layer: usize,
    positions: &[usize],
) -> RopeTables {
    if layer_ratio(config, layer) > 0 {
        tables_for(
            config,
            config.compress_rope_theta,
            config.original_seq_len,
            config.rope_head_dim,
            positions,
        )
    } else {
        rope_tables(config, positions)
    }
}

/// The body of the two builders: the reference's `precompute_freqs_cis` (model.py:368-389) reduced
/// to the rows `positions` asks for.
///
/// The frequencies stay f32 because the reference's are an f32 tensor, while the YaRN band edges are
/// computed in f64 and then narrowed, because the reference computes them with Python floats and
/// only narrows them where they meet that tensor.
fn tables_for(
    config: &ModelConfig,
    base: f64,
    original_seq_len: usize,
    dim: usize,
    positions: &[usize],
) -> RopeTables {
    let pairs = dim / 2;
    let base_f32 = base as f32;
    let mut freqs: Vec<f32> = (0..pairs)
        .map(|k| 1.0 / base_f32.powf(2.0 * k as f32 / dim as f32))
        .collect();
    if original_seq_len > 0 && pairs > 0 {
        // YaRN: the dimensions whose wavelength already completes beta_fast rotations inside the
        // original context keep their frequency, those beyond beta_slow rotations are divided by the
        // factor, and the band between fades linearly (model.py:377-386).
        let corrected = |rotations: f64| {
            dim as f64 * (original_seq_len as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corrected(config.beta_fast).floor().max(0.0);
        let high = corrected(config.beta_slow).ceil().min(dim as f64 - 1.0);
        let span = (high - low).max(1e-3);
        let factor = config.rope_factor as f32;
        for (k, freq) in freqs.iter_mut().enumerate() {
            let ramp = (((k as f64 - low) / span).clamp(0.0, 1.0)) as f32;
            let smooth = 1.0 - ramp;
            *freq = *freq / factor * ramp + *freq * smooth;
        }
    }
    let mut cos = Vec::with_capacity(positions.len() * pairs);
    let mut sin = Vec::with_capacity(positions.len() * pairs);
    for &position in positions {
        for &freq in &freqs {
            let angle = position as f32 * freq;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    RopeTables { cos, sin }
}

/// Rotates the last `rope` values of each of `rows` rows of `width` values, using the per-row
/// tables (model.py:392-406; the reference rotates the last `rope_head_dim` of a head, model.py:772).
///
/// # Errors
///
/// [EngineError::Shape] when the slice is not `rows` whole rows, or the tables are too short.
fn rotate_rows(
    values: &mut [f32],
    rows: usize,
    width: usize,
    rope: usize,
    tables: &RopeTables,
    inverse: bool,
) -> Result<(), EngineError> {
    if rope == 0 {
        return Ok(());
    }
    if width < rope || !rope.is_multiple_of(2) {
        return Err(shape(
            "rope rotation",
            format!("a {rope}-long tail does not fit a {width}-wide row"),
        ));
    }
    if width == 0 || values.len() < rows * width {
        return Err(shape(
            "rope rotation",
            format!(
                "{rows} rows of {width} need {} values, got {}",
                rows * width,
                values.len()
            ),
        ));
    }
    let pairs = rope / 2;
    if tables.cos.len() < rows * pairs || tables.sin.len() < rows * pairs {
        return Err(shape(
            "rope rotation",
            format!(
                "{rows} rows need {} table entries, got {} cosines",
                rows * pairs,
                tables.cos.len()
            ),
        ));
    }
    for row in 0..rows {
        let start = row * width + width - rope;
        let slice = &mut values[start..start + rope];
        let cos = &tables.cos[row * pairs..(row + 1) * pairs];
        let sin = &tables.sin[row * pairs..(row + 1) * pairs];
        if inverse {
            linalg::rope_rotate_inverse(slice, cos, sin)?;
        } else {
            linalg::rope_rotate(slice, cos, sin)?;
        }
    }
    Ok(())
}

/// The positions of a run of head rows: every head of a token rotates at that token's own position,
/// because the reference applies one table row to all of a token's heads (model.py:392-406, 772).
fn head_row_positions(start_pos: usize, tokens: usize, heads: usize) -> Vec<usize> {
    (start_pos..start_pos + tokens)
        .flat_map(|position| std::iter::repeat_n(position, heads))
        .collect()
}

/// The dot product of two equally long slices, summed in element order.
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// The position a slot holds when it holds nothing.
const EMPTY_SLOT: usize = usize::MAX;

/// One layer's window ring over the tokens it has seen.
///
/// `window_size` rows of `head_dim` values: position `p` lives in slot `p % window_size`
/// (model.py:709-718, 1044-1051). A row is either the one written for that position or gone once
/// the ring wrapped past it, which is what [`WindowRing::row`] answers.
pub struct WindowRing {
    /// The slot rows, `window_size * head_dim` values.
    rows: Vec<f32>,
    /// The position each slot holds, or [`EMPTY_SLOT`].
    positions: Vec<usize>,
    /// How many slots currently hold a row.
    filled: usize,
    /// The greatest position written, or none.
    last: Option<usize>,
    /// The width of one row.
    head_dim: usize,
    /// The number of slots.
    window: usize,
}

impl WindowRing {
    /// A ring of `config.window_size` rows of `config.head_dim` values.
    #[must_use]
    pub fn for_config(config: &ModelConfig) -> Self {
        Self {
            rows: vec![0.0; config.window_size * config.head_dim],
            positions: vec![EMPTY_SLOT; config.window_size],
            filled: 0,
            last: None,
            head_dim: config.head_dim,
            window: config.window_size,
        }
    }

    /// Writes `row` at the slot `position` maps to, overwriting whatever that slot held.
    ///
    /// Writing the same position again is a no-op, so a caller that replays a step keeps the ring
    /// consistent.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the row is not one head vector, or the ring has no slots.
    pub fn write(&mut self, position: usize, row: &[f32]) -> Result<(), EngineError> {
        if self.window == 0 || row.len() != self.head_dim {
            return Err(shape(
                "window ring write",
                format!(
                    "a {} value row does not fit a {}-slot ring of width {}",
                    row.len(),
                    self.window,
                    self.head_dim
                ),
            ));
        }
        let slot = position % self.window;
        if self.positions[slot] != position {
            if self.positions[slot] == EMPTY_SLOT && self.filled < self.window {
                self.filled += 1;
            }
            self.positions[slot] = position;
            self.rows[slot * self.head_dim..(slot + 1) * self.head_dim].copy_from_slice(row);
        }
        self.last = Some(self.last.map_or(position, |held| held.max(position)));
        Ok(())
    }

    /// The row held for `position`, or `None` when it was evicted or never written.
    #[must_use]
    pub fn row(&self, position: usize) -> Option<&[f32]> {
        if self.window == 0 {
            return None;
        }
        let slot = position % self.window;
        (self.positions[slot] == position)
            .then(|| &self.rows[slot * self.head_dim..(slot + 1) * self.head_dim])
    }

    /// How many rows the ring holds, at most `window_size`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filled
    }

    /// Whether the ring holds no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// The greatest position written, or `None` while the ring is empty.
    #[must_use]
    pub fn last_position(&self) -> Option<usize> {
        self.last
    }

    /// The width of one row.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The number of slots.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.window
    }

    /// The row in `slot`, which is zero while the slot holds nothing.
    fn slot_row(&self, slot: usize) -> &[f32] {
        &self.rows[slot * self.head_dim..(slot + 1) * self.head_dim]
    }

    /// Empties the ring in place: every slot a zero row that holds nothing.
    fn clear(&mut self) {
        self.rows.fill(0.0);
        self.positions.fill(EMPTY_SLOT);
        self.filled = 0;
        self.last = None;
    }
}

/// The attention state one layer owns: its window ring, its compressed KV and the indexer's keys.
///
/// The compressor's pool width is the layer's `compress_ratio`, and this type does not know which
/// layer it belongs to, so those buffers are sized on the first call that needs them. The reference
/// instead preallocates every one of them for the whole model (model.py:452-453, 664-679); the
/// caches here start empty and grow with the sequence, so a test config needs no `max_seq_len`.
pub struct AttnState {
    /// The window ring this layer writes.
    pub window: WindowRing,
    /// The compressed KV rows this layer published, `head_dim` values per row
    /// (model.py:748-761). A layer that does not own a compressor leaves this empty and reads the
    /// channel.
    pub compressed: Vec<f32>,
    /// How many compressed rows this layer attends to: `(start_pos + seqlen) / compress_ratio`
    /// (model.py:744). It counts the published rows for a layer that reads the channel too.
    pub compressed_len: usize,
    /// The index keys this layer owns, `index_head_dim` values per row (model.py:520-525, 547).
    /// Empty for a layer whose keys come from the channel.
    pub indexer: Vec<f32>,
    /// The chunk the last call on this state saw: its first position and its row count. The fixed
    /// entry-point signatures carry neither, and [`compressed_indices`] needs both.
    step: Option<(usize, usize)>,
    /// The window KV length the last call used: the offset the compressed ids are shifted by
    /// (model.py:580, where it is `kv.size(1)` after the window path). `None` before the first
    /// call, which makes a standalone [`compressed_indices`] refuse rather than guess.
    window_rows: Option<usize>,
    /// The compressor's latent for this step, before RoPE, `head_dim` values per row
    /// (model.py:477, 744-747).
    latent: Vec<f32>,
    /// How many rows of `latent` hold a value.
    latent_rows: usize,
    /// `weights_proj(x)` scaled by the indexer's own factor, `index_n_heads` values per query
    /// (model.py:555). [`compressed_indices`] carries no `x`, so the caller prepares it here.
    index_weights: Vec<f32>,
    /// The compressor's partial group, `compress_ratio` rows of `head_dim` (model.py:450-456).
    kv_state: Vec<f32>,
    /// The gate scores of those rows, `-inf` in the slots that hold nothing so the softmax ignores
    /// them (model.py:454-455).
    score_state: Vec<f32>,
}

impl AttnState {
    /// Fresh state for one layer or MTP stage of `config`.
    #[must_use]
    pub fn for_config(config: &ModelConfig) -> Self {
        Self {
            window: WindowRing::for_config(config),
            compressed: Vec::new(),
            compressed_len: 0,
            indexer: Vec::new(),
            step: None,
            window_rows: None,
            latent: Vec::new(),
            latent_rows: 0,
            index_weights: Vec::new(),
            kv_state: Vec::new(),
            score_state: Vec::new(),
        }
    }

    /// Drops everything this layer holds: the ring, both caches and the compressor's partial group.
    ///
    /// The reference has no equivalent because one process runs one sequence (model.py:1179-1180);
    /// this exists so a caller can start a second sequence with the same states.
    pub fn reset(&mut self) {
        self.window.clear();
        self.compressed.clear();
        self.compressed_len = 0;
        self.indexer.clear();
        self.step = None;
        self.window_rows = None;
        self.latent.clear();
        self.latent_rows = 0;
        self.index_weights.clear();
        self.kv_state.clear();
        self.score_state.clear();
    }
}

/// The layer's compression ratio, or 0 when the config names none.
///
/// The reference indexes `compress_ratios` directly (model.py:636) and would raise on a short list;
/// a test config that is short of entries gets no compression here instead.
fn layer_ratio(config: &ModelConfig, layer: usize) -> usize {
    config.compress_ratio(layer).unwrap_or(0)
}

/// Whether the layer is a backbone block rather than an MTP stage (model.py:653).
fn is_backbone(config: &ModelConfig, layer: usize) -> bool {
    !config.is_mtp(layer)
}

/// Whether the layer compresses its own KV (model.py:654, 658-659).
fn is_kv_source(config: &ModelConfig, layer: usize) -> bool {
    is_backbone(config, layer) && config.kv_source_layers.contains(&layer)
}

/// Whether the layer runs its own indexer (model.py:655, 660-661).
fn is_index_source(config: &ModelConfig, layer: usize) -> bool {
    is_backbone(config, layer) && config.index_source_layers.contains(&layer)
}

/// Whether the layer owns the index keys the indexer reads (model.py:500).
fn owns_index_keys(config: &ModelConfig, layer: usize) -> bool {
    config.kv_source_layers.contains(&layer)
}

/// How many whole rows of `width` a slice holds.
///
/// # Errors
///
/// [EngineError::Shape] when the length is not a whole number of rows.
fn rows_of(values: &[f32], width: usize, what: &str) -> Result<usize, EngineError> {
    if width == 0 || !values.len().is_multiple_of(width) {
        return Err(shape(
            what,
            format!("{} values are not whole rows of {width}", values.len()),
        ));
    }
    Ok(values.len() / width)
}

/// A weight source with the element limit the caller of this pass gave.
///
/// Every tensor this file materialises is decoded under `limit`, and [linalg::REFERENCE_TENSOR_LIMIT]
/// is deliberately not used here: it is the default a caller applies when it names no limit, not a
/// limit of the reference model, and every tensor in this file is sized by the caller's configuration
/// rather than by a size the reference fixes.
#[derive(Clone, Copy)]
struct Reads<'a> {
    /// The source the tensors come from.
    source: &'a dyn WeightSource,
    /// The caller's element limit for any single tensor.
    limit: usize,
}

/// Reads one tensor by name, decoded to f32, under the caller's element limit.
///
/// # Errors
///
/// [EngineError::Missing] when the source has no such tensor, [`EngineError::TooLarge`] when it is
/// over `src.limit` elements, and whatever the decoder reports.
fn weight(src: Reads<'_>, name: &str) -> Result<Vec<f32>, EngineError> {
    linalg::decode_by_name(src.source, name, src.limit)
}

/// Reads a block-scaled fp8 Linear weight and multiplies in its E8M0 grid (model.py:229-232).
///
/// # Errors
///
/// As [weight], plus [EngineError::Shape] when the weight or grid does not match `rows` by `cols`.
fn dense_weight(
    src: Reads<'_>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, EngineError> {
    let mut values = weight(src, &format!("{name}.weight"))?;
    let scales = weight(src, &format!("{name}.scale"))?;
    linalg::apply_scales(&mut values, rows, cols, &scales, ScaleGrid::Block32)?;
    Ok(values)
}

/// Reads a weight whose scale grid may be absent: `wo_a` is fp8 in the file and bf16 after
/// convert.py (model.py:784-787, tensor.rs's note on its grid).
///
/// # Errors
///
/// As [weight], plus [EngineError::Shape] when a grid that is present does not match.
fn scaled_or_plain_weight(
    src: Reads<'_>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, EngineError> {
    let mut values = weight(src, &format!("{name}.weight"))?;
    let scale_name = format!("{name}.scale");
    if src.source.spec(&scale_name).is_ok() {
        let scales = weight(src, &scale_name)?;
        linalg::apply_scales(&mut values, rows, cols, &scales, ScaleGrid::Block32)?;
    }
    Ok(values)
}

/// The sparse attention kernel for one query: gathers `indices` out of `kv` and weights them
/// (kernel.py:311-389).
///
/// `q` is one position's queries, `heads * head_dim` values; `kv` is whole rows of `head_dim`;
/// `sink` is one value per head. An index outside `kv` - [`NO_INDEX`], the reference's -1 - is a
/// zero row with a masked score. The score is `q . kv * scale`, the running maximum starts at
/// [`SCORE_FLOOR`] so a query with nothing to attend to comes out zero, and the attention sink is
/// added to the denominator only (kernel.py:352-353): a large sink shrinks an output rather than
/// adding a value to it.
///
/// The exponentials are the f32 `exp` of a score difference. The kernel evaluates
/// `exp2((score - max) * log2e)` (kernel.py:345), which is the same function up to the last bits.
///
/// # Errors
///
/// [EngineError::Shape] when the arguments do not describe whole heads, whole rows, or a long enough
/// output.
#[allow(clippy::too_many_arguments)] // the kernel's own signature (kernel.py:311): the query, the gathered KV, the sink, the indices, the head geometry, the scale and the output buffer are all distinct inputs.
pub fn sparse_attention(
    q: &[f32],
    kv: &[f32],
    sink: &[f32],
    indices: &[usize],
    heads: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), EngineError> {
    if head_dim == 0 || !kv.len().is_multiple_of(head_dim) {
        return Err(shape(
            "sparse attention kv",
            format!("{} values are not whole rows of {head_dim}", kv.len()),
        ));
    }
    if q.len() < heads * head_dim || sink.len() < heads || out.len() < heads * head_dim {
        return Err(shape(
            "sparse attention",
            format!(
                "{heads} heads of {head_dim} need {} query values, {heads} sinks and {} out values; got {}, {} and {}",
                heads * head_dim,
                heads * head_dim,
                q.len(),
                sink.len(),
                out.len()
            ),
        ));
    }
    let rows = kv.len() / head_dim;
    for head in 0..heads {
        let query = &q[head * head_dim..(head + 1) * head_dim];
        let output = &mut out[head * head_dim..(head + 1) * head_dim];
        let mut scores_max = SCORE_FLOOR;
        for &index in indices {
            if index < rows {
                let row = &kv[index * head_dim..(index + 1) * head_dim];
                scores_max = scores_max.max(dot(query, row) * scale);
            }
        }
        output.fill(0.0);
        let mut denominator = (sink[head] - scores_max).exp();
        for &index in indices {
            if index >= rows {
                continue;
            }
            let row = &kv[index * head_dim..(index + 1) * head_dim];
            let weight = (dot(query, row) * scale - scores_max).exp();
            for (slot, value) in output.iter_mut().zip(row.iter()) {
                *slot += weight * value;
            }
            denominator += weight;
        }
        // A row with nothing to attend to leaves the accumulator at zero, which is what
        // kernel.py:320-324's -1e30 floor is for. A denominator of zero can only come from a sink of
        // -inf, which no checkpoint holds, and dividing by it would be the reference's NaN.
        if denominator > 0.0 {
            for value in output.iter_mut() {
                *value /= denominator;
            }
        }
    }
    Ok(())
}

/// Which window slots each query of a chunk attends to (model.py:409-426).
///
/// One row per query. During prefill (`start_pos == 0`) a row is chunk-relative: query `i` attends
/// to rows `max(0, i + 1 - window_size) ..= i` of this chunk, and [`NO_INDEX`] stands for the
/// reference's -1 where a row is before the sequence started. Outside prefill a row is the ring's
/// slots in the order the reference lists them, oldest first, with [`NO_INDEX`] for the slots the
/// sequence has not reached; the row count is `window_size`, and the caller's ring must hold the
/// positions this formula assumes, which it does when rows are written in order.
#[must_use]
pub fn window_indices(config: &ModelConfig, start_pos: usize, seqlen: usize) -> Vec<usize> {
    let window = config.window_size;
    if window == 0 || seqlen == 0 {
        return Vec::new();
    }
    if start_pos == 0 {
        let width = seqlen.min(window);
        let mut ids = vec![NO_INDEX; seqlen * width];
        for query in 0..seqlen {
            let first = (query + 1).saturating_sub(window);
            for slot in 0..width {
                let row = first + slot;
                ids[query * width + slot] = if row > query { NO_INDEX } else { row };
            }
        }
        ids
    } else {
        let mut ids = vec![NO_INDEX; seqlen * window];
        for query in 0..seqlen {
            let position = start_pos + query;
            let filled = window.min(position + 1);
            let oldest = position % window + 1;
            for slot in 0..window {
                ids[query * window + slot] = if slot < window - filled {
                    NO_INDEX
                } else {
                    (oldest + slot) % window
                };
            }
        }
        ids
    }
}

/// Pools `count` rows into one by weighting each element's rows by the softmax of its gate scores
/// (model.py:475, 482).
///
/// The softmax runs per element of the latent, not per row: the reference's score tensor is
/// `[..., ratio, head_dim]` and reduces over the ratio axis of that shape (model.py:475). An
/// all-`-inf` slice - a group with empty slots - contributes zero.
fn pool_group(
    dest: &mut [f32],
    kv: &[f32],
    score: &[f32],
    base: usize,
    count: usize,
    head_dim: usize,
    weights: &mut [f32],
) {
    for element in 0..head_dim {
        let mut max = f32::NEG_INFINITY;
        for row in 0..count {
            max = max.max(score[(base + row) * head_dim + element]);
        }
        if !max.is_finite() {
            dest[element] = 0.0;
            continue;
        }
        let mut total = 0.0f32;
        for row in 0..count {
            let weight = (score[(base + row) * head_dim + element] - max).exp();
            weights[row] = weight;
            total += weight;
        }
        let mut acc = 0.0f32;
        for row in 0..count {
            let weight = if total > 0.0 {
                weights[row] / total
            } else {
                0.0
            };
            acc += weight * kv[(base + row) * head_dim + element];
        }
        dest[element] = acc;
    }
}

/// Pools `ratio` consecutive pre-RoPE KV rows into one latent with a learned softmax gate
/// (model.py:429-486).
///
/// Fills [`AttnState::latent`] with the rows this step produced, which is none while a group is
/// still filling (model.py:483-484). The latent is pre-RoPE deliberately (model.py:434): the indexer
/// needs the unrotated form, so the caller rotates it afterwards.
///
/// Every weight here is a plain matmul: the reference's compressor weights are bf16 at ratio 1 and
/// f32 above it (model.py:446-448), and this crate has neither dtype.
///
/// # Errors
///
/// [EngineError] from any weight read or shape mismatch, including an input that is not whole rows.
fn compressor_step(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: Reads<'_>,
) -> Result<(), EngineError> {
    state.latent.clear();
    state.latent_rows = 0;
    let ratio = layer_ratio(config, layer);
    if ratio == 0 || !is_kv_source(config, layer) {
        return Ok(());
    }
    let dim = config.dim;
    let head_dim = config.head_dim;
    let seqlen = rows_of(x, dim, "compressor input")?;
    let end_pos = start_pos + seqlen;
    let norm = weight(src, &format!("{prefix}.attn.compressor.norm.weight"))?;
    if ratio == 1 {
        // model.py:461-462: one token per group, so a plain projection with no gate and no pooling.
        let wkv = weight(src, &format!("{prefix}.attn.compressor.wkv.weight"))?;
        let mut latent = vec![0.0; seqlen * head_dim];
        linalg::matmul(&mut latent, x, &wkv, head_dim, dim, seqlen)?;
        linalg::rms_norm(&mut latent, &norm, head_dim, config.norm_eps as f32)?;
        state.latent = latent;
        state.latent_rows = seqlen;
        return Ok(());
    }
    // Above ratio 1 the reference keeps a partial group between steps; its buffers are the layer's
    // ratio, so they are sized on first use here.
    if state.kv_state.len() != ratio * head_dim {
        state.kv_state = vec![0.0; ratio * head_dim];
        state.score_state = vec![f32::NEG_INFINITY; ratio * head_dim];
    }
    let wkv = weight(src, &format!("{prefix}.attn.compressor.wkv.weight"))?;
    let wgate = weight(src, &format!("{prefix}.attn.compressor.wgate.weight"))?;
    let mut kv = vec![0.0; seqlen * head_dim];
    linalg::matmul(&mut kv, x, &wkv, head_dim, dim, seqlen)?;
    let mut score = vec![0.0; seqlen * head_dim];
    linalg::matmul(&mut score, x, &wgate, head_dim, dim, seqlen)?;
    // One scratch weight per pool slot, reused by every group in this call.
    let mut weights = vec![0.0f32; ratio];
    let rows = end_pos / ratio - start_pos / ratio;
    let mut latent = vec![0.0; rows * head_dim];
    if start_pos == 0 {
        // model.py:466-475: whole groups are pooled here and the trailing partial group waits in the
        // state for a later step.
        let remainder = seqlen % ratio;
        let groups = (seqlen - remainder) / ratio;
        for group in 0..groups {
            let dest = &mut latent[group * head_dim..(group + 1) * head_dim];
            pool_group(
                dest,
                &kv,
                &score,
                group * ratio,
                ratio,
                head_dim,
                &mut weights,
            );
        }
        for slot in 0..remainder {
            let row = seqlen - remainder + slot;
            state.kv_state[slot * head_dim..(slot + 1) * head_dim]
                .copy_from_slice(&kv[row * head_dim..(row + 1) * head_dim]);
            state.score_state[slot * head_dim..(slot + 1) * head_dim]
                .copy_from_slice(&score[row * head_dim..(row + 1) * head_dim]);
        }
    } else {
        // model.py:476-482: one token per step fills a slot, and the group is pooled only when it
        // completes. A chunk longer than one token is that rule applied in order, which also keeps
        // the pooling in step with the slots each group actually used.
        let mut written = 0usize;
        for row in 0..seqlen {
            let position = start_pos + row;
            let slot = position % ratio;
            state.kv_state[slot * head_dim..(slot + 1) * head_dim]
                .copy_from_slice(&kv[row * head_dim..(row + 1) * head_dim]);
            state.score_state[slot * head_dim..(slot + 1) * head_dim]
                .copy_from_slice(&score[row * head_dim..(row + 1) * head_dim]);
            if (position + 1).is_multiple_of(ratio) {
                let dest = &mut latent[written * head_dim..(written + 1) * head_dim];
                pool_group(
                    dest,
                    &state.kv_state,
                    &state.score_state,
                    0,
                    ratio,
                    head_dim,
                    &mut weights,
                );
                written += 1;
            }
        }
    }
    linalg::rms_norm(&mut latent, &norm, head_dim, config.norm_eps as f32)?;
    state.latent = latent;
    state.latent_rows = rows;
    Ok(())
}

/// The geometry and inputs one indexer call needs, so that the helper taking them stays inside the
/// argument count every other function in this module respects.
struct IndexRequest<'a> {
    /// The state: this layer's index-key cache and the per-query weights attention_forward left.
    state: &'a AttnState,
    /// The model geometry.
    config: &'a ModelConfig,
    /// The layer running its indexer.
    layer: usize,
    /// The query latent, the output of `wq_a` and `q_norm` (model.py:527, 550).
    q: &'a [f32],
    /// The weight source.
    src: Reads<'a>,
    /// The layer prefix, so the indexer's own tensors are `{prefix}.attn.indexer.*`.
    prefix: &'a str,
    /// The index keys: this layer's cache, or the channel's rows when it does not own them
    /// (model.py:537-548, 554).
    keys: &'a [f32],
    /// The HSI candidate mask, when this layer reads one (model.py:573-575).
    candidates: Option<&'a [bool]>,
    /// The first position of the chunk.
    start_pos: usize,
    /// Its row count.
    seqlen: usize,
    /// The layer's compression ratio.
    ratio: usize,
    /// How many compressed positions the chunk has: `(start_pos + seqlen) / ratio` (model.py:744).
    n_pos: usize,
}

/// One index source's selection (model.py:550-580).
///
/// Writes the per-query positions into `out`, in the shared index space, with [`NO_INDEX`] where a
/// query cannot reach a chosen position, and returns the HSI block mask when this layer is the
/// candidate source so that the caller can publish it.
///
/// # Errors
///
/// [EngineError] from any weight read, plus [EngineError::Shape] when a length does not match the
/// geometry.
fn index_source_indices(
    request: &IndexRequest<'_>,
    out: &mut Vec<usize>,
) -> Result<Option<Vec<bool>>, EngineError> {
    let config = request.config;
    let head_dim = config.index_head_dim;
    let heads = config.index_n_heads;
    let seqlen = request.seqlen;
    let n_pos = request.n_pos;
    // The query projection, rotated and fp4 round-tripped exactly like the keys (model.py:550-552).
    let mut q_index = vec![0.0; seqlen * heads * head_dim];
    let wq_b = dense_weight(
        request.src,
        &format!("{}.attn.indexer.wq_b", request.prefix),
        heads * head_dim,
        config.q_lora_rank,
    )?;
    linalg::matmul(
        &mut q_index,
        request.q,
        &wq_b,
        heads * head_dim,
        config.q_lora_rank,
        seqlen,
    )?;
    let tables = rope_tables_for_layer(
        config,
        request.layer,
        &head_row_positions(request.start_pos, seqlen, heads),
    );
    rotate_rows(
        &mut q_index,
        seqlen * heads,
        head_dim,
        config.rope_head_dim,
        &tables,
        false,
    )?;
    round_trip_rows(
        &mut q_index,
        head_dim,
        INDEX_QUANT_BLOCK,
        Element::Fp4,
        ScaleRule::PowerOfTwo,
    );
    // The per-query weights attention_forward prepared (model.py:555). The signature of
    // compressed_indices carries no x, so they arrive through the state.
    if request.state.index_weights.len() < seqlen * heads {
        return Err(shape(
            "compressed indices",
            format!(
                "{} query weights for {seqlen} queries of {heads} index heads",
                request.state.index_weights.len()
            ),
        ));
    }
    let keys_needed = n_pos * head_dim;
    if request.keys.len() < keys_needed {
        return Err(shape(
            "compressed indices",
            format!(
                "the index keys hold {} values; {n_pos} positions of {head_dim} need {keys_needed}",
                request.keys.len()
            ),
        ));
    }
    // The score: rectified query-key dot products combined by the query's own weights, summed over
    // the index heads in order (model.py:556-557).
    let mut score = vec![0.0f32; seqlen * n_pos];
    for query in 0..seqlen {
        let weights = &request.state.index_weights[query * heads..(query + 1) * heads];
        let q_row = &q_index[query * heads * head_dim..(query + 1) * heads * head_dim];
        for head in 0..heads {
            let q_head = &q_row[head * head_dim..(head + 1) * head_dim];
            let weight = weights[head];
            for position in 0..n_pos {
                let key = &request.keys[position * head_dim..(position + 1) * head_dim];
                let raw = dot(q_head, key);
                if raw > 0.0 {
                    score[query * n_pos + position] += raw * weight;
                }
            }
        }
    }
    // Reachability: a position becomes visible once the query has passed its last token, which the
    // reference masks rather than skipping, so that an unreachable position can never outrank a
    // reachable one (model.py:561-567).
    for query in 0..seqlen {
        let reach = (request.start_pos + query + 1) / request.ratio;
        for position in reach..n_pos {
            score[query * n_pos + position] = f32::NEG_INFINITY;
        }
    }
    // The two levels: the candidate source ranks blocks here, the layers after it mask with them
    // (model.py:569-575).
    let mut published = None;
    if config.candidate_source_layer == request.layer {
        published = Some(select_candidate_blocks(
            &score,
            seqlen,
            n_pos,
            request.start_pos,
            request.ratio,
            config.candidate_topk_blocks,
            config.candidate_block_size,
        ));
    } else if config.candidate_source_layer < request.layer {
        let candidates = request.candidates.ok_or_else(|| {
            shape(
                "compressed indices",
                format!(
                    "layer {} reads the HSI candidate blocks, but no source has published them",
                    request.layer
                ),
            )
        })?;
        let needed = seqlen * n_pos;
        if candidates.len() < needed {
            return Err(shape(
                "compressed indices",
                format!(
                    "the HSI candidate mask holds {} values; {needed} are needed",
                    candidates.len()
                ),
            ));
        }
        for query in 0..seqlen {
            for position in 0..n_pos {
                if !candidates[query * n_pos + position] {
                    score[query * n_pos + position] = f32::NEG_INFINITY;
                }
            }
        }
    }
    // The top-k by score, re-sorted into position order, then shifted into the shared index space
    // where the window rows come first (model.py:577-580). Ties go to the lower position.
    let topk = config.index_topk.min(n_pos);
    let offset = request.state.window_rows.ok_or_else(|| {
        shape(
            "compressed indices",
            format!(
                "no window length is recorded for layer {}: run attention_forward on this state for this chunk first",
                request.layer
            ),
        )
    })?;
    out.clear();
    out.reserve(seqlen * topk);
    let mut order: Vec<usize> = Vec::with_capacity(n_pos);
    for query in 0..seqlen {
        let reach = (request.start_pos + query + 1) / request.ratio;
        order.clear();
        order.extend(0..n_pos);
        order.sort_by(|left, right| {
            score[query * n_pos + *right]
                .partial_cmp(&score[query * n_pos + *left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.cmp(right))
        });
        order.truncate(topk);
        order.sort_unstable();
        for &position in &order {
            out.push(if position < reach {
                position + offset
            } else {
                NO_INDEX
            });
        }
    }
    Ok(published)
}

/// Level one of the two-level top-k: which compressed positions a query may look at
/// (model.py:583-610).
///
/// Blocks are scored by their best position, the block holding the query's newest position is pinned
/// in whatever its score, the best `topk_blocks` blocks are kept, and a block whose score is -inf
/// - one the query cannot reach at all - is dropped. The mask is per position, row-major per query.
fn select_candidate_blocks(
    score: &[f32],
    seqlen: usize,
    n_pos: usize,
    start_pos: usize,
    ratio: usize,
    topk_blocks: usize,
    block_size: usize,
) -> Vec<bool> {
    let mut mask = vec![false; seqlen * n_pos];
    if block_size == 0 || n_pos == 0 || seqlen == 0 {
        return mask;
    }
    let num_blocks = n_pos.div_ceil(block_size);
    let keep = topk_blocks.min(num_blocks);
    let mut block_scores = vec![f32::NEG_INFINITY; num_blocks];
    let mut order: Vec<usize> = Vec::with_capacity(num_blocks);
    for query in 0..seqlen {
        let row = &score[query * n_pos..(query + 1) * n_pos];
        for (block, best) in block_scores.iter_mut().enumerate() {
            let start = block * block_size;
            let end = (start + block_size).min(n_pos);
            *best = row[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
        }
        let reach = (start_pos + query + 1) / ratio;
        if reach > 0 {
            let last = (reach - 1) / block_size;
            if let Some(best) = block_scores.get_mut(last) {
                *best = f32::INFINITY;
            }
        }
        order.clear();
        order.extend(0..num_blocks);
        order.sort_by(|left, right| {
            block_scores[*right]
                .partial_cmp(&block_scores[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.cmp(right))
        });
        order.truncate(keep);
        for &block in &order {
            if block_scores[block] > f32::NEG_INFINITY {
                let start = block * block_size;
                let end = (start + block_size).min(n_pos);
                mask[query * n_pos + start..query * n_pos + end].fill(true);
            }
        }
    }
    mask
}

/// Writes `new_rows` at `first_row` in a cache, growing it with zeros when it is short.
///
/// The reference's caches are preallocated for `max_seq_len`; these grow with the sequence instead,
/// which is why the write is a resize and a copy rather than an assignment (model.py:523-525).
fn store_rows(cache: &mut Vec<f32>, first_row: usize, new_rows: &[f32], width: usize) {
    if width == 0 || new_rows.is_empty() {
        return;
    }
    let needed = (first_row + new_rows.len() / width) * width;
    if cache.len() < needed {
        cache.resize(needed, 0.0);
    }
    let start = first_row * width;
    cache[start..start + new_rows.len()].copy_from_slice(new_rows);
}

/// Publishes `new_rows` into the channel's copy of a cache.
///
/// The reference's channel holds a pointer at the last owner's cache (model.py:748, 1173), so a
/// different layer taking the slot over replaces its contents instead of extending them. That is
/// what keeps a consumer reading one layer's cache and not a mixture of several.
fn publish_rows(
    cache: &mut Vec<f32>,
    rows: &mut usize,
    owner: &mut Option<usize>,
    layer: usize,
    first_row: usize,
    new_rows: &[f32],
    width: usize,
) {
    if *owner != Some(layer) {
        cache.clear();
        *owner = Some(layer);
    }
    store_rows(cache, first_row, new_rows, width);
    *rows = cache.len().checked_div(width).unwrap_or(0);
}

/// Which compressed positions each query of this chunk attends to, in the shared index space, with
/// [`NO_INDEX`] where a query has nothing (model.py:527-580).
///
/// `q` is this layer's query latent, the output of `wq_a` and `q_norm`: the indexer projects it
/// with its own `wq_b` (model.py:550). `prefix` is the layer prefix, so the indexer's own tensors
/// are `{prefix}.attn.indexer.*`.
///
/// Run [`attention_forward`] on the same state for this chunk first. The published per-query
/// weights and the window length the ids are shifted by are per-step inputs that the fixed signature
/// carries neither of, so the state holds them. A layer that is not an index source reads what its
/// source published instead of computing anything (model.py:725-726).
///
/// `limit` caps the elements of any single tensor this pass materialises.
///
/// [EngineError::Shape] when the state has no step recorded, a layer that reads the channel finds
/// nothing published for this chunk, the published rows are too short, or a tensor has the wrong
/// geometry.
#[allow(clippy::too_many_arguments)] // the prescribed signature is fixed by the task and clippy's limit is 7; all eight inputs are distinct.
pub fn compressed_indices(
    state: &AttnState,
    config: &ModelConfig,
    layer: usize,
    q: &[f32],
    src: &dyn WeightSource,
    prefix: &str,
    out: &mut Vec<usize>,
    limit: usize,
) -> Result<(), EngineError> {
    let src = Reads { source: src, limit };
    out.clear();
    let ratio = layer_ratio(config, layer);
    if ratio == 0 {
        // No compressor, so no compressed positions and no indexer at all (model.py:775).
        return Ok(());
    }
    let Some((start_pos, seqlen)) = state.step else {
        return Err(shape(
            "compressed indices",
            format!(
                "layer {layer} has no step recorded: run attention_forward on this state first"
            ),
        ));
    };
    if !is_index_source(config, layer) {
        // model.py:725-726: the layers between index sources reuse what their source published.
        return with_shared(|shared| {
            if !shared.topk_ready || shared.topk_step != Some((start_pos, seqlen)) {
                return Err(shape(
                    "compressed indices",
                    format!(
                        "layer {layer} reuses the compressed positions of its index source, but none has published any for the chunk at {start_pos} of {seqlen} rows"
                    ),
                ));
            }
            out.extend_from_slice(&shared.topk);
            Ok(())
        });
    }
    let n_pos = (start_pos + seqlen) / ratio;
    if n_pos == 0 {
        // model.py:729-730: nothing to choose from yet, and the empty answer is published as such.
        with_shared(|shared| {
            shared.topk.clear();
            shared.topk_rows = seqlen;
            shared.topk_cols = 0;
            shared.topk_ready = true;
            shared.topk_step = Some((start_pos, seqlen));
        });
        return Ok(());
    }
    let owns = owns_index_keys(config, layer);
    let published = with_shared(|shared| {
        let keys: &[f32] = if owns {
            &state.indexer
        } else {
            &shared.index_keys
        };
        let candidates =
            (config.candidate_source_layer < layer).then_some(shared.candidates.as_slice());
        let request = IndexRequest {
            state,
            config,
            layer,
            q,
            src,
            prefix,
            keys,
            candidates,
            start_pos,
            seqlen,
            ratio,
            n_pos,
        };
        index_source_indices(&request, out)
    })?;
    with_shared(|shared| {
        shared.topk.clear();
        shared.topk.extend_from_slice(out);
        shared.topk_rows = seqlen;
        shared.topk_cols = out.len().checked_div(seqlen).unwrap_or(0);
        shared.topk_ready = true;
        shared.topk_step = Some((start_pos, seqlen));
        if let Some(mask) = published {
            shared.candidates = mask;
            shared.candidate_rows = seqlen;
            shared.candidate_cols = n_pos;
            shared.candidate_step = Some((start_pos, seqlen));
        }
    });
    Ok(())
}

/// The chunk's queries: `wq_b(q_norm(wq_a(x)))` with the RoPE tail rotated, and the latent the
/// indexer projects with its own `wq_b` (model.py:770-772, 550).
///
/// # Errors
///
/// [EngineError] from any weight read or shape mismatch.
fn query_path(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    src: Reads<'_>,
) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
    let dim = config.dim;
    let rank = config.q_lora_rank;
    let head_dim = config.head_dim;
    let heads = config.n_heads;
    let seqlen = rows_of(x, dim, "attention input")?;
    let wq_a = dense_weight(src, &format!("{prefix}.attn.wq_a"), rank, dim)?;
    let q_norm = weight(src, &format!("{prefix}.attn.q_norm.weight"))?;
    let wq_b = dense_weight(src, &format!("{prefix}.attn.wq_b"), heads * head_dim, rank)?;
    let mut latent = vec![0.0; seqlen * rank];
    linalg::matmul(&mut latent, x, &wq_a, rank, dim, seqlen)?;
    linalg::rms_norm(&mut latent, &q_norm, rank, config.norm_eps as f32)?;
    let mut q = vec![0.0; seqlen * heads * head_dim];
    linalg::matmul(&mut q, &latent, &wq_b, heads * head_dim, rank, seqlen)?;
    let tables =
        rope_tables_for_layer(config, layer, &head_row_positions(start_pos, seqlen, heads));
    rotate_rows(
        &mut q,
        seqlen * heads,
        head_dim,
        config.rope_head_dim,
        &tables,
        false,
    )?;
    Ok((q, latent))
}

/// A chunk's window KV: `kv_norm(wkv(x))` with its RoPE tail rotated, then the whole row
/// round-tripped through fp8 in place (model.py:700-707).
///
/// The round trip covers the rotated tail too, which is why it comes last: the reference quantises
/// the post-RoPE vector (model.py:702-707).
///
/// # Errors
///
/// [EngineError] from any weight read or shape mismatch.
fn window_kv(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    src: Reads<'_>,
) -> Result<Vec<f32>, EngineError> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let seqlen = rows_of(x, dim, "attention input")?;
    let wkv = dense_weight(src, &format!("{prefix}.attn.wkv"), head_dim, dim)?;
    let norm = weight(src, &format!("{prefix}.attn.kv_norm.weight"))?;
    let mut kv = vec![0.0; seqlen * head_dim];
    linalg::matmul(&mut kv, x, &wkv, head_dim, dim, seqlen)?;
    linalg::rms_norm(&mut kv, &norm, head_dim, config.norm_eps as f32)?;
    let positions: Vec<usize> = (start_pos..start_pos + seqlen).collect();
    let tables = rope_tables_for_layer(config, layer, &positions);
    rotate_rows(
        &mut kv,
        seqlen,
        head_dim,
        config.rope_head_dim,
        &tables,
        false,
    )?;
    round_trip_rows(
        &mut kv,
        head_dim,
        WINDOW_QUANT_BLOCK,
        Element::Fp8,
        ScaleRule::PowerOfTwo,
    );
    Ok(kv)
}

/// The per-query indexer weights: `weights_proj(x)` times `index_head_dim^-0.5 * n_heads^-0.5`
/// (model.py:513, 555).
///
/// The reference computes that factor with Python floats and narrows it where it meets the tensor, so
/// it is computed in f64 here and narrowed once.
///
/// # Errors
///
/// [EngineError] from any weight read or shape mismatch.
fn index_weights(
    config: &ModelConfig,
    prefix: &str,
    x: &[f32],
    seqlen: usize,
    src: Reads<'_>,
) -> Result<Vec<f32>, EngineError> {
    let dim = config.dim;
    let heads = config.index_n_heads;
    let projection = weight(src, &format!("{prefix}.attn.indexer.weights_proj.weight"))?;
    let mut values = vec![0.0; seqlen * heads];
    linalg::matmul(&mut values, x, &projection, heads, dim, seqlen)?;
    let factor = ((config.index_head_dim as f64).powf(-0.5) * (heads as f64).powf(-0.5)) as f32;
    for value in values.iter_mut() {
        *value *= factor;
    }
    Ok(values)
}

/// Turns this step's latent into index keys, round-trips them and stores them (model.py:536-548).
///
/// The reference rotates a latent that stands for the first token of its group, so group `j` takes
/// position `j * ratio` (model.py:538-543). It runs before the compressed cache is written, because
/// the indexer needs the latent before RoPE (model.py:531, 749-750).
///
/// # Errors
///
/// [EngineError] from any weight read or shape mismatch.
fn write_index_keys(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    start_pos: usize,
    state: &mut AttnState,
    src: Reads<'_>,
) -> Result<(), EngineError> {
    if !owns_index_keys(config, layer) || state.latent_rows == 0 {
        return Ok(());
    }
    let ratio = layer_ratio(config, layer);
    if ratio == 0 {
        return Ok(());
    }
    let head_dim = config.head_dim;
    let index_head_dim = config.index_head_dim;
    let wk = weight(src, &format!("{prefix}.attn.indexer.wk.weight"))?;
    let norm = weight(src, &format!("{prefix}.attn.indexer.k_norm.weight"))?;
    let mut keys = vec![0.0; state.latent_rows * index_head_dim];
    linalg::matmul(
        &mut keys,
        &state.latent,
        &wk,
        index_head_dim,
        head_dim,
        state.latent_rows,
    )?;
    linalg::rms_norm(&mut keys, &norm, index_head_dim, config.norm_eps as f32)?;
    let first_row = start_pos / ratio;
    let positions: Vec<usize> = (0..state.latent_rows)
        .map(|row| (first_row + row) * ratio)
        .collect();
    let tables = rope_tables_for_layer(config, layer, &positions);
    rotate_rows(
        &mut keys,
        state.latent_rows,
        index_head_dim,
        config.rope_head_dim,
        &tables,
        false,
    )?;
    round_trip_rows(
        &mut keys,
        index_head_dim,
        INDEX_QUANT_BLOCK,
        Element::Fp4,
        ScaleRule::PowerOfTwo,
    );
    store_rows(&mut state.indexer, first_row, &keys, index_head_dim);
    with_shared(|shared| {
        publish_rows(
            &mut shared.index_keys,
            &mut shared.index_rows,
            &mut shared.index_owner,
            layer,
            first_row,
            &keys,
            index_head_dim,
        );
    });
    Ok(())
}

/// Rotates this step's latent, round-trips it through fp4 with E4M3 scales and stores it
/// (model.py:751-761).
///
/// # Errors
///
/// [EngineError] from a shape mismatch.
fn write_compressed_cache(
    config: &ModelConfig,
    layer: usize,
    start_pos: usize,
    state: &mut AttnState,
) -> Result<(), EngineError> {
    if state.latent_rows == 0 || !is_kv_source(config, layer) {
        return Ok(());
    }
    let ratio = layer_ratio(config, layer);
    if ratio == 0 {
        return Ok(());
    }
    let head_dim = config.head_dim;
    let first_row = start_pos / ratio;
    let positions: Vec<usize> = (0..state.latent_rows)
        .map(|row| (first_row + row) * ratio)
        .collect();
    let tables = rope_tables_for_layer(config, layer, &positions);
    rotate_rows(
        &mut state.latent,
        state.latent_rows,
        head_dim,
        config.rope_head_dim,
        &tables,
        false,
    )?;
    round_trip_rows(
        &mut state.latent,
        head_dim,
        COMPRESS_QUANT_BLOCK,
        Element::Fp4,
        ScaleRule::E4M3Six,
    );
    let rows = std::mem::take(&mut state.latent);
    store_rows(&mut state.compressed, first_row, &rows, head_dim);
    with_shared(|shared| {
        publish_rows(
            &mut shared.compressed,
            &mut shared.compressed_rows,
            &mut shared.compressed_owner,
            layer,
            first_row,
            &rows,
            head_dim,
        );
    });
    state.latent = rows;
    Ok(())
}

/// The compressed KV rows this layer attends to (model.py:739-763).
///
/// A layer with a compressor reads its own cache; every other layer reads the rows its source
/// published, and a missing or short publication is an error rather than a zero cache.
///
/// # Errors
///
/// [EngineError::Shape] when the cache the layer needs does not hold `compress_len` rows.
fn fetch_compressed(
    config: &ModelConfig,
    layer: usize,
    state: &AttnState,
    compress_len: usize,
) -> Result<Vec<f32>, EngineError> {
    let head_dim = config.head_dim;
    let needed = compress_len * head_dim;
    if is_kv_source(config, layer) {
        if state.compressed.len() < needed {
            return Err(shape(
                "compressed KV",
                format!(
                    "layer {layer} holds {} values; {compress_len} rows of {head_dim} need {needed}",
                    state.compressed.len()
                ),
            ));
        }
        Ok(state.compressed[..needed].to_vec())
    } else {
        with_shared(|shared| {
            if shared.compressed_rows < compress_len {
                return Err(shape(
                    "compressed KV",
                    format!(
                        "layer {layer} reads what its kv source published, but {} rows of {compress_len} are there",
                        shared.compressed_rows
                    ),
                ));
            }
            Ok(shared.compressed[..needed].to_vec())
        })
    }
}

/// The grouped output projection (model.py:783-788).
///
/// `wo_a` is block-diagonal: group `g`'s slice of the head dimension goes through the
/// `o_lora_rank` rows that belong to it, and `wo_b` mixes the `o_groups * o_lora_rank` results
/// back to `dim`. The weight is fp8 in the file and bf16 after convert.py, so its grid is optional.
///
/// # Errors
///
/// [EngineError] from any weight read or shape mismatch.
fn output_projection(
    config: &ModelConfig,
    prefix: &str,
    src: Reads<'_>,
    out: &mut [f32],
    seqlen: usize,
) -> Result<(), EngineError> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let heads = config.n_heads;
    let groups = config.o_groups;
    let per_group = heads / groups;
    let group_in = per_group * head_dim;
    let wide = groups * config.o_lora_rank;
    let wo_a = scaled_or_plain_weight(src, &format!("{prefix}.attn.wo_a"), wide, group_in)?;
    let wo_b = dense_weight(src, &format!("{prefix}.attn.wo_b"), dim, wide)?;
    let mut grouped = vec![0.0; seqlen * wide];
    for query in 0..seqlen {
        for group in 0..groups {
            let source = &out[query * heads * head_dim + group * group_in..][..group_in];
            let block =
                &wo_a[group * config.o_lora_rank * group_in..][..config.o_lora_rank * group_in];
            let dest =
                &mut grouped[query * wide + group * config.o_lora_rank..][..config.o_lora_rank];
            linalg::matmul(dest, source, block, config.o_lora_rank, group_in, 1)?;
        }
    }
    linalg::matmul(out, &grouped, &wo_b, dim, wide, seqlen)?;
    Ok(())
}

/// The backbone CSA2 pass for one layer (model.py:765-789).
///
/// The window path always runs; the compressed path runs only for a layer whose `compress_ratio` is
/// positive (model.py:775), which is where the compressor, the indexer and the cross-layer channel
/// come in. The output is the attention result projected by `wo_a` and `wo_b`, so `out`'s first
/// `seqlen * dim` values hold the result afterwards; during the pass it is written as
/// `seqlen * n_heads * head_dim` values first, so the buffer must be the wider of the two and must
/// not be resized between layers (a buffer that is too short is a typed [EngineError::Shape]).
///
/// `limit` caps the elements of any single tensor this pass materialises.
///
/// [EngineError] from any weight read or shape mismatch, and [EngineError::Shape] when a layer reads a
/// published cache that no source has written for this chunk.
#[allow(clippy::too_many_arguments)] // the reference's own signature: the chunk, its first position, the layer, its name, the state, the source and the output buffer are all distinct inputs and bundling them would hide which one a call got wrong.
pub fn attention_forward(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: &dyn WeightSource,
    out: &mut [f32],
    limit: usize,
) -> Result<(), EngineError> {
    let src = Reads { source: src, limit };
    let dim = config.dim;
    let head_dim = config.head_dim;
    let heads = config.n_heads;
    let seqlen = rows_of(x, dim, "attention input")?;
    if out.len() < seqlen * dim {
        return Err(shape(
            "attention output",
            format!(
                "{seqlen} rows need {} values, got {}",
                seqlen * dim,
                out.len()
            ),
        ));
    }
    // The attention write is wider than the projection: one concatenated head vector per query
    // before `output_projection` narrows it to `dim`. The caller's buffer is therefore the max of
    // the two widths, and a buffer sized only `seqlen * dim` would panic below at released
    // geometry (64 heads of 512 = 32,768 against dim 5,120), so it is a typed refusal here.
    let wide = seqlen
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(head_dim))
        .ok_or_else(|| {
            shape(
                "attention output",
                "the head geometry does not fit a buffer".to_owned(),
            )
        })?;
    if out.len() < wide {
        return Err(shape(
            "attention output",
            format!(
                "{seqlen} rows of {heads} heads of {head_dim} write {wide} values, got {}",
                out.len()
            ),
        ));
    }
    if head_dim == 0 || config.o_groups == 0 || !heads.is_multiple_of(config.o_groups) {
        return Err(shape(
            "attention geometry",
            format!(
                "{heads} heads of {head_dim} do not split into {} groups",
                config.o_groups
            ),
        ));
    }
    let end_pos = start_pos + seqlen;
    let ratio = layer_ratio(config, layer);
    // The number of complete groups after this chunk (model.py:744). Zero when there is no
    // compression, and zero division is what a ratio of zero would otherwise be.
    let compress_len = end_pos.checked_div(ratio).unwrap_or(0);
    let scale = (head_dim as f64).powf(-0.5) as f32;
    let (q, latent) = query_path(config, layer, prefix, x, start_pos, src)?;
    let kv_rows = window_kv(config, layer, prefix, x, start_pos, src)?;
    state.step = Some((start_pos, seqlen));
    let window_rows = if start_pos == 0 {
        seqlen
    } else {
        config.window_size
    };
    state.window_rows = Some(window_rows);
    // The compressed side of the layer (model.py:775-778).
    let mut compressed_ids: Vec<usize> = Vec::new();
    let mut compressed_rows: Vec<f32> = Vec::new();
    if ratio > 0 {
        compressor_step(config, layer, prefix, x, start_pos, state, src)?;
        // The index keys read the latent before the compressed cache rotates it (model.py:531).
        write_index_keys(config, layer, prefix, start_pos, state, src)?;
        if is_index_source(config, layer) && compress_len > 0 {
            state.index_weights = index_weights(config, prefix, x, seqlen, src)?;
        } else {
            state.index_weights.clear();
        }
        compressed_indices(
            state,
            config,
            layer,
            &latent,
            src.source,
            prefix,
            &mut compressed_ids,
            limit,
        )?;
        write_compressed_cache(config, layer, start_pos, state)?;
        state.compressed_len = compress_len;
        compressed_rows = fetch_compressed(config, layer, state, compress_len)?;
    } else {
        state.latent.clear();
        state.latent_rows = 0;
        state.compressed_len = 0;
        state.index_weights.clear();
    }
    let compressed_stride = compressed_ids.len().checked_div(seqlen).unwrap_or(0);
    // The window KV the indices index into: the whole chunk during prefill, the whole ring after it.
    let sink = weight(src, &format!("{prefix}.attn.attn_sink"))?;
    let mut kv = vec![0.0; (window_rows + compressed_rows.len() / head_dim) * head_dim];
    kv[window_rows * head_dim..].copy_from_slice(&compressed_rows);
    if start_pos == 0 {
        // Prefill attends over this chunk and keeps its last window for later steps (model.py:708-716).
        let first = seqlen.saturating_sub(config.window_size);
        for row in first..seqlen {
            state
                .window
                .write(row, &kv_rows[row * head_dim..(row + 1) * head_dim])?;
        }
        kv[..seqlen * head_dim].copy_from_slice(&kv_rows[..seqlen * head_dim]);
    }
    let ids = window_indices(config, start_pos, seqlen);
    let window_width = if start_pos == 0 {
        seqlen.min(config.window_size)
    } else {
        config.window_size
    };
    let mut combined: Vec<usize> = Vec::with_capacity(window_width + compressed_stride);
    for query in 0..seqlen {
        if start_pos > 0 {
            // A step after prefill writes its own row into the ring before attending, so the ring
            // holds the positions this query may see and nothing later (model.py:717-719).
            let position = start_pos + query;
            state
                .window
                .write(position, &kv_rows[query * head_dim..(query + 1) * head_dim])?;
            for slot in 0..config.window_size {
                kv[slot * head_dim..(slot + 1) * head_dim]
                    .copy_from_slice(state.window.slot_row(slot));
            }
        }
        combined.clear();
        combined.extend_from_slice(&ids[query * window_width..(query + 1) * window_width]);
        combined.extend_from_slice(
            &compressed_ids[query * compressed_stride..(query + 1) * compressed_stride],
        );
        sparse_attention(
            &q[query * heads * head_dim..(query + 1) * heads * head_dim],
            &kv,
            &sink,
            &combined,
            heads,
            head_dim,
            scale,
            &mut out[query * heads * head_dim..(query + 1) * heads * head_dim],
        )?;
        // Off-by-default capture hook (task-14): offers the query's inputs and pre-projection
        // output to the fixture generator. It is inactive unless explicitly armed, and it copies
        // values the reference has already computed, so the reference arithmetic is untouched.
        attention_case::record(attention_case::Frame {
            layer,
            position: start_pos + query,
            start_pos,
            seqlen,
            ratio,
            heads,
            head_dim,
            window_rows,
            window_width,
            compressed_stride,
            q: &q[query * heads * head_dim..(query + 1) * heads * head_dim],
            sink: &sink,
            window_ids: &ids[query * window_width..(query + 1) * window_width],
            window_kv: &kv[..window_rows * head_dim],
            compressed_ids: &compressed_ids
                [query * compressed_stride..(query + 1) * compressed_stride],
            compressed_kv: &kv[window_rows * head_dim..],
            out: &out[query * heads * head_dim..(query + 1) * heads * head_dim],
        });
    }
    let tables =
        rope_tables_for_layer(config, layer, &head_row_positions(start_pos, seqlen, heads));
    rotate_rows(
        &mut out[..seqlen * heads * head_dim],
        seqlen * heads,
        head_dim,
        config.rope_head_dim,
        &tables,
        true,
    )?;
    output_projection(config, prefix, src, out, seqlen)?;
    Ok(())
}

/// The DSpark stage pass, which attends the main rows through its own ring (model.py:1032-1074).
///
/// `start_pos` is the position of the first row of `main_x`, which is what the reference's own
/// `freqs_cis[start_pos : start_pos + seqlen]` means and what the caller in `forward::mtp` passes;
/// the block of draft rows in `x` follows it at `start_pos + main_rows` (model.py:1039, 1055). When
/// `main_x` is empty the first row of `x` is at `start_pos` instead.
///
/// The main rows' KV is written into this stage's ring at their own positions and every draft query
/// attends `0..min(window_size, start_pos + main_rows)` of the ring's slots - the positions written
/// so far, which is the reference's `arange(min(win, start_pos + 1))` generalised to a main chunk of
/// any length - plus all of the block's own rows at `window_size + row`
/// (`get_dspark_topk_idxs`, model.py:1021-1029). The block is therefore not causal inside itself:
/// the reference lets the drafts see each other, while the ring itself never holds a position at or
/// after the block.
///
/// At the start of a sequence (`start_pos == 0`) there is nothing to attend to, so the reference only
/// seeds the ring with the main rows and returns `x` unchanged (model.py:1044-1052); with no main
/// rows at all this seeds from `x` and then keeps them, which is the same step for a stage that has
/// no main input. A stage that has main rows only ever stores those, as the reference does
/// (model.py:1065). The reference asserts `compress_ratio == 0` for this class (model.py:1034), and
/// no compressor or indexer path is taken here. The query is rotated with the stage's own tables, an
/// index outside the gathered rows stays a zero row with a masked score, and `out`'s first
/// `block_rows * dim` values hold the projected result afterwards (like the backbone pass, it is
/// written as `block_rows * n_heads * head_dim` values first, so the buffer must be the wider of
/// the two; too short is a typed [EngineError::Shape]).
///
/// `limit` caps the elements of any single tensor this pass materialises.
///
/// [EngineError] from any weight read or shape mismatch.
#[allow(clippy::too_many_arguments)] // the reference's own signature: the stage's rows, the main rows, the first position, the name, the state, the source and the output buffer are all distinct inputs and bundling them would hide which one a call got wrong.
pub fn dspark_attention_forward(
    config: &ModelConfig,
    stage: usize,
    prefix: &str,
    x: &[f32],
    main_x: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: &dyn WeightSource,
    out: &mut [f32],
    limit: usize,
) -> Result<(), EngineError> {
    let src = Reads { source: src, limit };
    let dim = config.dim;
    let head_dim = config.head_dim;
    let heads = config.n_heads;
    let window = config.window_size;
    let block_rows = rows_of(x, dim, "dspark input")?;
    let main_rows = rows_of(main_x, dim, "dspark main input")?;
    if out.len() < block_rows * dim {
        return Err(shape(
            "dspark output",
            format!(
                "{block_rows} rows need {} values, got {}",
                block_rows * dim,
                out.len()
            ),
        ));
    }
    if head_dim == 0 || config.o_groups == 0 || !heads.is_multiple_of(config.o_groups) {
        return Err(shape(
            "dspark geometry",
            format!(
                "{heads} heads of {head_dim} do not split into {} groups",
                config.o_groups
            ),
        ));
    }
    // As in the backbone pass: the stage writes `block_rows * n_heads * head_dim` values before
    // its projection narrows them to `block_rows * dim`, so a caller-sized `block_rows * dim`
    // buffer is a typed refusal rather than a panic at released geometry.
    let wide = block_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(head_dim))
        .ok_or_else(|| {
            shape(
                "dspark output",
                "the head geometry does not fit a buffer".to_owned(),
            )
        })?;
    if out.len() < wide {
        return Err(shape(
            "dspark output",
            format!(
                "{block_rows} rows of {heads} heads of {head_dim} write {wide} values, got {}",
                out.len()
            ),
        ));
    }
    // The stage's own layer id, which selects its RoPE geometry (model.py:636-640, 1107).
    let layer = config.n_layers + stage;
    let draft_start = start_pos + main_rows;
    if start_pos == 0 {
        // Prefill: seed the ring from the main rows and return the stage's rows untouched
        // (model.py:1044-1052), or from the stage's own rows when no main input is passed.
        let (seed, seed_rows) = if main_rows > 0 {
            (
                window_kv(config, layer, prefix, main_x, start_pos, src)?,
                main_rows,
            )
        } else {
            (
                window_kv(config, layer, prefix, x, start_pos, src)?,
                block_rows,
            )
        };
        let first = seed_rows.saturating_sub(window);
        for row in first..seed_rows {
            state
                .window
                .write(start_pos + row, &seed[row * head_dim..(row + 1) * head_dim])?;
        }
        out[..block_rows * dim].copy_from_slice(&x[..block_rows * dim]);
        return Ok(());
    }
    // A step after prefill: the block's own queries and KV at the draft positions (model.py:1055-1062).
    let (q, _latent) = query_path(config, layer, prefix, x, draft_start, src)?;
    let kv_rows = window_kv(config, layer, prefix, x, draft_start, src)?;
    if main_rows > 0 {
        let main_kv = window_kv(config, layer, prefix, main_x, start_pos, src)?;
        for row in 0..main_rows {
            state.window.write(
                start_pos + row,
                &main_kv[row * head_dim..(row + 1) * head_dim],
            )?;
        }
    }
    // The reference's ids: the ring's written slots, then the block's own rows by their offset into
    // the gathered KV.
    let filled = window.min(draft_start);
    let mut ids = Vec::with_capacity(filled + block_rows);
    ids.extend(0..filled);
    ids.extend(window..window + block_rows);
    let mut kv = vec![0.0; (window + block_rows) * head_dim];
    for slot in 0..window {
        kv[slot * head_dim..(slot + 1) * head_dim].copy_from_slice(state.window.slot_row(slot));
    }
    kv[window * head_dim..].copy_from_slice(&kv_rows);
    let sink = weight(src, &format!("{prefix}.attn.attn_sink"))?;
    let scale = (head_dim as f64).powf(-0.5) as f32;
    for row in 0..block_rows {
        sparse_attention(
            &q[row * heads * head_dim..(row + 1) * heads * head_dim],
            &kv,
            &sink,
            &ids,
            heads,
            head_dim,
            scale,
            &mut out[row * heads * head_dim..(row + 1) * heads * head_dim],
        )?;
    }
    let tables = rope_tables_for_layer(
        config,
        layer,
        &head_row_positions(draft_start, block_rows, heads),
    );
    rotate_rows(
        &mut out[..block_rows * heads * head_dim],
        block_rows * heads,
        head_dim,
        config.rope_head_dim,
        &tables,
        true,
    )?;
    output_projection(config, prefix, src, out, block_rows)?;
    if main_rows == 0 {
        // With no main rows the stage's own rows are what a later step can attend to.
        for row in 0..block_rows {
            state.window.write(
                draft_start + row,
                &kv_rows[row * head_dim..(row + 1) * head_dim],
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The element limit the tests' passes run under: far above any tensor of a tiny configuration,
    /// so only the limit test ever sees this number.
    const LIMIT: usize = 1 << 20;

    use std::collections::BTreeMap;

    use crate::dtype::DType;
    use crate::tensor::{TensorPlan, TensorSpec};
    use crate::weights::WeightError;

    /// A weight source built from the tensor plan, so a test cannot get a name, dtype or shape wrong,
    /// with small deterministic values that decode to something benign.
    ///
    /// The bf16 and fp8 values are exactly representable, so a weight reads back as what was written;
    /// scale grids are 1.0 (E8M0 code 127), which leaves the fp8 weights as they were encoded.
    struct Hand {
        held: BTreeMap<String, (DType, Vec<usize>, Vec<u8>)>,
    }

    /// A small exactly-representable value for element `i`.
    fn pattern(i: usize) -> f32 {
        (i % 11) as f32 * 0.0625 - 0.25
    }

    /// The bf16 bit pattern of an f32, truncated. Every value here has four fractional bits, so the
    /// truncation loses nothing.
    fn to_bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    impl Hand {
        fn for_config(config: &ModelConfig) -> Self {
            let plan = TensorPlan::language_model(config);
            let mut held = BTreeMap::new();
            for spec in plan.specs() {
                let elements = usize::try_from(spec.elements()).expect("a test plan fits usize");
                let mut bytes = vec![0u8; spec.dtype.storage_bytes(elements)];
                match spec.dtype {
                    DType::F32 => {
                        let values: Vec<u8> = (0..elements)
                            .flat_map(|i| pattern(i).to_le_bytes())
                            .collect();
                        bytes = values;
                    }
                    DType::BF16 => {
                        let values: Vec<u8> = (0..elements)
                            .flat_map(|i| to_bf16(pattern(i)).to_le_bytes())
                            .collect();
                        bytes = values;
                    }
                    DType::F8E4M3 => {
                        bytes = (0..elements)
                            .map(|i| quant::e4m3_encode(pattern(i) * 0.25))
                            .collect();
                    }
                    DType::F8E8M0 => bytes.fill(127),
                    _ => {}
                }
                held.insert(spec.name.clone(), (spec.dtype, spec.shape.clone(), bytes));
            }
            Self { held }
        }

        /// Removes a tensor, which is how a test names an input a path must not read.
        fn drop_tensor(&mut self, name: &str) {
            let removed = self.held.remove(name);
            assert!(removed.is_some(), "{name} is not in the plan");
        }
    }

    impl WeightSource for Hand {
        fn names(&self) -> Vec<String> {
            self.held.keys().cloned().collect()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.held
                .get(name)
                .map(|(dtype, shape, _)| TensorSpec {
                    name: name.to_owned(),
                    dtype: *dtype,
                    shape: shape.clone(),
                })
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            self.held
                .get(name)
                .map(|(_, _, data)| data.clone())
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }
    }

    /// A chunk of `rows` deterministic input rows.
    fn chunk(config: &ModelConfig, rows: usize, seed: usize) -> Vec<f32> {
        let dim = config.dim;
        (0..rows * dim)
            .map(|i| ((i + seed * dim) % 13) as f32 * 0.0625 - 0.375)
            .collect()
    }

    /// Compares two slices within a tolerance, which is what two arithmetic orders of the same sum
    /// need. The bits are not compared because the two paths group their additions differently.
    fn close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len(), "lengths differ");
        for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (a - b).abs() <= tolerance,
                "element {index}: {a} and {b} differ by more than {tolerance}"
            );
        }
    }

    /// The reference's own YaRN geometry: a compressed layer takes `compress_rope_theta` and the
    /// correction `original_seq_len` turns on, a plain layer takes `rope_theta` and none
    /// (model.py:680-687).
    #[test]
    fn the_rope_geometry_follows_the_layer_compression() {
        let config = ModelConfig {
            compress_ratios: vec![2, 0, 0],
            ..ModelConfig::tiny()
        };
        let base = rope_tables(&config, &[1, 2]);
        let compressed = rope_tables_for_layer(&config, 0, &[1, 2]);
        let plain = rope_tables_for_layer(&config, 1, &[1, 2]);
        assert_eq!(
            base.cos, plain.cos,
            "a layer with no compression uses the base table"
        );
        assert_eq!(base.sin, plain.sin);
        assert_ne!(
            base.cos, compressed.cos,
            "a compressing layer uses compress_rope_theta, so its table differs at position 1"
        );

        // The correction's ramp reaches one pair and not the other at this geometry: pair 0 keeps its
        // frequency whatever the factor, pair 1 is divided by it (model.py:377-386).
        let by_factor = |factor: f64| {
            let config = ModelConfig {
                rope_factor: factor,
                ..config.clone()
            };
            rope_tables_for_layer(&config, 0, &[1])
        };
        let one = by_factor(1.0);
        let two = by_factor(2.0);
        let angle = |tables: &RopeTables, pair: usize| tables.sin[pair].atan2(tables.cos[pair]);
        assert_eq!(one.cos[0], two.cos[0], "pair 0 is below the ramp");
        assert!(
            (angle(&two, 1) * 2.0 - angle(&one, 1)).abs() < 1e-9,
            "pair 1 is past the ramp, so twice the factor halves its frequency: {} and {}",
            angle(&two, 1),
            angle(&one, 1)
        );
    }

    /// The ring holds the last `window_size` positions, addressed by their own position
    /// (model.py:709-718).
    #[test]
    fn the_ring_wraps_and_evicts() {
        let config = ModelConfig::tiny();
        let head_dim = config.head_dim;
        let mut ring = WindowRing::for_config(&config);
        assert!(ring.is_empty());
        assert_eq!(ring.last_position(), None);
        for position in 0..12usize {
            ring.write(position, &vec![position as f32; head_dim])
                .expect("a head vector");
        }
        assert_eq!(ring.len(), config.window_size);
        assert_eq!(ring.last_position(), Some(11));
        assert_eq!(ring.row(11), Some(&vec![11.0f32; head_dim][..]));
        assert!(ring.row(4).is_some(), "the last eight positions survive");
        assert!(
            ring.row(3).is_none(),
            "position 3 was evicted by position 11"
        );

        // A row that is not one head vector, and a ring with no slots, are refused rather than
        // written.
        let error = ring.write(12, &[0.0; 3]).unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        let mut empty = WindowRing::for_config(&ModelConfig {
            window_size: 0,
            ..ModelConfig::tiny()
        });
        assert!(empty.is_empty());
        assert_eq!(empty.row(0), None);
        assert!(empty.write(0, &[0.0; 4]).is_err());
    }

    /// Prefill ids are causal and chunk-relative; the ids after prefill are the ring's slots in the
    /// order the reference lists them (model.py:409-426).
    #[test]
    fn window_indices_pin_causality_and_the_ring_order() {
        let config = ModelConfig::tiny();
        let prefill = window_indices(&config, 0, 5);
        assert_eq!(prefill.len(), 5 * 5);
        assert_eq!(
            &prefill[..5],
            &[0, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX],
            "the first query sees only itself"
        );
        assert_eq!(&prefill[4 * 5..], &[0, 1, 2, 3, 4]);

        let decode = window_indices(&config, 10, 1);
        assert_eq!(decode, vec![3, 4, 5, 6, 7, 0, 1, 2]);

        // A chunk longer than the window: the width is the window, and the oldest row of query 9 is
        // row 2, not row 0.
        let long = window_indices(&config, 0, 10);
        assert_eq!(long.len(), 10 * 8);
        assert_eq!(&long[8 * 8..9 * 8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&long[9 * 8..10 * 8], &[2, 3, 4, 5, 6, 7, 8, 9]);

        let early = window_indices(&config, 2, 1);
        assert_eq!(
            early,
            vec![NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, 0, 1, 2],
            "a short sequence leaves the slots it has not reached marked"
        );
        assert!(window_indices(&config, 0, 0).is_empty());
    }

    /// The RoPE table is looked up by position, not by row: a token's heads share its table entry and
    /// the next token moves on to the next entry (model.py:392-406, 772).
    ///
    /// A chunk of several positions with several heads is what makes this visible: a table built for
    /// one position per row cannot rotate it at all.
    #[test]
    fn the_rope_table_is_looked_up_by_position() {
        assert_eq!(head_row_positions(4, 3, 2), vec![4, 4, 5, 5, 6, 6]);
        assert_eq!(head_row_positions(0, 1, 1), vec![0]);

        // Three positions and two heads: the rotation has twelve cosines to read, six of them the
        // three positions' entries twice over.
        let config = ModelConfig {
            compress_ratios: vec![0, 0, 0],
            ..ModelConfig::tiny()
        };
        let positions = head_row_positions(0, 3, config.n_heads);
        let tables = rope_tables_for_layer(&config, 0, &positions);
        assert_eq!(tables.cos.len(), positions.len() * config.rope_head_dim / 2);
        let hand = Hand::for_config(&config);
        let mut state = AttnState::for_config(&config);
        let mut out = vec![0.0; 3 * config.dim];
        attention_forward(
            &config,
            0,
            "layers.0",
            &chunk(&config, 3, 0),
            0,
            &mut state,
            &hand,
            &mut out,
            LIMIT,
        )
        .expect("a three-token, two-head pass");
    }

    /// A prefill's early rows are what a shorter chunk produces, bit for bit: a query cannot see a row
    /// after it (model.py:417-420).
    #[test]
    fn a_prefill_is_causal_bit_for_bit() {
        let config = ModelConfig {
            compress_ratios: vec![0, 0, 0],
            ..ModelConfig::tiny()
        };
        let hand = Hand::for_config(&config);
        let dim = config.dim;
        let long = chunk(&config, 4, 0);
        let short = chunk(&config, 2, 0);
        let mut state = AttnState::for_config(&config);
        let mut wide = vec![0.0; 4 * dim];
        attention_forward(
            &config, 0, "layers.0", &long, 0, &mut state, &hand, &mut wide, LIMIT,
        )
        .expect("a prefill");
        let mut state = AttnState::for_config(&config);
        let mut narrow = vec![0.0; 2 * dim];
        attention_forward(
            &config,
            0,
            "layers.0",
            &short,
            0,
            &mut state,
            &hand,
            &mut narrow,
            LIMIT,
        )
        .expect("a prefill");
        assert_eq!(
            &wide[..2 * dim],
            &narrow[..],
            "rows 0 and 1 of a four-row prefill must not see rows 2 and 3"
        );
    }

    /// The step-by-step path and the prefill path agree over the same rows (model.py:708-719).
    #[test]
    fn stepping_after_a_prefill_agrees_with_the_prefill() {
        let config = ModelConfig {
            compress_ratios: vec![0, 0, 0],
            ..ModelConfig::tiny()
        };
        let hand = Hand::for_config(&config);
        let dim = config.dim;
        let all = chunk(&config, 4, 0);
        let mut state = AttnState::for_config(&config);
        let mut batch = vec![0.0; 4 * dim];
        attention_forward(
            &config, 0, "layers.0", &all, 0, &mut state, &hand, &mut batch, LIMIT,
        )
        .expect("a prefill");

        let mut state = AttnState::for_config(&config);
        let mut stepped = vec![0.0; 4 * dim];
        attention_forward(
            &config,
            0,
            "layers.0",
            &all[..2 * dim],
            0,
            &mut state,
            &hand,
            &mut stepped[..2 * dim],
            LIMIT,
        )
        .expect("a prefill");
        for position in 2..4usize {
            attention_forward(
                &config,
                0,
                "layers.0",
                &all[position * dim..(position + 1) * dim],
                position,
                &mut state,
                &hand,
                &mut stepped[position * dim..(position + 1) * dim],
                LIMIT,
            )
            .expect("a decode step");
        }
        close(&batch, &stepped, 1e-5);
    }

    /// The attention sink is a denominator term: with one key the output is the key scaled by
    /// `sigmoid(score - sink)`, and a query with nothing to attend to is zero (kernel.py:311-389).
    #[test]
    fn the_sink_only_shrinks_and_an_empty_row_is_zero() {
        let q = [1.0f32, 0.0];
        let key = [0.5f32, 0.25];
        let scale = 1.0f32;
        for sink in [-2.0f32, 0.0, 3.0] {
            let mut out = [0.0f32; 2];
            sparse_attention(&q, &key, &[sink], &[0], 1, 2, scale, &mut out).expect("one key");
            let denominator = 1.0 + (sink - 0.5).exp();
            assert!(
                (out[0] - 0.5 / denominator).abs() < 1e-7,
                "sink {sink} gave {}",
                out[0]
            );
            assert!((out[1] - 0.25 / denominator).abs() < 1e-7);
        }
        let mut masked = [7.0f32; 2];
        sparse_attention(&q, &key, &[0.0], &[NO_INDEX], 1, 2, scale, &mut masked)
            .expect("a masked row");
        assert_eq!(
            masked,
            [0.0, 0.0],
            "a query with nothing to attend to is zero"
        );

        let error = sparse_attention(&q, &key, &[], &[0], 1, 2, scale, &mut [0.0; 2]).unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
    }

    /// The HSI keeps the block holding the query's newest position whatever its score, keeps the best
    /// of the rest, and drops a block the query cannot reach (model.py:583-610).
    #[test]
    fn the_hsi_keeps_the_newest_block_and_drops_an_unreachable_one() {
        // Two queries, six positions, blocks of two, from a chunk at position 4 and ratio two. Query
        // 0 reaches positions 0 and 1, so its pinned block is 0; query 1 reaches up to position 2, so
        // its pinned block is 1. Each is pinned over a better-scoring block.
        let score = [
            0.0f32, 5.0, -1.0, 2.0, 9.0, -3.0, 10.0, 1.0, 0.0, 0.0, 5.0, 5.0,
        ];
        let mask = select_candidate_blocks(&score, 2, 6, 4, 2, 1, 2);
        assert_eq!(
            &mask[..6],
            &[true, true, false, false, false, false],
            "query 0 keeps its pinned block, not the best-scoring one"
        );
        assert_eq!(
            &mask[6..],
            &[false, false, true, true, false, false],
            "query 1 pins a different block"
        );

        // A block whose best position is -inf is dropped even when it is among the best blocks.
        let mut unreachable = score;
        unreachable[4] = f32::NEG_INFINITY;
        unreachable[5] = f32::NEG_INFINITY;
        let mask = select_candidate_blocks(&unreachable, 2, 6, 0, 2, 3, 2);
        assert!(!mask[4] && !mask[5], "an unreachable block is never kept");
        assert_eq!(&mask[..4], &[true; 4], "the reachable blocks are kept");
        assert_eq!(&mask[6..10], &[true; 4]);
    }

    /// The compressor publishes one latent per completed group, and a step that leaves a group
    /// unfinished publishes none (model.py:429-486, 744).
    #[test]
    fn the_compressor_waits_for_a_whole_group() {
        let config = ModelConfig::tiny();
        let hand = Hand::for_config(&config);
        let dim = config.dim;
        let mut state = AttnState::for_config(&config);
        let mut out = vec![0.0; 6 * dim];
        attention_forward(
            &config,
            0,
            "layers.0",
            &chunk(&config, 6, 0),
            0,
            &mut state,
            &hand,
            &mut out,
            LIMIT,
        )
        .expect("a prefill of six rows at ratio two");
        assert_eq!(state.latent_rows, 3);
        assert_eq!(state.compressed_len, 3);
        assert_eq!(
            state.indexer.len(),
            3 * config.index_head_dim,
            "one key per latent row"
        );

        let mut step = vec![0.0; dim];
        attention_forward(
            &config,
            0,
            "layers.0",
            &chunk(&config, 1, 6),
            6,
            &mut state,
            &hand,
            &mut step,
            LIMIT,
        )
        .expect("a step that leaves the group half full");
        assert_eq!(state.latent_rows, 0, "no group completes at position 6");
        assert_eq!(state.compressed_len, 3);

        attention_forward(
            &config,
            0,
            "layers.0",
            &chunk(&config, 1, 7),
            7,
            &mut state,
            &hand,
            &mut step,
            LIMIT,
        )
        .expect("a step that completes the group");
        assert_eq!(state.latent_rows, 1);
        assert_eq!(state.compressed_len, 4);
        assert!(state.latent.iter().all(|value| value.is_finite()));
    }

    /// The positions an index source returns are reachable, sorted, inside the caches, and no more
    /// than `index_topk` per query (model.py:550-580).
    #[test]
    fn the_indexer_returns_reachable_sorted_positions() {
        reset_shared_attention();
        let rows = 6usize;
        let ratio = 2usize;
        // A budget of one is where the reachability mask is visible in the answer: without it an
        // unreachable position can take the only slot and be turned into NO_INDEX.
        for index_topk in [1usize, 3] {
            let config = ModelConfig {
                index_topk,
                ..ModelConfig::tiny()
            };
            let hand = Hand::for_config(&config);
            let dim = config.dim;
            let mut state = AttnState::for_config(&config);
            let mut out = vec![0.0; rows * dim];
            attention_forward(
                &config,
                0,
                "layers.0",
                &chunk(&config, rows, 0),
                0,
                &mut state,
                &hand,
                &mut out,
                LIMIT,
            )
            .expect("a prefill");
            let q = chunk(&config, rows, 3);
            let mut ids = Vec::new();
            compressed_indices(&state, &config, 0, &q, &hand, "layers.0", &mut ids, LIMIT)
                .expect("an indexer over the published keys");
            let window_rows = rows;
            let n_pos = rows / ratio;
            let topk = index_topk.min(n_pos);
            assert_eq!(ids.len(), rows * topk);
            for query in 0..rows {
                let row = &ids[query * topk..(query + 1) * topk];
                let reach = (query + 1) / ratio;
                let mut last = 0usize;
                let mut real = 0usize;
                for &id in row {
                    if id == NO_INDEX {
                        continue;
                    }
                    real += 1;
                    assert!(
                        id >= window_rows,
                        "an id in the window's own space is impossible"
                    );
                    let position = id - window_rows;
                    assert!(
                        position < n_pos,
                        "{position} is outside the {n_pos} published rows"
                    );
                    assert!(
                        position < reach,
                        "query {query} cannot reach position {position}"
                    );
                    assert!(position >= last, "ids must be ascending");
                    last = position + 1;
                }
                // Every reachable position scores finite, so the budget is used in full.
                assert_eq!(
                    real,
                    topk.min(reach),
                    "query {query} with a budget of {topk}"
                );
            }
        }
    }

    /// A layer between sources reads what its source published, and refuses when there is nothing to
    /// read (model.py:1166-1181, 725-726).
    #[test]
    fn a_consumer_reads_the_channel_and_refuses_an_empty_one() {
        reset_shared_attention();
        let config = ModelConfig {
            compress_ratios: vec![2, 2, 1],
            index_source_layers: vec![0],
            candidate_source_layer: 0,
            ..ModelConfig::tiny()
        };
        let hand = Hand::for_config(&config);
        let dim = config.dim;
        let rows = 4usize;
        let x = chunk(&config, rows, 0);
        let q = chunk(&config, rows, 3);

        let mut state = AttnState::for_config(&config);
        let mut out = vec![0.0; rows * dim];
        let error = attention_forward(
            &config, 1, "layers.1", &x, 0, &mut state, &hand, &mut out, LIMIT,
        )
        .expect_err("a consumer before its source must refuse");
        assert!(error.to_string().contains("source"), "{error}");

        // The source runs first, and the consumer then sees exactly the positions the source published.
        let mut source = AttnState::for_config(&config);
        attention_forward(
            &config,
            0,
            "layers.0",
            &x,
            0,
            &mut source,
            &hand,
            &mut out,
            LIMIT,
        )
        .expect("the source's own pass");
        let mut published = Vec::new();
        compressed_indices(
            &source,
            &config,
            0,
            &q,
            &hand,
            "layers.0",
            &mut published,
            LIMIT,
        )
        .expect("the source's own positions");
        let mut consumer = AttnState::for_config(&config);
        attention_forward(
            &config,
            1,
            "layers.1",
            &x,
            0,
            &mut consumer,
            &hand,
            &mut out,
            LIMIT,
        )
        .expect("a consumer after its source");
        let mut reused = Vec::new();
        compressed_indices(
            &consumer,
            &config,
            1,
            &q,
            &hand,
            "layers.1",
            &mut reused,
            LIMIT,
        )
        .expect("the published positions");
        assert_eq!(published, reused);

        // A second sequence must not read the first one's publication.
        reset_shared_attention();
        let mut second = AttnState::for_config(&config);
        let error = attention_forward(
            &config,
            1,
            "layers.1",
            &x,
            0,
            &mut second,
            &hand,
            &mut out,
            LIMIT,
        )
        .expect_err("a reset channel holds nothing");
        assert!(error.to_string().contains("source"), "{error}");
    }

    /// A layer with no compression never reads a compressor or indexer tensor, and a ratio of one
    /// never reads the gate (model.py:461-462, 658-661).
    #[test]
    fn the_compressor_and_indexer_are_read_only_where_they_are_used() {
        reset_shared_attention();
        let plain = ModelConfig {
            compress_ratios: vec![0, 0, 0],
            ..ModelConfig::tiny()
        };
        let mut hand = Hand::for_config(&plain);
        for name in [
            "layers.0.attn.compressor.wkv.weight",
            "layers.0.attn.compressor.norm.weight",
            "layers.0.attn.indexer.wq_b.weight",
            "layers.0.attn.indexer.wq_b.scale",
            "layers.0.attn.indexer.wk.weight",
            "layers.0.attn.indexer.k_norm.weight",
            "layers.0.attn.indexer.weights_proj.weight",
        ] {
            hand.drop_tensor(name);
        }
        let dim = plain.dim;
        let mut state = AttnState::for_config(&plain);
        let mut out = vec![0.0; 4 * dim];
        attention_forward(
            &plain,
            0,
            "layers.0",
            &chunk(&plain, 4, 0),
            0,
            &mut state,
            &hand,
            &mut out,
            LIMIT,
        )
        .expect("a layer with no compression ignores both modules");
        assert!(state.compressed.is_empty() && state.indexer.is_empty());

        let ratio_one = ModelConfig {
            compress_ratios: vec![1, 1, 1],
            kv_source_layers: vec![0],
            index_source_layers: vec![0],
            candidate_source_layer: 0,
            ..ModelConfig::tiny()
        };
        let hand = Hand::for_config(&ratio_one);
        // At ratio one the plan holds no gate at all - model.py:461-462 short-circuits it - so a
        // successful pass is evidence the pooled path was not taken.
        assert!(
            hand.spec("layers.0.attn.compressor.wgate.weight").is_err(),
            "the plan holds no compressor gate at ratio one"
        );
        let mut state = AttnState::for_config(&ratio_one);
        let mut out = vec![0.0; 4 * dim];
        attention_forward(
            &ratio_one,
            0,
            "layers.0",
            &chunk(&ratio_one, 4, 0),
            0,
            &mut state,
            &hand,
            &mut out,
            LIMIT,
        )
        .expect("a ratio of one has no gate to read");
        assert_eq!(state.compressed_len, 4, "one latent per token at ratio one");
    }

    /// DSpark attends the main rows through its own ring, writes only its own state, and never lets a
    /// block see the row the next step will write (model.py:1032-1074).
    #[test]
    fn dspark_uses_its_own_ring_and_cannot_see_the_next_row() {
        let config = ModelConfig {
            compress_ratios: vec![2, 2, 0],
            ..ModelConfig::tiny()
        };
        let hand = Hand::for_config(&config);
        let dim = config.dim;
        let head_dim = config.head_dim;
        let main = chunk(&config, 2, 0);
        let block = chunk(&config, 3, 5);
        let start_pos = 4usize;
        let draft_start = start_pos + 2;

        let mut stage = AttnState::for_config(&config);
        let mut out = vec![0.0; 3 * dim];
        dspark_attention_forward(
            &config, 0, "mtp.0", &block, &main, start_pos, &mut stage, &hand, &mut out, LIMIT,
        )
        .expect("a draft block");
        assert!(stage.window.row(start_pos).is_some());
        assert!(stage.window.row(start_pos + 1).is_some());
        assert_eq!(
            stage.window.len(),
            2,
            "the ring holds the main rows and nothing else"
        );
        assert!(
            stage.window.row(draft_start).is_none(),
            "the block's own rows must not enter the ring, or a later step would see them"
        );
        assert!(out.iter().all(|value| value.is_finite()));

        // The backbone's states are not touched, and a row already sitting in the slot the next step
        // would use is not in this block's ids.
        let backbone = AttnState::for_config(&config);
        assert!(backbone.window.is_empty() && backbone.compressed_len == 0);
        let mut future = AttnState::for_config(&config);
        future
            .window
            .write(draft_start, &vec![9.0; head_dim])
            .expect("a head vector");
        let mut other = vec![0.0; 3 * dim];
        dspark_attention_forward(
            &config,
            0,
            "mtp.0",
            &block,
            &main,
            start_pos,
            &mut future,
            &hand,
            &mut other,
            LIMIT,
        )
        .expect("a draft block");
        assert_eq!(
            out, other,
            "a query cannot see the row at its own position or later"
        );
    }

    /// At the start of a sequence a DSpark stage only seeds its ring and returns its input unchanged
    /// (model.py:1044-1052).
    #[test]
    fn dspark_seeding_returns_the_input_and_fills_the_ring() {
        let config = ModelConfig {
            compress_ratios: vec![2, 2, 0],
            ..ModelConfig::tiny()
        };
        let hand = Hand::for_config(&config);
        let dim = config.dim;
        let main = chunk(&config, 3, 0);
        let block = chunk(&config, 3, 5);
        let mut stage = AttnState::for_config(&config);
        let mut out = vec![0.0; 3 * dim];
        dspark_attention_forward(
            &config, 0, "mtp.0", &block, &main, 0, &mut stage, &hand, &mut out, LIMIT,
        )
        .expect("a seeding pass");
        assert_eq!(out, block, "the reference returns x at start_pos 0");
        assert_eq!(stage.window.len(), 3);
        assert_eq!(stage.window.last_position(), Some(2));
    }

    /// The local round trip is the crate's own activation quantiser for the block sizes the released
    /// geometry uses (kernel.py:98-181).
    #[test]
    fn the_round_trip_is_the_crates_quantiser() {
        let values: Vec<f32> = (0..96).map(|i| ((i % 17) as f32 - 8.0) * 0.03125).collect();
        let mut mine = values.clone();
        round_trip_rows(&mut mine, 96, 32, Element::Fp8, ScaleRule::PowerOfTwo);
        let expected = quant::quantize_activation_fp8(&values)
            .expect("96 divides by 32")
            .dequantize();
        assert_eq!(mine, expected);

        let mut mine = values.clone();
        round_trip_rows(&mut mine, 96, 32, Element::Fp4, ScaleRule::PowerOfTwo);
        let expected = quant::quantize_activation_fp4(&values)
            .expect("96 divides by 32")
            .dequantize();
        assert_eq!(mine, expected);
    }

    /// A pass is deterministic and its output is finite, and the shapes it cannot explain are refused
    /// rather than indexed past their end.
    #[test]
    fn a_pass_is_deterministic_and_refuses_bad_shapes() {
        reset_shared_attention();
        let config = ModelConfig::tiny();
        let hand = Hand::for_config(&config);
        let dim = config.dim;
        let x = chunk(&config, 5, 0);
        let mut first = AttnState::for_config(&config);
        let mut left = vec![0.0; 5 * dim];
        attention_forward(
            &config, 0, "layers.0", &x, 0, &mut first, &hand, &mut left, LIMIT,
        )
        .expect("a pass");
        reset_shared_attention();
        let mut second = AttnState::for_config(&config);
        let mut right = vec![0.0; 5 * dim];
        attention_forward(
            &config,
            0,
            "layers.0",
            &x,
            0,
            &mut second,
            &hand,
            &mut right,
            LIMIT,
        )
        .expect("a pass");
        assert_eq!(left, right);
        assert!(left.iter().all(|value| value.is_finite()));
        assert_eq!(first.window.len(), config.window_size.min(5));
        assert_eq!(first.window.last_position(), Some(4));

        let mut state = AttnState::for_config(&config);
        let mut short = vec![0.0; 5 * dim];
        let error = attention_forward(
            &config,
            0,
            "layers.0",
            &x[..dim - 1],
            0,
            &mut state,
            &hand,
            &mut short,
            LIMIT,
        )
        .unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        let error = attention_forward(
            &config,
            0,
            "layers.0",
            &x,
            0,
            &mut state,
            &hand,
            &mut short[..dim],
            LIMIT,
        )
        .unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");

        // DSpark's own refusals: a block that is not whole rows, and an output that cannot hold it.
        let mut stage = AttnState::for_config(&config);
        let error = dspark_attention_forward(
            &config,
            0,
            "mtp.0",
            &x[..dim - 1],
            &[],
            4,
            &mut stage,
            &hand,
            &mut short[..],
            LIMIT,
        )
        .unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        let error = dspark_attention_forward(
            &config,
            0,
            "mtp.0",
            &x,
            &[],
            4,
            &mut stage,
            &hand,
            &mut short[..dim],
            LIMIT,
        )
        .unwrap_err();
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
    }
    /// The caller's element limit reaches every tensor this file materialises, through every entry
    /// point, and the refusal names the caller's limit rather than a constant of this module.
    #[test]
    fn the_callers_element_limit_caps_every_tensor_the_pass_reads() {
        let config = ModelConfig::tiny();
        let hand = Hand::for_config(&config);
        let rows = 2usize;
        // `wq_a` is rank x dim elements, the widest tensor a tiny pass reads.
        let widest = config.q_lora_rank * config.dim;

        let mut state = AttnState::for_config(&config);
        let mut out = vec![0.0; rows * config.dim];
        let error = attention_forward(
            &config,
            0,
            "layers.0",
            &chunk(&config, rows, 0),
            0,
            &mut state,
            &hand,
            &mut out,
            widest - 1,
        )
        .unwrap_err();
        match error {
            EngineError::TooLarge {
                name,
                elements,
                limit,
            } => {
                assert_eq!(name, "layers.0.attn.wq_a.weight");
                assert_eq!(elements, widest);
                assert_eq!(limit, widest - 1, "the caller's limit is the one enforced");
            }
            other => panic!("a tensor above the caller's limit must be refused: {other}"),
        }

        // Under a limit that fits every tensor the same pass runs and stays finite.
        let mut state = AttnState::for_config(&config);
        attention_forward(
            &config,
            0,
            "layers.0",
            &chunk(&config, rows, 0),
            0,
            &mut state,
            &hand,
            &mut out,
            LIMIT,
        )
        .expect("a pass under the caller's limit");
        assert!(out.iter().all(|value| value.is_finite()));

        // The indexer and the DSpark stage decode under the caller's limit too.
        let mut indices = Vec::new();
        let error = compressed_indices(
            &state,
            &config,
            0,
            &chunk(&config, rows, 3),
            &hand,
            "layers.0",
            &mut indices,
            8,
        )
        .unwrap_err();
        assert!(
            matches!(error, EngineError::TooLarge { limit: 8, .. }),
            "{error}"
        );
        let mut draft_state = AttnState::for_config(&config);
        let mut draft = vec![0.0; rows * config.dim];
        let error = dspark_attention_forward(
            &config,
            0,
            "mtp.0",
            &chunk(&config, rows, 0),
            &[],
            1,
            &mut draft_state,
            &hand,
            &mut draft,
            8,
        )
        .unwrap_err();
        assert!(
            matches!(error, EngineError::TooLarge { limit: 8, .. }),
            "{error}"
        );
        assert!(
            draft_state.window.is_empty(),
            "a refused pass writes no ring rows"
        );
    }
}
