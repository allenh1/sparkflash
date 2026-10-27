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
use super::weight_view::WeightView;
use crate::config::ModelConfig;
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::parallel::par_aligned_mut;
use crate::parallel_tp::BitGatherHandle;
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

/// The compressed KV rows currently published in the shared channel: the owning layer, the row
/// count and the rows themselves, `head_dim` values per row.
///
/// Diagnostic accessor for tests that assert the channel holds one owner's **whole** cache: the
/// reference's channel points at the last owner's cache (model.py:748, 1173), so a slot holding
/// only the rows the most recent call appended would be a mixture with holes.
#[must_use]
pub fn published_compressed_rows() -> (Option<usize>, usize, Vec<f32>) {
    with_shared(|shared| {
        (
            shared.compressed_owner,
            shared.compressed_rows,
            shared.compressed.clone(),
        )
    })
}

/// The index keys currently published in the shared channel, in the same shape as
/// [published_compressed_rows]: the owning layer, the row count and the keys,
/// `index_head_dim` values per row.
#[must_use]
pub fn published_index_keys() -> (Option<usize>, usize, Vec<f32>) {
    with_shared(|shared| {
        (
            shared.index_owner,
            shared.index_rows,
            shared.index_keys.clone(),
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
/// The frequencies stay f32 because the reference's are an f32 tensor. The YaRN band edges are
/// computed in f64, because the reference computes them with Python floats; everything they
/// feed afterwards is f32 tensor arithmetic, so the ramp and the blend below are f32 as well.
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
        // The ramp is f32 tensor arithmetic in the reference (model.py:384): a f32 arange minus
        // the integer band edge, divided by the integer span (or by 1e-3 in the degenerate band),
        // then clamped. Computing it in f64 and narrowing changes one of the 32 released
        // frequencies by an ulp, which moves about 3% of the released table's entries.
        let span: f32 = if high > low {
            (high - low) as f32
        } else {
            1e-3
        };
        let factor = config.rope_factor as f32;
        for (k, freq) in freqs.iter_mut().enumerate() {
            let ramp = ((k as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            // freqs / factor * (1 - smooth) + freqs * smooth (model.py:386). The reference
            // re-derives (1 - smooth) from the rounded smooth instead of reusing ramp, so this
            // does too: the two are not always the same f32 value.
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
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
#[derive(Clone)]
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

/// The compressor seam's constant weights, decoded once per layer.
///
/// A KV-source layer's seam step runs on every position it serves, and its compressor matrices never
/// change, so reading and decoding them per position was pure waste: the released geometry's pair is
/// two `[head_dim x dim]` tensors (2.6M values each) and the seam's own cached source hands out a
/// fresh copy of the bytes on every call. The decoded values are exactly what the materialised view
/// path would have produced - `quant::decode` widens every stored dtype exactly - so the matvec over
/// them is bit-identical to the fused view's.
#[derive(Clone)]
struct CompressorWeights {
    /// `compressor.norm.weight`: the f32 RMS norm applied after the projection.
    norm: Vec<f32>,
    /// `compressor.wkv.weight`, `[head_dim x dim]` row-major f32.
    wkv: Vec<f32>,
    /// `compressor.wgate.weight`, `[head_dim x dim]` row-major; absent at ratio 1, which has no
    /// gate.
    wgate: Option<Vec<f32>>,
}

/// The attention state one layer owns: its window ring, its compressed KV and the indexer's keys.
///
/// The compressor's pool width is the layer's `compress_ratio`, and this type does not know which
/// layer it belongs to, so those buffers are sized on the first call that needs them. The reference
/// instead preallocates every one of them for the whole model (model.py:452-453, 664-679); the
/// caches here start empty and grow with the sequence, so a test config needs no `max_seq_len`.
#[derive(Clone)]
pub struct AttnState {
    /// The compressor's decoded weights, filled on this layer's first seam step and reused for the
    /// run (see [CompressorWeights]). Sequence state [AttnState::reset] drops; this does not.
    compressor: Option<Box<CompressorWeights>>,
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
            compressor: None,
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
        // The compressor weights are the layer's, not the sequence's: they stay across a reset.
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

/// The KV source a consumer layer reads: the nearest preceding layer that compresses its own KV.
///
/// The reference resolves the same relation statically, once, and asserts the source precedes its
/// consumer (cache_engine.py:160-172); a consumer never reads a later source or its own cache.
fn planned_kv_source(config: &ModelConfig, layer: usize) -> Option<usize> {
    config
        .kv_source_layers
        .iter()
        .rev()
        .copied()
        .find(|&source| source < layer)
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

/// Narrows a vector in place to bf16, the dtype of the reference module output that its
/// dynamic quantisers and its bf16 GEMMs consume (F5).
///
/// The reference's modules are bf16 in and bf16 out: RMSNorm casts its result back to the
/// input dtype (model.py:281-293) and the fp8 Linears hand back bf16 (model.py:187-205). A
/// host stage that keeps the value in f32 carries bits native never had, and the quantiser that
/// consumes it can pick a different code; rounding at the seam is the native arithmetic.
fn narrow_bf16(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = quant::bf16_round(*value);
    }
}

/// Narrows a host tensor to the bf16 value the reference module would have produced and then runs
/// the dynamic quantiser over it (F5).
///
/// The reference quantisers (act_quant over fp8, fp4_act_quant) are called on bf16 tensors in
/// every path: the window KV, the compressed latent, the indexer keys and q. A host stage that
/// quantises f32 instead can pick a different code, so the two steps live together here.
fn narrow_and_round_trip(
    values: &mut [f32],
    width: usize,
    block: usize,
    element: Element,
    rule: ScaleRule,
) {
    narrow_bf16(values);
    round_trip_rows(values, width, block, element, rule);
}

/// Reads a block-scaled Linear weight in the representation its source stores it in.
///
/// A released checkpoint serves F8-E4M3 codes plus E8M0 Block32 scales, so this returns
/// [WeightView::Coded] and the caller's matmul runs the fused kernel with no f32 weight buffer.
/// An f32 source (every fixture) returns [WeightView::F32], the exact old materialised path. The
/// dtype branch is once per weight read, never per element.
///
/// # Errors
///
/// As [weight], plus [EngineError::Shape] when the weight or grid does not match `rows` by `cols`.
fn dense_view<'a>(
    src: Reads<'a>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightView<'a>, EngineError> {
    WeightView::read_dense(src.source, name, rows, cols, ScaleGrid::Block32, src.limit)
}

/// Reads a weight whose scale grid may be absent: `wo_a` is fp8 in the file and bf16 after
/// convert.py (model.py:784-787, tensor.rs's note on its grid).
///
/// # Errors
///
/// As [weight], plus [EngineError::Shape] when a grid that is present does not match.
fn scaled_or_plain_view<'a>(
    src: Reads<'a>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightView<'a>, EngineError> {
    WeightView::read_optional_scale(src.source, name, rows, cols, ScaleGrid::Block32, src.limit)
}

/// Reads a plain BF16 (or f32) Linear weight, e.g. the compressor, indexer or head projections.
///
/// # Errors
///
/// As [weight], plus [EngineError::Shape] when the stored element count does not match `rows` by
/// `cols`.
fn plain_view<'a>(
    src: Reads<'a>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightView<'a>, EngineError> {
    WeightView::read_plain(src.source, name, rows, cols, src.limit)
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
    let attention_head = |head: usize, output: &mut [f32]| {
        let query = &q[head * head_dim..(head + 1) * head_dim];
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
            // kernel.py:339-341 rounds every softmax probability to bf16 before the P*V
            // product (T.copy(acc_s, acc_s_cast) then T.gemm(acc_s_cast, kv_shared, acc_o)),
            // and the gemm's operand is that bf16 tensor. The denominator keeps the f32
            // probability (kernel.py:342-353), so only the P*V product sees the rounded value.
            let probability = quant::bf16_round(weight);
            for (slot, value) in output.iter_mut().zip(row.iter()) {
                *slot += probability * value;
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
    };
    if heads >= 4 {
        // One lane per head: each head is an independent dot/softmax/P*V sequence, and no lane
        // touches another head's scores or output. Per-head arithmetic and head order are unchanged.
        par_aligned_mut(&mut out[..heads * head_dim], head_dim, 1, |first, chunk| {
            for (index, lane) in chunk.chunks_mut(head_dim).enumerate() {
                attention_head(first + index, lane);
            }
        });
    } else {
        for head in 0..heads {
            attention_head(head, &mut out[head * head_dim..(head + 1) * head_dim]);
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

/// Checks that a device arm's rows have the shape the host rule would have produced.
///
/// The seam's device arms return plain f32 rows so the host code after the projection cannot tell
/// them apart from the host rule's output. A row count that does not match is a broken arm, not a
/// fallback: the seam has already decided the geometry, so an arm that cannot fill it returns
/// None instead, and this check is for the arm that filled it wrongly.
fn device_rows(rows: Vec<f32>, expected: usize, what: &str) -> Result<Vec<f32>, EngineError> {
    if rows.len() != expected {
        return Err(EngineError::Shape {
            what: what.to_owned(),
            detail: format!(
                "the device arm returned {} values, the host rule produces {expected}",
                rows.len()
            ),
        });
    }
    Ok(rows)
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
#[allow(clippy::too_many_arguments)] // the seam step's inputs plus the arm it may delegate to
fn compressor_step(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: Reads<'_>,
    mut backend: Option<&mut dyn AttentionProjectionBackend>,
) -> Result<(), EngineError> {
    let AttnState {
        compressor,
        latent: latent_out,
        latent_rows,
        kv_state,
        score_state,
        ..
    } = state;
    latent_out.clear();
    *latent_rows = 0;
    let ratio = layer_ratio(config, layer);
    if ratio == 0 || !is_kv_source(config, layer) {
        return Ok(());
    }
    let dim = config.dim;
    let head_dim = config.head_dim;
    let seqlen = rows_of(x, dim, "compressor input")?;
    let end_pos = start_pos + seqlen;
    // One decode of this layer's compressor weights per run; the seam runs them on every position.
    if compressor.is_none() {
        *compressor = Some(Box::new(read_compressor_weights(
            config, prefix, ratio, src,
        )?));
    }
    let cached = compressor
        .as_deref()
        .expect("the compressor weights were just read");
    if ratio == 1 {
        // model.py:461-462: one token per group, so a plain projection with no gate and no pooling.
        // The device arm, when it serves this layer, hands back the projection's f32 accumulators;
        // only the host rule is replaced, the narrowing and the norm below still run.
        let device = match backend.as_mut() {
            Some(arm) => arm.compressor_wkv(config, layer, prefix, x, src.source, src.limit)?,
            None => None,
        };
        let mut latent = match device {
            Some(rows) => device_rows(rows, seqlen * head_dim, "compressor wkv projection")?,
            None => {
                let mut host = vec![0.0; seqlen * head_dim];
                linalg::matmul_hot_rows(&mut host, x, &cached.wkv, head_dim, dim, seqlen)?;
                host
            }
        };
        // At ratio 1 the reference keeps bf16 wkv weights and returns norm(wkv(x)) as a bf16
        // tensor (model.py:446-449, 461-463).
        narrow_bf16(&mut latent);
        linalg::rms_norm(&mut latent, &cached.norm, head_dim, config.norm_eps as f32)?;
        narrow_bf16(&mut latent);
        *latent_out = latent;
        *latent_rows = seqlen;
        return Ok(());
    }
    // Above ratio 1 the reference keeps a partial group between steps; its buffers are the layer's
    // ratio, so they are sized on first use here.
    if kv_state.len() != ratio * head_dim {
        *kv_state = vec![0.0; ratio * head_dim];
        *score_state = vec![f32::NEG_INFINITY; ratio * head_dim];
    }
    let wgate = cached.wgate.as_deref().ok_or_else(|| EngineError::Shape {
        what: "compressor weights".to_owned(),
        detail: format!("layer {layer} pools at ratio {ratio} but its gate matrix was not read"),
    })?;
    // Above ratio 1 both projections are replaced together or not at all: a group's latent is
    // pooled from the pair, so mixing one arm's rows with the other host rule's rows is not a state
    // the seam has. Each call is a no-op on None, which leaves that projection's host rule to run.
    let device_kv = match backend.as_mut() {
        Some(arm) => arm.compressor_wkv(config, layer, prefix, x, src.source, src.limit)?,
        None => None,
    };
    let device_score = match backend.as_mut() {
        Some(arm) => arm.compressor_wgate(config, layer, prefix, x, src.source, src.limit)?,
        None => None,
    };
    let kv = match device_kv {
        Some(rows) => device_rows(rows, seqlen * head_dim, "compressor wkv projection")?,
        None => {
            let mut host = vec![0.0; seqlen * head_dim];
            linalg::matmul_hot_rows(&mut host, x, &cached.wkv, head_dim, dim, seqlen)?;
            host
        }
    };
    let score = match device_score {
        Some(rows) => device_rows(rows, seqlen * head_dim, "compressor wgate projection")?,
        None => {
            let mut host = vec![0.0; seqlen * head_dim];
            linalg::matmul_hot_rows(&mut host, x, wgate, head_dim, dim, seqlen)?;
            host
        }
    };
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
            kv_state[slot * head_dim..(slot + 1) * head_dim]
                .copy_from_slice(&kv[row * head_dim..(row + 1) * head_dim]);
            score_state[slot * head_dim..(slot + 1) * head_dim]
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
            kv_state[slot * head_dim..(slot + 1) * head_dim]
                .copy_from_slice(&kv[row * head_dim..(row + 1) * head_dim]);
            score_state[slot * head_dim..(slot + 1) * head_dim]
                .copy_from_slice(&score[row * head_dim..(row + 1) * head_dim]);
            if (position + 1).is_multiple_of(ratio) {
                let dest = &mut latent[written * head_dim..(written + 1) * head_dim];
                pool_group(
                    dest,
                    kv_state,
                    score_state,
                    0,
                    ratio,
                    head_dim,
                    &mut weights,
                );
                written += 1;
            }
        }
    }
    // Above ratio 1 the pooled value is narrowed to bf16 before the norm and the norm casts
    // its result back to bf16: norm(kv.to(dtype)) (model.py:484-486).
    narrow_bf16(&mut latent);
    linalg::rms_norm(&mut latent, &cached.norm, head_dim, config.norm_eps as f32)?;
    narrow_bf16(&mut latent);
    *latent_out = latent;
    *latent_rows = rows;
    Ok(())
}

/// Reads and decodes one layer's compressor weights for [AttnState]'s cache.
///
/// The values are the source's own: `quant::decode` handles every dtype the plan stores these
/// tensors in, so an F32 tensor is a straight read and a BF16 one widens exactly, which is what the
/// fused view's matmul does with the same bytes.
///
/// # Errors
///
/// [EngineError] from the weight read, the element limit, or a tensor whose element count is not
/// [ModelConfig::head_dim] rows of [ModelConfig::dim].
fn read_compressor_weights(
    config: &ModelConfig,
    prefix: &str,
    ratio: usize,
    src: Reads<'_>,
) -> Result<CompressorWeights, EngineError> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let norm = weight(src, &format!("{prefix}.attn.compressor.norm.weight"))?;
    let wkv = decoded_compressor_weight(
        src,
        &format!("{prefix}.attn.compressor.wkv.weight"),
        head_dim,
        dim,
    )?;
    let wgate = if ratio == 1 {
        None
    } else {
        Some(decoded_compressor_weight(
            src,
            &format!("{prefix}.attn.compressor.wgate.weight"),
            head_dim,
            dim,
        )?)
    };
    Ok(CompressorWeights { norm, wkv, wgate })
}

/// One `[rows x cols]` compressor weight, decoded to the f32 values the matvec reads.
///
/// # Errors
///
/// [EngineError::Shape] when the source decodes a different element count, and the read errors of
/// [linalg::decode_by_name].
fn decoded_compressor_weight(
    src: Reads<'_>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, EngineError> {
    let expected = rows.checked_mul(cols).ok_or_else(|| EngineError::Shape {
        what: "compressor weight".to_owned(),
        detail: format!("{name} is {rows}x{cols}, which overflows"),
    })?;
    let values = linalg::decode_by_name(src.source, name, src.limit)?;
    if values.len() != expected {
        return Err(EngineError::Shape {
            what: "compressor weight".to_owned(),
            detail: format!(
                "{name} is {rows}x{cols}, so it is {expected} values; the source decoded {}",
                values.len()
            ),
        });
    }
    Ok(values)
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

/// The activation the reference's indexer wq_b receives (model.py:519, 187-199).
///
/// The q_norm output is a bf16 tensor, and the ColumnParallelLinear at the model dtype
/// quantises it to E4M3 over 32-value blocks with power-of-two scales before the fp8 GEMM.
/// This is the one implementation of that step: the host path and the device arm's twin both
/// reach it through [indexer_wq_b_native]. It is also public so a device arm stages exactly this
/// row - the same bf16 narrowing and the same fp8 round trip - instead of re-deriving it, which is
/// what keeps the device arm's f32 activation input the host rule's input.
pub fn indexer_activation_native(latent: &[f32], width: usize) -> Vec<f32> {
    let mut values: Vec<f32> = latent
        .iter()
        .map(|value| quant::bf16_round(*value))
        .collect();
    round_trip_rows(
        &mut values,
        width,
        INDEX_QUANT_BLOCK,
        Element::Fp8,
        ScaleRule::PowerOfTwo,
    );
    values
}

/// The indexer query projection's native rule, as a host function (model.py:519, 550).
///
/// The reference's `wq_b` is a `ColumnParallelLinear` at the default dtype: the bf16 `q_norm`
/// output is narrowed, quantised to E4M3 over 32-value blocks with power-of-two (ue8m0) scales,
/// multiplied against the same-quantised weight, accumulated in f32, and narrowed to bf16.
/// `weight` is the dequantised fp8 weight, row-major `[cols][inner]`. The host path calls this
/// and the device arm's twin test compares against it, so the two arms cannot drift into
/// different arithmetic.
///
/// # Errors
///
/// [EngineError::Shape] when `latent` is shorter than `rows x inner` or `weight` is shorter
/// than `cols x inner`.
pub fn indexer_wq_b_native(
    latent: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    inner: usize,
) -> Result<Vec<f32>, EngineError> {
    let latent_values = rows.saturating_mul(inner);
    let weight_values = cols.saturating_mul(inner);
    if latent.len() < latent_values || weight.len() < weight_values {
        return Err(shape(
            "indexer wq_b",
            format!(
                "{rows} rows of {inner} values need {latent_values} latent and {cols} output columns need {weight_values} weight values, got {} and {}",
                latent.len(),
                weight.len()
            ),
        ));
    }
    let activation = indexer_activation_native(&latent[..latent_values], inner);
    let mut out = vec![0.0; rows * cols];
    linalg::matmul(&mut out, &activation, weight, cols, inner, rows)?;
    for value in out.iter_mut() {
        *value = quant::bf16_round(*value);
    }
    Ok(out)
}

/// How many rows of the indexer query projection fit one bf16 output staging buffer.
///
/// The device arm stages `rows x cols` bf16 values through the shared attention output buffer,
/// so a prefill chunk is projected in row chunks. GEMM rows are independent, which makes the
/// chunk size an allocation detail rather than an arithmetic one; at least one row is returned,
/// and a width that cannot fit any row returns 1 so the caller gets a typed refusal rather than
/// a division by zero.
#[must_use]
pub fn indexer_chunk_rows(cols: usize, max_out_bytes: usize) -> usize {
    let bytes_per_row = cols.saturating_mul(2).max(2);
    (max_out_bytes / bytes_per_row).max(1)
}

/// Projects an indexer query chunk in row chunks that fit `max_out_bytes`.
///
/// `project(start, count)` returns a chunk's `count x cols` bf16-representable values. This is
/// the one chunking implementation: the device arm passes a closure over its kernel launch and
/// the host test passes one over [indexer_wq_b_native], so the test proves the arithmetic is
/// transparent to the chunking instead of asserting it.
///
/// # Errors
///
/// [EngineError::Shape] when a chunk returns the wrong number of values, plus whatever
/// `project` returns.
pub fn indexer_wq_b_chunked<F>(
    rows: usize,
    cols: usize,
    max_out_bytes: usize,
    mut project: F,
) -> Result<Vec<f32>, EngineError>
where
    F: FnMut(usize, usize) -> Result<Vec<f32>, EngineError>,
{
    let chunk = indexer_chunk_rows(cols, max_out_bytes);
    let mut out = Vec::with_capacity(rows.saturating_mul(cols));
    let mut start = 0;
    while start < rows {
        let count = chunk.min(rows - start);
        let values = project(start, count)?;
        if values.len() != count * cols {
            return Err(shape(
                "indexer wq_b chunk",
                format!(
                    "rows {start}..{} of {cols} columns need {} values, got {}",
                    start + count,
                    count * cols,
                    values.len()
                ),
            ));
        }
        out.extend_from_slice(&values);
        start += count;
    }
    Ok(out)
}

/// The host path of the indexer query projection: read the fp8 weight in its grid and run the
/// native rule.
///
/// This is [indexer_wq_b_native] with the weight kept in its stored representation, so the fp8
/// projection runs the fused kernel instead of a materialised f32 matrix. The activation, the
/// matmul order and the bf16 narrowing are the same calls in the same order.
fn index_source_wq_b(request: &IndexRequest<'_>) -> Result<Vec<f32>, EngineError> {
    let config = request.config;
    let cols = config.index_n_heads.saturating_mul(config.index_head_dim);
    let inner = config.q_lora_rank;
    let latent_values = request.seqlen.saturating_mul(inner);
    if request.q.len() < latent_values {
        return Err(shape(
            "indexer wq_b",
            format!(
                "{} rows of {inner} values need {latent_values} latent, got {}",
                request.seqlen,
                request.q.len()
            ),
        ));
    }
    let wq_b = dense_view(
        request.src,
        &format!("{}.attn.indexer.wq_b", request.prefix),
        cols,
        inner,
    )?;
    let activation = indexer_activation_native(&request.q[..latent_values], inner);
    let mut out = vec![0.0; request.seqlen * cols];
    wq_b.matmul(&mut out, &activation, cols, inner, request.seqlen)?;
    for value in out.iter_mut() {
        *value = quant::bf16_round(*value);
    }
    Ok(out)
}

/// The indexer's scores for one index source, bf16 all the way (model.py:553-557).
///
/// The einsum hands back bf16, the rectified product narrows to bf16, and the per-head sum
/// accumulates in f32 and narrows once.
fn index_scores(
    q_index: &[f32],
    keys: &[f32],
    weights: &[f32],
    seqlen: usize,
    n_pos: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut score = vec![0.0f32; seqlen * n_pos];
    for query in 0..seqlen {
        let query_weights = &weights[query * heads..(query + 1) * heads];
        let q_row = &q_index[query * heads * head_dim..(query + 1) * heads * head_dim];
        let sums = &mut score[query * n_pos..(query + 1) * n_pos];
        for head in 0..heads {
            let q_head = &q_row[head * head_dim..(head + 1) * head_dim];
            let weight = query_weights[head];
            for (position, sum) in sums.iter_mut().enumerate() {
                let key = &keys[position * head_dim..(position + 1) * head_dim];
                let raw = quant::bf16_round(dot(q_head, key));
                if raw > 0.0 {
                    *sum += quant::bf16_round(raw * weight);
                }
            }
        }
        for value in sums.iter_mut() {
            *value = quant::bf16_round(*value);
        }
    }
    score
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
    backend: Option<&mut dyn AttentionProjectionBackend>,
) -> Result<Option<Vec<bool>>, EngineError> {
    let config = request.config;
    let head_dim = config.index_head_dim;
    let heads = config.index_n_heads;
    let seqlen = request.seqlen;
    let n_pos = request.n_pos;
    // The query projection, rotated and fp4 round-tripped exactly like the keys (model.py:550-552).
    // The projection is the one primitive a device arm may replace: both arms run the same native
    // rule (fp8 W8A8 with a bf16 output), one through [indexer_wq_b_native], the other through the
    // pinned dense kernel, so a launch or chunking choice cannot change the arithmetic.
    let mut q_index = if let Some(backend) = backend {
        match backend.indexer_wq_b(
            config,
            request.layer,
            request.prefix,
            request.q,
            request.src.source,
            request.src.limit,
        )? {
            Some(values) => values,
            None => index_source_wq_b(request)?,
        }
    } else {
        index_source_wq_b(request)?
    };
    if q_index.len() != seqlen * heads * head_dim {
        return Err(shape(
            "indexer query projection",
            format!(
                "{seqlen} rows of {heads} index heads x {head_dim} need {} values, got {}",
                seqlen * heads * head_dim,
                q_index.len()
            ),
        ));
    }
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
    // The rope writes back into the bf16 q tensor (model.py:392-409, 550-552).
    narrow_and_round_trip(
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
    let mut score = index_scores(
        &q_index,
        request.keys,
        &request.state.index_weights,
        seqlen,
        n_pos,
        heads,
        head_dim,
    );
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

/// Publishes an owner's cache into the channel.
///
/// The reference's channel holds a pointer at the last owner's cache (model.py:748, 1173), so the
/// slot must hold the owner's **whole** cache: a consumer reading rows `0..len` sees one layer's
/// history, never a mixture of several layers' rows and never holes. Copying the full cache on each
/// publication is the host equivalent of moving that pointer. Writing only the rows the current call
/// appended assumed the owning layer stayed the same across steps, which is false whenever the
/// backbone has more than one KV source (the released model has four: layers 2, 8, 14 and 20).
///
/// This changes results on the shipped decode path, not only on a chunk path: the one-token loop
/// went through this same publication and could hand a consumer holed rows whenever the owner
/// changed inside a step, so a real checkpoint's decode was reading a mixture that the f32 fixtures
/// never exercised (their configs keep one source, or publish all rows at position zero, so no
/// multi-row owner transition left a hole). Treat a change in the released checkpoint's token
/// sequence as this fix working, not as a regression.
fn publish_rows(
    cache: &mut Vec<f32>,
    rows: &mut usize,
    owner: &mut Option<usize>,
    layer: usize,
    owner_cache: &[f32],
    width: usize,
) {
    if width == 0 {
        return;
    }
    let published = owner_cache.len() - owner_cache.len() % width;
    *owner = Some(layer);
    cache.clear();
    cache.extend_from_slice(&owner_cache[..published]);
    *rows = published / width;
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
/// The implementation of [compressed_indices] and [compressed_indices_with_backend].
#[allow(clippy::too_many_arguments)] // the prescribed signature is fixed by the task and clippy's limit is 7; all nine inputs are distinct.
fn compressed_indices_inner(
    state: &AttnState,
    config: &ModelConfig,
    layer: usize,
    q: &[f32],
    src: &dyn WeightSource,
    prefix: &str,
    out: &mut Vec<usize>,
    limit: usize,
    backend: Option<&mut dyn AttentionProjectionBackend>,
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
        if !owns {
            debug_assert_eq!(
                shared.index_owner,
                planned_kv_source(config, layer),
                "layer {layer} reuses index keys only from its planned kv source (cache_engine.py:160-172)"
            );
        }
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
        index_source_indices(&request, out, backend)
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

/// Chooses the compressed positions each query of this chunk attends to, in the shared index
/// space, with [`NO_INDEX`] where a query has nothing (model.py:527-580).
///
/// `q` is this layer's query latent, the output of `wq_a` and `q_norm`: the indexer projects it
/// with its own `wq_b` (model.py:550). Run [attention_forward] on the same state for this chunk
/// first: the published per-query weights and the window length the ids are shifted by are
/// per-step inputs that the fixed signature carries neither of, so the state holds them. A layer
/// that is not an index source reads what its source published instead of computing anything
/// (model.py:725-726). `limit` caps the elements of any single tensor this pass materialises.
///
/// This is the host arm: the indexer's query projection runs the engine's native host rule
/// ([indexer_wq_b_native]). [`compressed_indices_with_backend`] is the same function with a
/// projection backend that may run `wq_b` on a device.
///
/// # Errors
///
/// [EngineError::Shape] when the state has no step recorded, a layer that reads the channel finds
/// nothing published for this chunk, the published rows are too short, or a tensor has the wrong
/// geometry.
#[allow(clippy::too_many_arguments)]
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
    compressed_indices_inner(state, config, layer, q, src, prefix, out, limit, None)
}

/// [`compressed_indices`] with a projection backend.
///
/// The backend's [`AttentionProjectionBackend::indexer_wq_b`] is consulted for the index source
/// layers; `None` from the backend keeps the engine's host native rule, so a backend that does
/// not implement the indexer projection is still the host arm.
///
/// # Errors
///
/// As [`compressed_indices`], plus whatever the backend returns.
#[allow(clippy::too_many_arguments)]
pub fn compressed_indices_with_backend(
    state: &AttnState,
    config: &ModelConfig,
    layer: usize,
    q: &[f32],
    src: &dyn WeightSource,
    prefix: &str,
    out: &mut Vec<usize>,
    limit: usize,
    backend: &mut dyn AttentionProjectionBackend,
) -> Result<(), EngineError> {
    compressed_indices_inner(
        state,
        config,
        layer,
        q,
        src,
        prefix,
        out,
        limit,
        Some(backend),
    )
}

/// One step's compressed-side reference seams, for a loop that runs the attention projections on
/// device and only needs the reference's host-side compressor, indexer and candidate selection.
///
/// The four-rank device loop owns the wq/kv/wo projections on device, but the plan declares the
/// compressor, the indexer and the two-level candidate select as host seams. This function runs
/// exactly those seams for one query, in [attention_forward]'s order, on the caller's input row:
/// the compressor pooling (model.py:446-486), the index keys and per-query weights
/// (model.py:513-557), the candidate select (model.py:561-610) and the fp4 compressed cache
/// (model.py:748-761). It advances the caller's state the same way, so a sequence of calls is the
/// same decode-step sequence the reference sees. What it does not do is recompute the attention
/// projections (q, kv, the output projection) or run sparse attention: the caller already ran
/// those on device from its own weights, and doing them again would only be host arithmetic the
/// caller does not consume.
///
/// The window ring is not written: the caller stages its own device KV rows and the ring is not an
/// input to any of the seams above. The window and compressed ids are reported in the reference's
/// own conventions so the caller can stage the kernel's operands: window_width is the number of
/// attended window rows this query has, and the compressed ids are the shared index-space values
/// (window_rows plus compressed row, with [NO_INDEX] where a query has nothing).
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSeamStep {
    /// The layer's compression ratio (0 for a window-only layer).
    pub ratio: usize,
    /// The window length the compressed ids are shifted by: this step's row count on the first
    /// chunk, the ring width after it (model.py:580, 744).
    pub window_rows: usize,
    /// How many window rows this query attends (ids that stand for a position).
    pub window_width: usize,
    /// The compressed ids this query selects, in the shared index space.
    pub compressed_ids: Vec<usize>,
    /// The compressed rows in row order, head_dim values per row: the caller stages these as the
    /// kernel's fp4 store (already the quantised values the reference stored).
    pub compressed_rows: Vec<f32>,
}

/// One query's compressed-side reference seams; see [AttentionSeamStep].
///
/// The caller must call this once per layer per position, in layer order, because the first
/// index/kv source publishes the channels its consumers read in the same step.
///
/// # Errors
/// [EngineError] from any weight read or shape mismatch, and the same channel refusals
/// [compressed_indices] raises when a consumer runs before its source.
#[allow(clippy::too_many_arguments)] // the query row, its query latent, the step position, the state, the source and the limit are distinct inputs.
pub fn attention_seam_step(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    q_latent: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: &dyn WeightSource,
    limit: usize,
) -> Result<AttentionSeamStep, EngineError> {
    attention_seam_step_with_backend(
        config,
        layer,
        prefix,
        x,
        q_latent,
        start_pos,
        state,
        src,
        limit,
        &mut HostAttentionProjectionBackend,
    )
}

/// [attention_seam_step] with an attention projection backend that may run the indexer's query
/// projection on a device.
///
/// The compressor, the index keys, the per-query weights, the candidate select and the fp4 cache
/// stay host code exactly as in [attention_seam_step]: the only statement the backend is consulted
/// for is the indexer's own wq_b ([AttentionProjectionBackend::indexer_wq_b]), which the host arm
/// runs as [indexer_wq_b_native]. A backend that returns `None` for a layer leaves that layer's
/// projection to the host rule, so a backend that cannot serve a shape falls back to exactly the
/// host arithmetic instead of changing it.
///
/// # Errors
///
/// As [attention_seam_step], plus whatever the backend returns.
#[allow(clippy::too_many_arguments)] // the same distinct inputs as [attention_seam_step], plus the backend.
pub fn attention_seam_step_with_backend(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    q_latent: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: &dyn WeightSource,
    limit: usize,
    backend: &mut dyn AttentionProjectionBackend,
) -> Result<AttentionSeamStep, EngineError> {
    let src = Reads { source: src, limit };
    let dim = config.dim;
    let seqlen = rows_of(x, dim, "attention seam input")?;
    if seqlen != 1 {
        return Err(shape(
            "attention seam step",
            format!(
                "{seqlen} rows: the seam step is one query at a time, which is the decode-step order the four-rank loop runs"
            ),
        ));
    }
    let end_pos = start_pos + seqlen;
    let ratio = layer_ratio(config, layer);
    let compress_len = end_pos.checked_div(ratio).unwrap_or(0);
    state.step = Some((start_pos, seqlen));
    let window_rows = if start_pos == 0 {
        seqlen
    } else {
        config.window_size
    };
    state.window_rows = Some(window_rows);
    let mut compressed_ids = Vec::new();
    if ratio > 0 {
        compressor_step(
            config,
            layer,
            prefix,
            x,
            start_pos,
            state,
            src,
            Some(&mut *backend),
        )?;
        write_index_keys(
            config,
            layer,
            prefix,
            start_pos,
            state,
            src,
            Some(&mut *backend),
        )?;
        if is_index_source(config, layer) && compress_len > 0 {
            state.index_weights =
                index_weights(config, layer, prefix, x, seqlen, src, Some(&mut *backend))?;
        } else {
            state.index_weights.clear();
        }
        compressed_indices_with_backend(
            state,
            config,
            layer,
            q_latent,
            src.source,
            prefix,
            &mut compressed_ids,
            limit,
            backend,
        )?;
        write_compressed_cache(config, layer, start_pos, state)?;
        state.compressed_len = compress_len;
    }
    let ids = window_indices(config, start_pos, seqlen);
    let window_width = ids.iter().filter(|id| **id != NO_INDEX).count();
    let compressed_rows = if ratio > 0 {
        fetch_compressed(config, layer, state, compress_len)?
    } else {
        Vec::new()
    };
    Ok(AttentionSeamStep {
        ratio,
        window_rows,
        window_width,
        compressed_ids,
        compressed_rows,
    })
}

/// The projection primitives of the attention sublayer - the five attention GEMMs plus the
/// indexer's own wq_b - split out so a device arm can replace the matrix multiplies without
/// replacing the host normalisation, RoPE, KV, cache, indexer scoring or softmax seams. The host
/// implementation is the reference arithmetic; [attention_forward_with_backend] uses it for every
/// non-device caller.
pub trait AttentionProjectionBackend {
    /// wq_a over the normalised attention input; returns the pre-norm latent.
    fn wq_a(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError>;
    /// wq_b over the q_norm'd latent; returns the pre-RoPE q rows.
    fn wq_b(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        latent: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError>;
    /// wkv over the normalised attention input; returns the pre-norm KV rows.
    fn wkv(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError>;
    /// The grouped wo_a; returns the concatenated per-group lo vectors.
    fn wo_a(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        wide: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError>;
    /// The wo_b projection of the grouped lo vectors.
    fn wo_b(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        grouped: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError>;

    /// The indexer's own query projection: `wq_b`, an fp8 W8A8 GEMM whose output is bf16
    /// (model.py:519, 550).
    ///
    /// The rows are `[seqlen][index_n_heads * index_head_dim]`, bf16-representable, and exactly
    /// the native rule's output ([indexer_wq_b_native]). `None` leaves the projection to the
    /// engine's host rule; a device arm returns `Some` and the host path is skipped.
    fn indexer_wq_b(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        latent: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let _ = (config, layer, prefix, latent, src, limit);
        Ok(None)
    }

    /// The compressor's `wkv` projection over the layer input: `[seqlen][head_dim]` f32
    /// accumulators, exactly what the host `matmul_hot_rows` call in [compressor_step] produces.
    ///
    /// The rows are not narrowed, normed or pooled here. At ratio 1 the caller narrows and norms
    /// them; above ratio 1 the caller pools them. `None` leaves the projection to the engine's host
    /// rule, which is what the host implementation returns.
    fn compressor_wkv(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let _ = (config, layer, prefix, x, src, limit);
        Ok(None)
    }

    /// The compressor's `wgate` projection over the layer input: `[seqlen][head_dim]` f32
    /// accumulators, the counterpart of [Self::compressor_wkv] that `wgate` runs on
    /// (model.py:455-458).
    ///
    /// `None` leaves the projection to the engine's host rule.
    fn compressor_wgate(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let _ = (config, layer, prefix, x, src, limit);
        Ok(None)
    }

    /// The indexer's key projection `wk` over this step's latent:
    /// `[latent_rows][index_head_dim]` f32 accumulators, before the k_norm, the RoPE and the fp4
    /// round trip.
    ///
    /// `None` leaves the projection to the engine's host rule.
    fn indexer_wk(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        latent: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let _ = (config, layer, prefix, latent, src, limit);
        Ok(None)
    }

    /// The indexer's per-query weights projection `weights_proj` over the layer input, rounded
    /// to bf16 as its bf16 Linear hands it on and BEFORE the index factor multiplies it
    /// (model.py:513, 555): `[seqlen][index_n_heads]` f32 values, every one bf16-representable.
    ///
    /// `None` leaves the projection to the engine's host rule.
    #[allow(clippy::too_many_arguments)] // the seam's inputs plus the layer the arm stages by
    fn indexer_weights_proj(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        seqlen: usize,
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let _ = (config, layer, prefix, x, seqlen, src, limit);
        Ok(None)
    }

    /// The CSA-union attention body over the already-built q, window KV, compressed rows and
    /// per-query candidate ids. Returns `true` when the backend wrote `out`; the default `false`
    /// keeps the engine's reference sparse-attention loop. A backend may support only a subset of
    /// shapes and return `false` for the rest, which is why this is a capability answer and not a
    /// `Result<()>`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        layer: usize,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        combined: &[usize],
        heads: usize,
        head_dim: usize,
        seqlen: usize,
        window_width: usize,
        compressed_stride: usize,
        scale: f32,
        start_pos: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let _ = (
            layer,
            q,
            kv,
            sink,
            combined,
            heads,
            head_dim,
            seqlen,
            window_width,
            compressed_stride,
            scale,
            start_pos,
            out,
        );
        Ok(false)
    }
}

/// The host reference backend. It is the old arithmetic unchanged; the device arm is a different
/// implementation of the same primitive contracts.
pub struct HostAttentionProjectionBackend;

impl AttentionProjectionBackend for HostAttentionProjectionBackend {
    fn wq_a(
        &mut self,
        config: &ModelConfig,
        _layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let src = Reads { source, limit };
        let dim = config.dim;
        let rank = config.q_lora_rank;
        let seqlen = rows_of(x, dim, "attention input")?;
        let wq_a = dense_view(src, &format!("{prefix}.attn.wq_a"), rank, dim)?;
        let mut latent = vec![0.0; seqlen * rank];
        wq_a.matmul(&mut latent, x, rank, dim, seqlen)?;
        Ok(latent)
    }

    fn wq_b(
        &mut self,
        config: &ModelConfig,
        _layer: usize,
        prefix: &str,
        latent: &[f32],
        source: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let src = Reads { source, limit };
        let rank = config.q_lora_rank;
        let cols = config.n_heads * config.head_dim;
        let seqlen = rows_of(latent, rank, "q latent")?;
        let wq_b = dense_view(src, &format!("{prefix}.attn.wq_b"), cols, rank)?;
        let mut q = vec![0.0; seqlen * cols];
        wq_b.matmul(&mut q, latent, cols, rank, seqlen)?;
        Ok(q)
    }

    fn wkv(
        &mut self,
        config: &ModelConfig,
        _layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let src = Reads { source, limit };
        let dim = config.dim;
        let head_dim = config.head_dim;
        let seqlen = rows_of(x, dim, "attention input")?;
        let wkv = dense_view(src, &format!("{prefix}.attn.wkv"), head_dim, dim)?;
        let mut kv = vec![0.0; seqlen * head_dim];
        wkv.matmul(&mut kv, x, head_dim, dim, seqlen)?;
        Ok(kv)
    }

    fn wo_a(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        wide: &[f32],
        source: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let src = Reads { source, limit };
        let head_dim = config.head_dim;
        let heads = config.n_heads;
        let groups = config.o_groups;
        let per_group = heads / groups;
        let group_in = per_group * head_dim;
        let wide_row = heads * head_dim;
        let seqlen = rows_of(wide, wide_row, "attention wide")?;
        let grouped_width = groups * config.o_lora_rank;
        let wo_a_name = format!("{prefix}.attn.wo_a");
        let wo_a = scaled_or_plain_view(src, &wo_a_name, grouped_width, group_in)?;
        // A grid whose row blocks do not start on a 32-row boundary (the tiny configuration's
        // o_lora_rank 8), or a packed grid whose block starts mid-byte, cannot slice its stored
        // representation by row block. Those geometries keep the old materialised path, which is
        // exact for any block size; the released checkpoint's o_lora_rank is a multiple of 32.
        let materialised = if wo_a.row_blocks_supported(config.o_lora_rank, group_in) {
            None
        } else {
            let scale_name = format!("{wo_a_name}.scale");
            let mut values = weight(src, &format!("{wo_a_name}.weight"))?;
            if src.source.spec(&scale_name).is_ok() {
                let scales = weight(src, &scale_name)?;
                linalg::apply_scales(
                    &mut values,
                    grouped_width,
                    group_in,
                    &scales,
                    ScaleGrid::Block32,
                )?;
            }
            Some(values)
        };
        let mut grouped = vec![0.0; seqlen * grouped_width];
        for query in 0..seqlen {
            let wide_row_values = &wide[query * wide_row..][..wide_row];
            let dest = &mut grouped[query * grouped_width..][..grouped_width];
            if let Some(values) = &materialised {
                for group in 0..groups {
                    let source = &wide_row_values[group * group_in..][..group_in];
                    let row_start = group * config.o_lora_rank;
                    let block = &values[row_start * group_in..][..config.o_lora_rank * group_in];
                    let block_out = &mut dest[row_start..][..config.o_lora_rank];
                    linalg::matmul(block_out, source, block, config.o_lora_rank, group_in, 1)?;
                }
            } else if wo_a.supports_grouped() {
                wo_a.matmul_grouped(dest, wide_row_values, groups, config.o_lora_rank, group_in)?;
            } else {
                for group in 0..groups {
                    let source = &wide_row_values[group * group_in..][..group_in];
                    let row_start = group * config.o_lora_rank;
                    let block_out = &mut dest[row_start..][..config.o_lora_rank];
                    wo_a.matmul_rows(
                        block_out,
                        source,
                        config.o_lora_rank,
                        group_in,
                        1,
                        row_start,
                    )?;
                }
            }
        }
        Ok(grouped)
    }

    fn wo_b(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        grouped: &[f32],
        source: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let src = Reads { source, limit };
        let grouped_width = config.o_groups * config.o_lora_rank;
        let seqlen = rows_of(grouped, grouped_width, "grouped lo")?;
        let wo_b = dense_view(
            src,
            &format!("{prefix}.attn.wo_b"),
            config.dim,
            grouped_width,
        )?;
        let mut out = vec![0.0; seqlen * config.dim];
        wo_b.matmul(&mut out, grouped, config.dim, grouped_width, seqlen)?;
        Ok(out)
    }
}

/// One rank's slice of the batch-18 Option-A attention shard (design brief section 3, steps 1-5).
///
/// The split is by OUTPUT rows: rank `r` holds heads `r * heads .. (r + 1) * heads` of `wq_b`, the
/// matching o-groups of `wo_a`, and computes its own head block of the sparse body. Each output
/// element's dot product is therefore still one ascending f32 accumulation over the full input
/// (the replicated q_a latent and the full KV rows), so nothing is re-associated: the rank-local
/// results are exactly the replicated results, and the only cross-rank operation is the exact
/// o_lora copy in [output_projection_with].
pub struct AttentionShard<'a> {
    /// The exact f32 gather the o_lora copy rides.
    pub gather: &'a BitGatherHandle,
    /// This rank's index.
    pub rank: usize,
    /// The number of heads this rank owns (contiguous, a whole number of o-groups).
    pub heads: usize,
    /// The number of o-groups this rank owns: `heads / (n_heads / o_groups)`.
    pub groups: usize,
}

/// The local head and o-group counts when this rank's source serves a `wq_b` output-row shard.
///
/// A whole-tensor source answers `None`, which is the replicated layout and keeps every old call
/// path unchanged. A sharded source is validated here once: the head block must be whole heads, the
/// rank's heads must be whole o-groups (64 heads / 8 groups / 4 ranks = 16 heads = 2 groups), and
/// its `wo_a` slice must be exactly those groups' rows. Anything else is refused before the body
/// runs, because a mis-aligned group split would silently mix heads from two groups.
///
/// # Errors
///
/// [EngineError::Shape] when the local geometry is not a whole-head, whole-group split of the
/// replicated geometry, or when the source disagrees with itself between `wq_b` and `wo_a`.
pub fn attention_shard_geometry(
    config: &ModelConfig,
    prefix: &str,
    source: &dyn WeightSource,
    world: usize,
) -> Result<Option<(usize, usize)>, EngineError> {
    let head_dim = config.head_dim;
    let heads_full = config.n_heads;
    let wq_b = source.spec(&format!("{prefix}.attn.wq_b.weight"))?;
    let rows = wq_b.shape.first().copied().unwrap_or(0);
    if rows == heads_full * head_dim {
        return Ok(None);
    }
    if world == 0 || head_dim == 0 || rows == 0 || !rows.is_multiple_of(head_dim) {
        return Err(EngineError::Shape {
            what: "attention shard".to_owned(),
            detail: format!(
                "{rows} wq_b rows are not the {world} ranks' whole heads of {head_dim}"
            ),
        });
    }
    let heads = rows / head_dim;
    if heads.checked_mul(world) != Some(heads_full) {
        return Err(EngineError::Shape {
            what: "attention shard".to_owned(),
            detail: format!(
                "{heads} local heads of {heads_full} over {world} ranks do not cover the model"
            ),
        });
    }
    let per_group = heads_full / config.o_groups.max(1);
    if per_group == 0 || !heads.is_multiple_of(per_group) {
        return Err(EngineError::Shape {
            what: "attention shard".to_owned(),
            detail: format!(
                "{heads} local heads do not split into whole o-groups of {per_group} heads"
            ),
        });
    }
    let groups = heads / per_group;
    if groups.checked_mul(world) != Some(config.o_groups) {
        return Err(EngineError::Shape {
            what: "attention shard".to_owned(),
            detail: format!(
                "{groups} local o-groups of {} over {world} ranks do not cover the groups",
                config.o_groups
            ),
        });
    }
    let wo_a = source.spec(&format!("{prefix}.attn.wo_a.weight"))?;
    let wo_a_rows = wo_a.shape.first().copied().unwrap_or(0);
    let expected = groups * config.o_lora_rank;
    if wo_a_rows != expected {
        return Err(EngineError::Shape {
            what: "attention shard".to_owned(),
            detail: format!(
                "wo_a serves {wo_a_rows} rows; {groups} local o-groups need {expected}"
            ),
        });
    }
    Ok(Some((heads, groups)))
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
    query_path_with(
        config,
        layer,
        prefix,
        x,
        start_pos,
        src.source,
        src.limit,
        &mut HostAttentionProjectionBackend,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn query_path_with(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    source: &dyn WeightSource,
    limit: usize,
    backend: &mut dyn AttentionProjectionBackend,
    shard: Option<&AttentionShard>,
) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
    let dim = config.dim;
    let rank = config.q_lora_rank;
    let head_dim = config.head_dim;
    let heads = shard.map_or(config.n_heads, |shard| shard.heads);
    let seqlen = rows_of(x, dim, "attention input")?;
    let q_norm = weight(
        Reads { source, limit },
        &format!("{prefix}.attn.q_norm.weight"),
    )?;
    let mut latent = backend.wq_a(config, layer, prefix, x, source, limit)?;
    linalg::rms_norm(&mut latent, &q_norm, rank, config.norm_eps as f32)?;
    // RMSNorm casts back to the bf16 input dtype, and wq_b's activation quantiser consumes
    // that bf16 value (model.py:281-293, 970-971), so the host seam narrows it (F5).
    narrow_bf16(&mut latent);
    let mut q = match shard {
        None => backend.wq_b(config, layer, prefix, &latent, source, limit)?,
        Some(_) => {
            // The rank's own head rows: the same ascending q_lora_rank dot per output element,
            // just fewer output elements. wq_b stays replicated for every other driver.
            let cols = heads * head_dim;
            let seqlen = rows_of(&latent, rank, "q latent")?;
            let wq_b = dense_view(
                Reads { source, limit },
                &format!("{prefix}.attn.wq_b"),
                cols,
                rank,
            )?;
            let mut q = vec![0.0; seqlen * cols];
            wq_b.matmul(&mut q, &latent, cols, rank, seqlen)?;
            q
        }
    };
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
    window_kv_with(
        config,
        layer,
        prefix,
        x,
        start_pos,
        src.source,
        src.limit,
        &mut HostAttentionProjectionBackend,
    )
}

#[allow(clippy::too_many_arguments)]
fn window_kv_with(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    source: &dyn WeightSource,
    limit: usize,
    backend: &mut dyn AttentionProjectionBackend,
) -> Result<Vec<f32>, EngineError> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let seqlen = rows_of(x, dim, "attention input")?;
    let norm = weight(
        Reads { source, limit },
        &format!("{prefix}.attn.kv_norm.weight"),
    )?;
    let mut kv = backend.wkv(config, layer, prefix, x, source, limit)?;
    linalg::rms_norm(&mut kv, &norm, head_dim, config.norm_eps as f32)?;
    // kv_norm casts back to bf16 (model.py:281-293, 700-707), and the fp8 window quantiser
    // reads that bf16 value (F5).
    narrow_bf16(&mut kv);
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
    // The rope writes back into the bf16 row before the fp8 window quantiser reads it
    // (model.py:700-707).
    narrow_and_round_trip(
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
    layer: usize,
    prefix: &str,
    x: &[f32],
    seqlen: usize,
    src: Reads<'_>,
    mut backend: Option<&mut dyn AttentionProjectionBackend>,
) -> Result<Vec<f32>, EngineError> {
    let dim = config.dim;
    let heads = config.index_n_heads;
    // The projection is a bf16 Linear over the bf16 layer input and hands back bf16; the factor
    // then multiplies that bf16 tensor and narrows the product (model.py:513, 555). A device arm
    // that serves the projection returns the same bf16-rounded rows, so the factor below still runs
    // here, on whichever rows the projection produced.
    let device = match backend.as_mut() {
        Some(arm) => {
            arm.indexer_weights_proj(config, layer, prefix, x, seqlen, src.source, src.limit)?
        }
        None => None,
    };
    let mut values = match device {
        Some(rows) => device_rows(rows, seqlen * heads, "indexer weights_proj projection")?,
        None => {
            let projection = plain_view(
                src,
                &format!("{prefix}.attn.indexer.weights_proj.weight"),
                heads,
                dim,
            )?;
            let rounded: Vec<f32> = x.iter().map(|value| quant::bf16_round(*value)).collect();
            let mut host = vec![0.0; seqlen * heads];
            projection.matmul(&mut host, &rounded, heads, dim, seqlen)?;
            for value in host.iter_mut() {
                *value = quant::bf16_round(*value);
            }
            host
        }
    };
    let factor = ((config.index_head_dim as f64).powf(-0.5) * (heads as f64).powf(-0.5)) as f32;
    for value in values.iter_mut() {
        *value = quant::bf16_round(*value * factor);
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
    mut backend: Option<&mut dyn AttentionProjectionBackend>,
) -> Result<(), EngineError> {
    if !owns_index_keys(config, layer) {
        return Ok(());
    }
    let ratio = layer_ratio(config, layer);
    if ratio == 0 {
        return Ok(());
    }
    if state.latent_rows == 0 {
        // Publish the owner's cache even when this forward adds no key row: the reference's channel
        // points at the owning layer's cache on every forward (model.py:748, 1173), so a consumer
        // that runs earlier in the next step must find this owner's cache and not a later source's.
        with_shared(|shared| {
            publish_rows(
                &mut shared.index_keys,
                &mut shared.index_rows,
                &mut shared.index_owner,
                layer,
                &state.indexer,
                config.index_head_dim,
            );
        });
        return Ok(());
    }
    let head_dim = config.head_dim;
    let index_head_dim = config.index_head_dim;
    // A device arm that serves this layer's wk returns the projection's f32 accumulators. The
    // k_norm, the bf16 narrowing, the RoPE and the fp4 round trip below are the host rule either
    // way, so an arm only ever replaces the one matmul.
    let device = match backend.as_mut() {
        Some(arm) => arm.indexer_wk(config, layer, prefix, &state.latent, src.source, src.limit)?,
        None => None,
    };
    let norm = weight(src, &format!("{prefix}.attn.indexer.k_norm.weight"))?;
    let mut keys = match device {
        Some(rows) => device_rows(
            rows,
            state.latent_rows * index_head_dim,
            "indexer wk projection",
        )?,
        None => {
            let wk = plain_view(
                src,
                &format!("{prefix}.attn.indexer.wk.weight"),
                index_head_dim,
                head_dim,
            )?;
            let mut host = vec![0.0; state.latent_rows * index_head_dim];
            wk.matmul(
                &mut host,
                &state.latent,
                index_head_dim,
                head_dim,
                state.latent_rows,
            )?;
            host
        }
    };
    linalg::rms_norm(&mut keys, &norm, index_head_dim, config.norm_eps as f32)?;
    // k_norm casts back to bf16 and the rope writes back into that bf16 tensor before the
    // fp4 round trip (model.py:281-293, 392-409, 540-546), so both are narrowed (F5).
    narrow_bf16(&mut keys);
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
    // The rope writes back into the bf16 key tensor before the fp4 quantiser reads it
    // (model.py:392-409, 540-546).
    narrow_and_round_trip(
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
            &state.indexer,
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
    if !is_kv_source(config, layer) {
        return Ok(());
    }
    let ratio = layer_ratio(config, layer);
    if ratio == 0 {
        return Ok(());
    }
    let head_dim = config.head_dim;
    // A step inside a partial group adds no row, but it still publishes the owner's whole cache
    // below: the reference's channel points at the owning layer's cache on every forward
    // (model.py:748, 1173), so a consumer that runs earlier in the next step must not read a later
    // source's cache in its place.
    if state.latent_rows > 0 {
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
        // The rope writes back into the bf16 latent before the compressed fp4 quantiser reads it
        // (model.py:757-761).
        narrow_and_round_trip(
            &mut state.latent,
            head_dim,
            COMPRESS_QUANT_BLOCK,
            Element::Fp4,
            ScaleRule::E4M3Six,
        );
        let rows = std::mem::take(&mut state.latent);
        store_rows(&mut state.compressed, first_row, &rows, head_dim);
        state.latent = rows;
    }
    with_shared(|shared| {
        publish_rows(
            &mut shared.compressed,
            &mut shared.compressed_rows,
            &mut shared.compressed_owner,
            layer,
            &state.compressed,
            head_dim,
        );
    });
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
            debug_assert_eq!(
                shared.compressed_owner,
                planned_kv_source(config, layer),
                "layer {layer} reads the compressed cache of its planned kv source, not another layer's (cache_engine.py:160-172 asserts the source precedes its consumer)"
            );
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
    output_projection_with(
        config,
        prefix,
        out,
        seqlen,
        src.source,
        src.limit,
        &mut HostAttentionProjectionBackend,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn output_projection_with(
    config: &ModelConfig,
    prefix: &str,
    out: &mut [f32],
    seqlen: usize,
    source: &dyn WeightSource,
    limit: usize,
    backend: &mut dyn AttentionProjectionBackend,
    shard: Option<&AttentionShard>,
) -> Result<(), EngineError> {
    let head_dim = config.head_dim;
    let dim = config.dim;
    let hidden_row = config.n_heads * head_dim;
    let grouped_width = config.o_groups * config.o_lora_rank;
    let grouped = match shard {
        None => {
            if out.len() < seqlen * hidden_row {
                return Err(shape(
                    "attention output projection",
                    format!(
                        "{seqlen} rows of {hidden_row} values need {} slots, got {}",
                        seqlen * hidden_row,
                        out.len()
                    ),
                ));
            }
            backend.wo_a(config, prefix, &out[..seqlen * hidden_row], source, limit)?
        }
        Some(shard) => {
            // The rank's own o-groups of wo_a over its own heads, then the exact rank-order copy
            // of the per-position o_lora vectors back into the full width. The copy has no
            // arithmetic, so it reconstructs the replicated vector bit for bit.
            let wide_row = shard.heads * head_dim;
            if out.len() < seqlen * wide_row {
                return Err(shape(
                    "attention output projection",
                    format!(
                        "{seqlen} rows of {wide_row} local head values need {} slots, got {}",
                        seqlen * wide_row,
                        out.len()
                    ),
                ));
            }
            let local = wo_a_local(
                config,
                prefix,
                &out[..seqlen * wide_row],
                seqlen,
                shard,
                source,
                limit,
            )?;
            let local_width = shard.groups * config.o_lora_rank;
            if local.len() != seqlen * local_width {
                return Err(shape(
                    "attention output projection",
                    format!(
                        "expected {} local o_lora values, got {}",
                        seqlen * local_width,
                        local.len()
                    ),
                ));
            }
            let world = shard.gather.world();
            if local_width.checked_mul(world) != Some(grouped_width) {
                return Err(shape(
                    "attention output projection",
                    format!(
                        "{local_width} local o_lora values over {world} ranks are not {grouped_width}"
                    ),
                ));
            }
            let mut gathered = vec![0.0f32; seqlen * grouped_width];
            shard.gather.gather_bits(&local, &mut gathered)?;
            // The gather is rank-major per call: [rank][position][local group]. wo_b reads
            // [position][group], so repack that axis order without touching a value.
            let mut ordered = vec![0.0f32; seqlen * grouped_width];
            for rank in 0..world {
                for position in 0..seqlen {
                    let from = (rank * seqlen + position) * local_width;
                    ordered[position * grouped_width + rank * local_width..][..local_width]
                        .copy_from_slice(&gathered[from..from + local_width]);
                }
            }
            ordered
        }
    };
    let projected = backend.wo_b(config, prefix, &grouped, source, limit)?;
    if projected.len() != seqlen * dim {
        return Err(shape(
            "attention output projection",
            format!(
                "expected {} projected values, got {}",
                seqlen * dim,
                projected.len()
            ),
        ));
    }
    out[..seqlen * dim].copy_from_slice(&projected);
    Ok(())
}

/// The grouped output projection over one rank's heads and o-groups.
///
/// This is [HostAttentionProjectionBackend::wo_a] with the rank's groups in place of all groups:
/// the local `wide` row holds the rank's heads in order, which is exactly its groups in order
/// because the head block is a whole number of groups. Every local row dot is the same ascending
/// accumulation the replicated path runs for that group.
fn wo_a_local(
    config: &ModelConfig,
    prefix: &str,
    wide: &[f32],
    seqlen: usize,
    shard: &AttentionShard,
    source: &dyn WeightSource,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let src = Reads { source, limit };
    let head_dim = config.head_dim;
    let per_group = config.n_heads / config.o_groups;
    let group_in = per_group * head_dim;
    let wide_row = shard.heads * head_dim;
    let grouped_width = shard.groups * config.o_lora_rank;
    if wide.len() < seqlen * wide_row {
        return Err(shape(
            "attention wo_a",
            format!(
                "{seqlen} rows of {wide_row} local head values need {} slots, got {}",
                seqlen * wide_row,
                wide.len()
            ),
        ));
    }
    let wo_a_name = format!("{prefix}.attn.wo_a");
    let wo_a = scaled_or_plain_view(src, &wo_a_name, grouped_width, group_in)?;
    // The same geometry exception as the replicated path: a grid whose row blocks are not whole
    // 32-row blocks must be materialised first. The released o_lora_rank 1024 is fine either way.
    let materialised = if wo_a.row_blocks_supported(config.o_lora_rank, group_in) {
        None
    } else {
        let scale_name = format!("{wo_a_name}.scale");
        let mut values = weight(src, &format!("{wo_a_name}.weight"))?;
        if src.source.spec(&scale_name).is_ok() {
            let scales = weight(src, &scale_name)?;
            linalg::apply_scales(
                &mut values,
                grouped_width,
                group_in,
                &scales,
                ScaleGrid::Block32,
            )?;
        }
        Some(values)
    };
    let mut grouped = vec![0.0; seqlen * grouped_width];
    for query in 0..seqlen {
        let wide_row_values = &wide[query * wide_row..][..wide_row];
        let dest = &mut grouped[query * grouped_width..][..grouped_width];
        if let Some(values) = &materialised {
            for group in 0..shard.groups {
                let source = &wide_row_values[group * group_in..][..group_in];
                let row_start = group * config.o_lora_rank;
                let block = &values[row_start * group_in..][..config.o_lora_rank * group_in];
                let block_out = &mut dest[row_start..][..config.o_lora_rank];
                linalg::matmul(block_out, source, block, config.o_lora_rank, group_in, 1)?;
            }
        } else if wo_a.supports_grouped() {
            wo_a.matmul_grouped(
                dest,
                wide_row_values,
                shard.groups,
                config.o_lora_rank,
                group_in,
            )?;
        } else {
            for group in 0..shard.groups {
                let source = &wide_row_values[group * group_in..][..group_in];
                let row_start = group * config.o_lora_rank;
                let block_out = &mut dest[row_start..][..config.o_lora_rank];
                wo_a.matmul_rows(
                    block_out,
                    source,
                    config.o_lora_rank,
                    group_in,
                    1,
                    row_start,
                )?;
            }
        }
    }
    Ok(grouped)
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
    attention_forward_with_backend(
        config,
        layer,
        prefix,
        x,
        start_pos,
        state,
        src,
        out,
        limit,
        &mut HostAttentionProjectionBackend,
    )
}

/// The batch-18 Option-A attention step: one rank's head block through the sparse body, its own
/// o-groups of wo_a, one exact o_lora all-gather and the replicated wo_b.
///
/// The sharded path is host-only on purpose: the body's arithmetic stays the reference f32
/// arithmetic, and the only new primitive is the exact gather. `out` keeps the replicated
/// contract of [attention_forward]: the first `positions * dim` values hold the residual row
/// afterwards.
///
/// # Errors
///
/// As [attention_forward], plus the gather's own transport error.
#[allow(clippy::too_many_arguments)]
pub fn attention_forward_sharded(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: &dyn WeightSource,
    out: &mut [f32],
    limit: usize,
    shard: &AttentionShard<'_>,
) -> Result<(), EngineError> {
    attention_forward_with(
        config,
        layer,
        prefix,
        x,
        start_pos,
        state,
        src,
        out,
        limit,
        &mut HostAttentionProjectionBackend,
        Some(shard),
    )
}

/// The attention stage of one layer with its projection and attention maths delegated to
/// the backend: the model-level f32 path for the CPU reference
/// (HostAttentionProjectionBackend) or a device seam. Writes the attention output for the
/// positions starting at start_pos into out.
#[allow(clippy::too_many_arguments)]
pub fn attention_forward_with_backend(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: &dyn WeightSource,
    out: &mut [f32],
    limit: usize,
    backend: &mut dyn AttentionProjectionBackend,
) -> Result<(), EngineError> {
    attention_forward_with(
        config, layer, prefix, x, start_pos, state, src, out, limit, backend, None,
    )
}

/// The body of [attention_forward_with_backend], with an optional output-row shard.
///
/// With a shard the head loop, the q rows and the o_lora rows are the rank's own; every value is
/// produced by the same ascending-f32 arithmetic as the replicated path. The device
/// `backend.attention` arm is skipped, because a device kernel has no rank-local head range yet.
#[allow(clippy::too_many_arguments)]
fn attention_forward_with(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: &dyn WeightSource,
    out: &mut [f32],
    limit: usize,
    backend: &mut dyn AttentionProjectionBackend,
    shard: Option<&AttentionShard<'_>>,
) -> Result<(), EngineError> {
    let src = Reads { source: src, limit };
    let dim = config.dim;
    let head_dim = config.head_dim;
    let heads_full = config.n_heads;
    let heads = shard.map_or(heads_full, |shard| shard.heads);
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
    if head_dim == 0 || config.o_groups == 0 || !heads_full.is_multiple_of(config.o_groups) {
        return Err(shape(
            "attention geometry",
            format!(
                "{heads_full} heads of {head_dim} do not split into {} groups",
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
    let (q, latent) = query_path_with(
        config, layer, prefix, x, start_pos, src.source, src.limit, backend, shard,
    )?;
    let kv_rows = window_kv_with(
        config, layer, prefix, x, start_pos, src.source, src.limit, backend,
    )?;
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
        compressor_step(config, layer, prefix, x, start_pos, state, src, None)?;
        // The index keys read the latent before the compressed cache rotates it (model.py:531).
        write_index_keys(config, layer, prefix, start_pos, state, src, None)?;
        if is_index_source(config, layer) && compress_len > 0 {
            state.index_weights = index_weights(config, layer, prefix, x, seqlen, src, None)?;
        } else {
            state.index_weights.clear();
        }
        compressed_indices_with_backend(
            state,
            config,
            layer,
            &latent,
            src.source,
            prefix,
            &mut compressed_ids,
            limit,
            &mut *backend,
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
    // The plan keeps attn_sink replicated; a sharded rank takes its own heads' values.
    let sink = match shard {
        None => sink,
        Some(shard) => {
            let start = shard.rank * shard.heads;
            let end = start + shard.heads;
            if sink.len() < heads_full {
                return Err(shape(
                    "attention sink",
                    format!("{} sinks for {heads_full} heads", sink.len()),
                ));
            }
            sink[start..end].to_vec()
        }
    };
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
    // The device body sees the prefill chunk at position 0, and one-ring-row decode steps through
    // the position-causal sequence entry. A backend that does not stage a decode step answers
    // false and the host loop below runs unchanged (the ring write is idempotent).
    let device_attention = if shard.is_some() {
        false
    } else if start_pos == 0 {
        let mut chunk_ids = Vec::with_capacity(seqlen * (window_width + compressed_stride));
        for query in 0..seqlen {
            chunk_ids.extend_from_slice(&ids[query * window_width..(query + 1) * window_width]);
            chunk_ids.extend_from_slice(
                &compressed_ids[query * compressed_stride..(query + 1) * compressed_stride],
            );
        }
        backend.attention(
            layer,
            &q,
            &kv,
            &sink,
            &chunk_ids,
            heads,
            head_dim,
            seqlen,
            window_width,
            compressed_stride,
            scale,
            start_pos,
            &mut out[..seqlen * heads * head_dim],
        )?
    } else if seqlen == 1 {
        let position = start_pos;
        state.window.write(position, &kv_rows[..head_dim])?;
        for slot in 0..config.window_size {
            kv[slot * head_dim..(slot + 1) * head_dim].copy_from_slice(state.window.slot_row(slot));
        }
        combined.clear();
        combined.extend_from_slice(&ids[..window_width]);
        combined.extend_from_slice(&compressed_ids[..compressed_stride]);
        backend.attention(
            layer,
            &q,
            &kv,
            &sink,
            &combined,
            heads,
            head_dim,
            seqlen,
            window_width,
            compressed_stride,
            scale,
            start_pos,
            &mut out[..seqlen * heads * head_dim],
        )?
    } else {
        false
    };
    if !device_attention {
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
    output_projection_with(
        config, prefix, out, seqlen, src.source, src.limit, backend, shard,
    )?;
    Ok(())
}

/// The host projection stages for the diagnostic twin: raw wq_a latent, raw wq_b q (before
/// q_norm/RoPE), q after q_norm/RoPE, raw wkv, the kv after kv_norm/RoPE, and the window row after
/// the fp8 round trip. All stages are produced by the host reference path on the same x.
#[derive(Debug, Clone)]
pub struct AttentionProjectionProbe {
    /// Raw wq_a output, before q_norm.
    pub latent_raw: Vec<f32>,
    /// The q_norm'd latent, the wq_b input.
    pub latent_normed: Vec<f32>,
    /// Raw wq_b output, after q_norm and before RoPE.
    pub q_raw: Vec<f32>,
    /// q after q_norm and RoPE.
    pub q_rot: Vec<f32>,
    /// Raw wkv output, before kv_norm.
    pub wkv_raw: Vec<f32>,
    /// The window KV after kv_norm, RoPE, and the fp8 round trip.
    pub window_kv: Vec<f32>,
}

/// Runs the host projection stages on one x row for the b57 projection twin.
///
/// # Errors
///
/// Any weight read or shape error from the host projection path.
pub fn attention_projection_probe(
    config: &ModelConfig,
    layer: usize,
    prefix: &str,
    x: &[f32],
    start_pos: usize,
    source: &dyn WeightSource,
    limit: usize,
) -> Result<AttentionProjectionProbe, EngineError> {
    let head_dim = config.head_dim;
    let rank = config.q_lora_rank;
    let mut backend = HostAttentionProjectionBackend;
    let latent_raw = backend.wq_a(config, layer, prefix, x, source, limit)?;
    let q_norm = weight(
        Reads { source, limit },
        &format!("{prefix}.attn.q_norm.weight"),
    )?;
    let mut latent = latent_raw.clone();
    linalg::rms_norm(&mut latent, &q_norm, rank, config.norm_eps as f32)?;
    narrow_bf16(&mut latent);
    let q_raw = backend.wq_b(config, layer, prefix, &latent, source, limit)?;
    let latent_normed = latent.clone();
    let tables = rope_tables_for_layer(
        config,
        layer,
        &head_row_positions(start_pos, 1, config.n_heads),
    );
    let mut q_rot = q_raw.clone();
    rotate_rows(
        &mut q_rot,
        config.n_heads,
        head_dim,
        config.rope_head_dim,
        &tables,
        false,
    )?;
    let wkv_raw = backend.wkv(config, layer, prefix, x, source, limit)?;
    let norm = weight(
        Reads { source, limit },
        &format!("{prefix}.attn.kv_norm.weight"),
    )?;
    let mut window_kv = wkv_raw.clone();
    linalg::rms_norm(&mut window_kv, &norm, head_dim, config.norm_eps as f32)?;
    narrow_bf16(&mut window_kv);
    rotate_rows(
        &mut window_kv,
        1,
        head_dim,
        config.rope_head_dim,
        &tables,
        false,
    )?;
    narrow_and_round_trip(
        &mut window_kv,
        head_dim,
        WINDOW_QUANT_BLOCK,
        Element::Fp8,
        ScaleRule::PowerOfTwo,
    );
    Ok(AttentionProjectionProbe {
        latent_raw,
        latent_normed,
        q_raw,
        q_rot,
        wkv_raw,
        window_kv,
    })
}

/// The DSpark ring write alone: the main rows' KV stored at their own positions, which is what
/// [dspark_attention_forward] does before it drafts (model.py:1040-1043, 1065). For a position whose
/// draft is not wanted, so the ring still holds every position a later draft attends.
///
/// # Errors
///
/// [EngineError] from any weight read or shape mismatch.
#[allow(clippy::too_many_arguments)] // dspark_attention_forward's inputs minus the draft rows.
pub fn dspark_seed_ring(
    config: &ModelConfig,
    stage: usize,
    prefix: &str,
    main_x: &[f32],
    start_pos: usize,
    state: &mut AttnState,
    src: &dyn WeightSource,
    limit: usize,
) -> Result<(), EngineError> {
    let src = Reads { source: src, limit };
    let head_dim = config.head_dim;
    let main_rows = rows_of(main_x, config.dim, "dspark main input")?;
    let layer = config.n_layers + stage;
    let main_kv = window_kv(config, layer, prefix, main_x, start_pos, src)?;
    for row in 0..main_rows {
        state.window.write(
            start_pos + row,
            &main_kv[row * head_dim..(row + 1) * head_dim],
        )?;
    }
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
    use crate::parallel::with_reference_threads;
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

    /// The reference's `precompute_freqs_cis` at the released rope geometry (rope_head_dim 64,
    /// base 160000 = `compress_rope_theta`, original_seq_len 16384, factor 16, beta 32/1), spelled
    /// out twice: with the reference's f32 ramp, and with the f64 ramp this host used before F8.
    fn released_yarn_tables(positions: &[usize], f64_ramp: bool) -> RopeTables {
        let (dim, base, original, factor) = (64usize, 160_000.0f64, 16_384usize, 16.0f32);
        let corrected = |rotations: f64| {
            dim as f64 * (original as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corrected(32.0).floor().max(0.0);
        let high = corrected(1.0).ceil().min(dim as f64 - 1.0);
        let span32: f32 = if high > low {
            (high - low) as f32
        } else {
            1e-3
        };
        let span64 = (high - low).max(1e-3);
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for &position in positions {
            for k in 0..dim / 2 {
                let freq = 1.0f32 / (base as f32).powf(2.0 * k as f32 / dim as f32);
                let ramp = if f64_ramp {
                    (((k as f64 - low) / span64).clamp(0.0, 1.0)) as f32
                } else {
                    ((k as f32 - low as f32) / span32).clamp(0.0, 1.0)
                };
                let smooth = 1.0 - ramp;
                let blended = if f64_ramp {
                    freq / factor * ramp + freq * smooth
                } else {
                    freq / factor * (1.0 - smooth) + freq * smooth
                };
                let angle = position as f32 * blended;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        RopeTables { cos, sin }
    }

    /// F8: the reference's YaRN band is f32 tensor arithmetic (model.py:377-386): the ramp is a f32
    /// arange minus the integer band edge over the integer span, clamped; the blend then re-derives
    /// (1 - smooth) from the rounded smooth instead of reusing the ramp. At the released geometry the
    /// f64-then-narrow ramp differs from that for one pair, and reusing the ramp differs for one more,
    /// which moves 3,862 of the 131,104 table cos entries over positions 0..=4096 (3,926 sin). This
    /// test spells the reference's arithmetic out independently, pins the tables to it, and pins that
    /// the old f64 variant is a different table so the fix cannot be undone silently.
    #[test]
    fn the_yarn_ramp_is_the_reference_f32_arithmetic() {
        let config = ModelConfig {
            original_seq_len: 16_384,
            rope_factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            ..ModelConfig::tiny()
        };
        let positions = [47usize, 4000];
        let tables = tables_for(&config, 160_000.0, 16_384, 64, &positions);
        let reference = released_yarn_tables(&positions, false);
        let f64_variant = released_yarn_tables(&positions, true);
        assert_eq!(
            tables.cos, reference.cos,
            "the tables must use the reference's f32 ramp"
        );
        assert_eq!(
            tables.sin, reference.sin,
            "the tables must use the reference's f32 ramp"
        );
        assert_ne!(
            reference.cos, f64_variant.cos,
            "the f64 ramp is a different table at this geometry, so the comparison above is live"
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

    /// The indexer's wq_b is an fp8 W8A8 ColumnParallelLinear: the bf16 q_norm output is
    /// quantised to E4M3 over 32-value blocks with power-of-two scales before the GEMM
    /// (model.py:519, 187-199).
    #[test]
    fn the_indexer_activation_is_bf16_narrowed_then_fp8_quantised() {
        let config = ModelConfig::tiny();
        let width = config.q_lora_rank;
        let q: Vec<f32> = (0..width).map(|index| 0.1 + index as f32 * 0.03).collect();
        let activation = indexer_activation_native(&q, width);
        assert_eq!(activation.len(), q.len());
        let narrowed: Vec<f32> = q.iter().map(|value| quant::bf16_round(*value)).collect();
        assert_ne!(
            activation, narrowed,
            "the E4M3 quantisation must move a value"
        );
        let mut twice = activation.clone();
        round_trip_rows(
            &mut twice,
            width,
            INDEX_QUANT_BLOCK,
            Element::Fp8,
            ScaleRule::PowerOfTwo,
        );
        assert_eq!(
            twice, activation,
            "a dequantised value must be a fixed point"
        );
    }

    /// The indexer's score path is bf16 in the reference: the einsum output, the rectified
    /// product and the per-head sum each narrow to bf16 (model.py:553-557). A margin smaller than
    /// half a bf16 step disappears, which is why the indexer's dtype is a discrete risk.
    #[test]
    fn the_indexer_scores_narrow_each_step_to_bf16() {
        let q = [0.1f32, 0.2];
        let keys = [0.3f32, 0.4, 1.0, 1.0];
        let weights = [0.5f32];
        let scores = index_scores(&q, &keys, &weights, 1, 2, 1, 2);
        let first_f32 = 0.1f32 * 0.3 + 0.2 * 0.4;
        let first_expected = quant::bf16_round(quant::bf16_round(first_f32) * 0.5);
        assert_eq!(
            scores[0], first_expected,
            "the dot and the product narrow to bf16"
        );
        assert_ne!(
            scores[0],
            first_f32 * 0.5,
            "the f32 score is not what the indexer uses"
        );
        let second_f32 = 0.1f32 * 1.0 + 0.2 * 1.0;
        let second_expected = quant::bf16_round(quant::bf16_round(second_f32) * 0.5);
        assert_eq!(scores[1], second_expected);
    }

    /// A property demo, not a measurement: a score margin below half a bf16 step disappears, so
    /// the top-k cannot separate the two positions and the tie rule (lower position) decides. The
    /// f32 score ordering would pick the other one. At the planned window lengths the released
    /// configuration never reaches this regime: index_topk (512) exceeds n_pos, so every reachable
    /// position is selected whatever the scores say (docs/archive/build/indexer-native-dtype.md).
    #[test]
    fn the_bf16_score_path_can_erase_a_margin_the_f32_path_keeps() {
        let q = [1.0f32, 0.0];
        let near = 1.0f32 + 2.0f32.powi(-10);
        let keys = [1.0f32, 0.0, near, 0.0];
        let weights = [1.0f32];
        let scores = index_scores(&q, &keys, &weights, 1, 2, 1, 2);
        assert_eq!(
            scores[0], scores[1],
            "the sub-step margin is gone, so the two positions tie"
        );
        let f32_scores = [1.0f32 * 1.0, 1.0f32 * near];
        assert!(
            f32_scores[1] > f32_scores[0],
            "the f32 path keeps the margin and ranks the later position first"
        );
    }

    /// weights_proj is a bf16 Linear and the factor multiplies its bf16 output, so every
    /// per-query weight is a bf16 value (model.py:513, 555).
    #[test]
    fn the_indexer_weights_are_bf16_narrowed() {
        reset_shared_attention();
        let config = ModelConfig::tiny();
        let hand = Hand::for_config(&config);
        let mut state = AttnState::for_config(&config);
        let rows = 4usize;
        let mut out = vec![0.0; rows * config.dim];
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
        assert!(!state.index_weights.is_empty());
        for value in &state.index_weights {
            assert_eq!(
                quant::bf16_round(*value),
                *value,
                "the per-query weights must be a bf16 tensor"
            );
        }
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

    /// F5: the reference's dynamic quantisers are called on bf16 tensors, so a host seam narrows the
    /// f32 value to bf16 first. Rounding before the quantiser is not cosmetic: there are blocks whose
    /// f32 elements straddle an E4M3 code boundary, and those pick a code the reference can never
    /// pick. The block below is found by a deterministic scan so the test states the rule instead of
    /// depending on hand-tuned magic constants.
    #[test]
    fn the_quantiser_seams_narrow_the_host_value_to_bf16_first() {
        let mut seed = 0x5EEDu32;
        let mut found = false;
        for _ in 0..10_000 {
            let mut block: Vec<f32> = Vec::with_capacity(32);
            for _ in 0..32 {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                block.push((seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0);
            }
            let mut narrowed = block.clone();
            narrow_bf16(&mut narrowed);
            let expected = quant::quantize_activation_fp8(&narrowed)
                .expect("32 divides by 32")
                .dequantize();
            let raw = quant::quantize_activation_fp8(&block)
                .expect("32 divides by 32")
                .dequantize();
            if expected != raw {
                // The scan found a block where the f32 bits would change the codes; the seam must
                // produce what the bf16 block produces.
                let mut mine = block.clone();
                narrow_and_round_trip(&mut mine, 32, 32, Element::Fp8, ScaleRule::PowerOfTwo);
                assert_eq!(
                    mine, expected,
                    "the seam quantises the bf16 value, not the f32 one"
                );
                assert_ne!(
                    expected, raw,
                    "quantising the f32 value picks a different code, which is the mis-rounding F5 names"
                );
                found = true;
                break;
            }
        }
        assert!(
            found,
            "a bf16 pre-round must be able to change an E4M3 code; the scan found no such block"
        );
    }

    /// F5: q_norm's output feeds wq_b's activation quantiser, and RMSNorm casts its result back to
    /// the bf16 input dtype (model.py:281-293, 970-971), so the host seam hands over a bf16 tensor.
    #[test]
    fn the_host_q_norm_output_is_a_bf16_tensor_before_the_wq_b_quantiser() {
        let config = ModelConfig {
            compress_ratios: vec![2, 0, 0],
            ..ModelConfig::tiny()
        };
        let hand = Hand::for_config(&config);
        // Perturb the input by less than a bf16 ulp on purpose: the seam must not carry those bits.
        let mut x = chunk(&config, 1, 0);
        for (index, value) in x.iter_mut().enumerate() {
            *value += 1.0 / 3000.0 * (index % 3) as f32;
        }
        let probe = attention_projection_probe(&config, 0, "layers.0", &x, 0, &hand, LIMIT)
            .expect("a probe");
        for value in &probe.latent_normed {
            assert_eq!(
                quant::bf16_round(*value),
                *value,
                "the q_norm output the wq_b quantiser reads must be a bf16 tensor"
            );
        }
    }

    /// F5: the compressor returns a bf16 latent (model.py:446-486, 484-486), which is what the
    /// compressed cache's fp4 quantiser consumes on the next seam.
    #[test]
    fn the_compressor_latent_is_a_bf16_tensor_before_the_fp4_quantiser() {
        let config = ModelConfig {
            compress_ratios: vec![2, 0, 0],
            ..ModelConfig::tiny()
        };
        let hand = Hand::for_config(&config);
        let mut x = chunk(&config, 2, 0);
        for (index, value) in x.iter_mut().enumerate() {
            *value += 1.0 / 3000.0 * (index % 3) as f32;
        }
        let mut state = AttnState::for_config(&config);
        compressor_step(
            &config,
            0,
            "layers.0",
            &x,
            0,
            &mut state,
            Reads {
                source: &hand,
                limit: LIMIT,
            },
            None,
        )
        .expect("one pooled group");
        assert_eq!(state.latent_rows, 1);
        for value in &state.latent {
            assert_eq!(
                quant::bf16_round(*value),
                *value,
                "the compressor latent is a bf16 tensor"
            );
        }
    }
    /// F2: every softmax probability is rounded to bf16 before the P*V product (kernel.py:339-341),
    /// while the denominator keeps the f32 probability (kernel.py:342-353). The rounding is not a
    /// no-op: a probability whose f32 bits sit below bf16 resolution changes the output.
    #[test]
    fn the_body_probabilities_are_bf16_rounded_before_the_value_product() {
        let head_dim = 4;
        // q = e0, so a row's score is its first slot times the scale; the two rows differ by
        // 0.0002 there, which is below the bf16 resolution of exp(0) = 1.
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let kv = [1.0f32, 1.0, 1.0, 1.0, 0.9998, 2.0, 2.0, 2.0];
        let sink = [-2.0f32];
        let indices = [0usize, 1];
        let scale = 0.25f32;
        let mut out = [0.0f32; 4];
        sparse_attention(&q, &kv, &sink, &indices, 1, head_dim, scale, &mut out)
            .expect("one head of four slots");

        // The same arithmetic spelled out: the denominator sums the f32 probabilities and the
        // sink, the numerator uses bf16(probability) * value.
        let first = 1.0f32 * scale;
        let second = 0.9998f32 * scale;
        let max = first.max(second);
        let first_probability = (first - max).exp();
        let second_probability = (second - max).exp();
        let denominator = (sink[0] - max).exp() + first_probability + second_probability;
        let rounded = quant::bf16_round(second_probability);
        let expected = [
            (first_probability * 1.0 + rounded * 0.9998) / denominator,
            (first_probability * 1.0 + rounded * 2.0) / denominator,
            (first_probability * 1.0 + rounded * 2.0) / denominator,
            (first_probability * 1.0 + rounded * 2.0) / denominator,
        ];
        assert_eq!(
            out, expected,
            "P*V uses the bf16-rounded probability and the f32 denominator"
        );
        let unrounded = [
            (first_probability * 1.0 + second_probability * 0.9998) / denominator,
            (first_probability * 1.0 + second_probability * 2.0) / denominator,
            (first_probability * 1.0 + second_probability * 2.0) / denominator,
            (first_probability * 1.0 + second_probability * 2.0) / denominator,
        ];
        assert_ne!(
            out, unrounded,
            "quantising the probability must be observable in the output"
        );
        assert_ne!(
            rounded, second_probability,
            "the test's probability must sit below bf16 resolution for the pin to bite"
        );
    }

    /// The chunk size is derived from the output buffer, not hardcoded: the shared 32,768-value
    /// staging buffer fits eight rows of 4,096 columns, and a buffer smaller than one row still
    /// yields one row.
    #[test]
    fn chunk_rows_are_derived_from_the_output_buffer() {
        assert_eq!(indexer_chunk_rows(4096, 32_768 * 2), 8);
        assert_eq!(indexer_chunk_rows(4096, 3 * 4096 * 2), 3);
        assert_eq!(indexer_chunk_rows(4096, 1), 1);
        assert_eq!(
            indexer_chunk_rows(0, 1),
            1,
            "an empty width still yields a chunk"
        );
        let rows = 19;
        let cols = 16;
        let chunk = indexer_chunk_rows(cols, 3 * cols * 2);
        assert_eq!(chunk, 3, "the buffer holds exactly three rows");
        let chunks: Vec<usize> = (0..rows)
            .step_by(chunk)
            .map(|start| (rows - start).min(chunk))
            .collect();
        assert_eq!(chunks, vec![3, 3, 3, 3, 3, 3, 1]);
    }

    /// Chunking is an allocation detail: each chunk projects its own rows through the same
    /// native rule, and the concatenation is bit-identical to one projection of all rows.
    #[test]
    fn chunked_projection_is_bit_identical_to_one_projection() {
        let rows = 19;
        let cols = 16;
        let inner = 32;
        let latent: Vec<f32> = (0..rows * inner).map(pattern).collect();
        let weight: Vec<f32> = (0..cols * inner).map(pattern).collect();
        let single = indexer_wq_b_native(&latent, &weight, rows, cols, inner)
            .expect("one projection of the whole chunk");
        let chunked = indexer_wq_b_chunked(rows, cols, 3 * cols * 2, |start, count| {
            let window = &latent[start * inner..(start + count) * inner];
            indexer_wq_b_native(window, &weight, count, cols, inner)
        })
        .expect("the same rows in chunks");
        assert_eq!(single.len(), rows * cols);
        assert_eq!(chunked.len(), single.len());
        for (index, (one, many)) in single.iter().zip(chunked.iter()).enumerate() {
            assert_eq!(
                one.to_bits(),
                many.to_bits(),
                "value {index} differs between one projection and the chunked one"
            );
        }
    }

    #[test]
    fn head_parallel_sparse_attention_is_bit_identical_to_one_thread() {
        // Heads are independent lanes: splitting them across threads must not move a single bit.
        let heads = 8usize;
        let head_dim = 8usize;
        let keys = 13usize;
        let q: Vec<f32> = (0..heads * head_dim)
            .map(|index| 0.013 * index as f32 - 0.4)
            .collect();
        let kv: Vec<f32> = (0..keys * head_dim)
            .map(|index| {
                let mixed = (index.wrapping_mul(7) % 11) as f32;
                (mixed - 5.0) * 0.02
            })
            .collect();
        let sink: Vec<f32> = (0..heads).map(|head| 0.1 * head as f32 - 0.2).collect();
        let indices: Vec<usize> = vec![0, 3, 4, 7, 9, 12, NO_INDEX, 2];
        let run = |threads: usize| {
            crate::parallel::with_reference_threads(threads, || {
                let mut out = vec![0.0f32; heads * head_dim];
                sparse_attention(&q, &kv, &sink, &indices, heads, head_dim, 0.5, &mut out)
                    .expect("the sparse kernel runs");
                out
            })
        };
        let serial = run(1);
        for threads in [2usize, 4, 20] {
            let parallel = run(threads);
            assert_eq!(parallel.len(), serial.len());
            for (index, (one, many)) in serial.iter().zip(parallel.iter()).enumerate() {
                assert_eq!(
                    one.to_bits(),
                    many.to_bits(),
                    "value {index} differs between one thread and {threads}"
                );
            }
        }
    }

    /// The released model has four KV sources (layers 2, 8, 14, 20), so every consumer's planned
    /// source is the nearest preceding one and every source reads its own cache.
    #[test]
    fn a_consumer_plans_to_read_the_nearest_preceding_kv_source() {
        let config = ModelConfig::v41_flash().expect("the released config parses");
        for (layer, source) in [
            (3, 2),
            (7, 2),
            (9, 8),
            (15, 14),
            (21, 20),
            (24, 20),
            (36, 20),
            (39, 20),
        ] {
            assert_eq!(
                planned_kv_source(&config, layer),
                Some(source),
                "layer {layer} plans to read layer {source}"
            );
        }
        assert_eq!(
            planned_kv_source(&config, 2),
            None,
            "the first source has no preceding source"
        );
    }

    /// A consumer that finds its planned source's cache in the channel reads it.
    #[test]
    fn a_consumer_reads_its_planned_source() {
        reset_shared_attention();
        let config = ModelConfig {
            kv_source_layers: vec![0, 4],
            ..ModelConfig::tiny()
        };
        let head_dim = config.head_dim;
        let owner_cache = vec![0.25f32; head_dim];
        with_shared(|shared| {
            publish_rows(
                &mut shared.compressed,
                &mut shared.compressed_rows,
                &mut shared.compressed_owner,
                4,
                &owner_cache,
                head_dim,
            );
        });
        let state = AttnState::for_config(&config);
        let rows = fetch_compressed(&config, 5, &state, 1).expect("the planned source serves");
        assert_eq!(rows, owner_cache);
    }

    /// Another layer's cache in the channel is a layer-order regression, and the reference asserts
    /// the owner-before-consumer relation at construction (cache_engine.py:160-172); the debug assert
    /// turns the silent cross-segment read into a fail-fast panic.
    #[test]
    #[should_panic(expected = "planned kv source")]
    fn a_consumer_refuses_an_owner_that_is_not_its_planned_source() {
        reset_shared_attention();
        let config = ModelConfig {
            kv_source_layers: vec![0, 4],
            ..ModelConfig::tiny()
        };
        let head_dim = config.head_dim;
        let owner_cache = vec![0.25f32; head_dim];
        with_shared(|shared| {
            publish_rows(
                &mut shared.compressed,
                &mut shared.compressed_rows,
                &mut shared.compressed_owner,
                0,
                &owner_cache,
                head_dim,
            );
        });
        let state = AttnState::for_config(&config);
        let _ = fetch_compressed(&config, 5, &state, 1);
    }

    /// The seam step must reproduce exactly what the full reference forward captured for the same
    /// query: the compressor cache rows, the index keys and the selected ids are the same
    /// arithmetic in the same order, because the seam step is the same calls the full forward makes
    /// minus the projections the device loop already ran. The comparison is against the capture
    /// hook's record, so it also pins the id conventions the kernel staging depends on.
    #[test]
    fn the_seam_step_matches_the_full_forward_capture() {
        use crate::forward::attention_case;
        let config = ModelConfig::tiny();
        let source = Hand::for_config(&config);
        let dim = config.dim;
        let layers = config.n_layers;
        let head_dim = config.head_dim;
        let positions = 9usize;
        let x: Vec<f32> = (0..dim).map(|i| pattern(i) * 0.5).collect();
        let prefix = |layer: usize| format!("layers.{layer}");

        // The full forward, one query per step, capturing every layer's frame.
        reset_shared_attention();
        let mut full_states: Vec<AttnState> = (0..layers)
            .map(|_| AttnState::for_config(&config))
            .collect();
        let mut records: Vec<Vec<attention_case::AttentionCaseRecord>> = Vec::new();
        for position in 0..positions {
            let mut row = Vec::new();
            for (layer, state) in full_states.iter_mut().enumerate() {
                attention_case::begin(&[layer], position).expect("capture");
                let wide = (config.n_heads * head_dim).max(dim);
                let mut out = vec![0.0f32; wide];
                attention_forward(
                    &config,
                    layer,
                    &prefix(layer),
                    &x,
                    position,
                    state,
                    &source,
                    &mut out,
                    LIMIT,
                )
                .expect("full forward");
                let captured = attention_case::finish();
                assert_eq!(captured.len(), 1, "one query captured");
                row.push(captured.into_iter().next().expect("record"));
            }
            records.push(row);
        }

        // The seam-only path on fresh states, fed the same query latent the full forward used.
        reset_shared_attention();
        let mut seam_states: Vec<AttnState> = (0..layers)
            .map(|_| AttnState::for_config(&config))
            .collect();
        for (position, position_records) in records.iter().enumerate() {
            for (layer, state) in seam_states.iter_mut().enumerate() {
                let mut backend = HostAttentionProjectionBackend;
                let (_q, latent) = query_path_with(
                    &config,
                    layer,
                    &prefix(layer),
                    &x,
                    position,
                    &source,
                    LIMIT,
                    &mut backend,
                    None,
                )
                .expect("query path");
                let seam = attention_seam_step(
                    &config,
                    layer,
                    &prefix(layer),
                    &x,
                    &latent,
                    position,
                    state,
                    &source,
                    LIMIT,
                )
                .expect("seam step");
                let record = &position_records[layer];
                let attended = record
                    .window_ids
                    .iter()
                    .filter(|id| **id != NO_INDEX)
                    .count();
                assert_eq!(
                    seam.ratio, record.ratio,
                    "layer {layer} position {position} ratio"
                );
                assert_eq!(
                    seam.window_rows, record.window_rows,
                    "layer {layer} position {position} window rows"
                );
                assert_eq!(
                    seam.window_width, attended,
                    "layer {layer} position {position} attended window rows"
                );
                assert_eq!(
                    seam.compressed_ids, record.compressed_ids,
                    "layer {layer} position {position} compressed ids"
                );
                assert_eq!(
                    seam.compressed_rows, record.compressed_kv,
                    "layer {layer} position {position} compressed rows"
                );
                assert_eq!(
                    seam.compressed_rows.len(),
                    record.compressed_rows * head_dim,
                    "layer {layer} position {position} compressed row count"
                );
            }
        }
    }

    /// A [Hand] that counts the reads each tensor name sees.
    struct Counting<'a> {
        inner: &'a Hand,
        reads: std::sync::Mutex<BTreeMap<String, usize>>,
    }

    impl<'a> Counting<'a> {
        fn new(inner: &'a Hand) -> Self {
            Self {
                inner,
                reads: std::sync::Mutex::new(BTreeMap::new()),
            }
        }

        fn reads_of(&self, name: &str) -> usize {
            self.reads
                .lock()
                .expect("read counter")
                .get(name)
                .copied()
                .unwrap_or(0)
        }
    }

    impl WeightSource for Counting<'_> {
        fn names(&self) -> Vec<String> {
            self.inner.names()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.inner.spec(name)
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            *self
                .reads
                .lock()
                .expect("read counter")
                .entry(name.to_owned())
                .or_insert(0) += 1;
            self.inner.read(name)
        }
    }

    /// The query latent the compressor seam tests feed: the index keys it holds only need values.
    fn zero_latent(config: &ModelConfig) -> Vec<f32> {
        vec![0.0; config.n_heads * config.head_dim]
    }

    /// The seam state a cold-versus-warm comparison checks beyond the seam's own fields.
    fn seam_state(state: &AttnState) -> (Vec<f32>, Vec<f32>, usize, &[f32], &[f32]) {
        (
            state.compressed.clone(),
            state.indexer.clone(),
            state.compressed_len,
            &state.kv_state,
            &state.score_state,
        )
    }

    #[test]
    fn the_compressor_weights_are_read_once_per_layer() {
        let config = ModelConfig::tiny();
        let hand = Hand::for_config(&config);
        let counting = Counting::new(&hand);
        let x = chunk(&config, 1, 0);
        let q = zero_latent(&config);
        reset_shared_attention();
        let mut state = AttnState::for_config(&config);
        for position in 0..3 {
            attention_seam_step(
                &config, 0, "layers.0", &x, &q, position, &mut state, &counting, LIMIT,
            )
            .expect("seam step");
        }
        // Layer 0 is a KV source at ratio 2: one projection and one gate, decoded once.
        assert_eq!(
            counting.reads_of("layers.0.attn.compressor.wkv.weight"),
            1,
            "the wkv weight was decoded more than once"
        );
        assert_eq!(
            counting.reads_of("layers.0.attn.compressor.wgate.weight"),
            1,
            "the gate weight was decoded more than once"
        );
    }

    #[test]
    fn the_compressor_cache_is_value_transparent() {
        let config = ModelConfig::tiny();
        let hand = Hand::for_config(&config);
        let x = chunk(&config, 1, 0);
        let q = zero_latent(&config);
        reset_shared_attention();
        let mut warm = AttnState::for_config(&config);
        for position in 0..2 {
            attention_seam_step(
                &config, 0, "layers.0", &x, &q, position, &mut warm, &hand, LIMIT,
            )
            .expect("seam step");
        }
        // The same state twice: one keeps the decoded weights, the other drops them and decodes
        // again, exactly as the per-position path did. The seam and the state must not differ.
        let mut cold = warm.clone();
        cold.compressor = None;
        let warm_seam =
            attention_seam_step(&config, 0, "layers.0", &x, &q, 2, &mut warm, &hand, LIMIT)
                .expect("warm seam step");
        let cold_seam =
            attention_seam_step(&config, 0, "layers.0", &x, &q, 2, &mut cold, &hand, LIMIT)
                .expect("cold seam step");
        assert_eq!(warm_seam, cold_seam, "the seam step differs with a cache");
        assert_eq!(warm.latent, cold.latent, "the latent differs with a cache");
        assert_eq!(
            seam_state(&warm),
            seam_state(&cold),
            "the seam state differs with a cache"
        );
    }

    /// A backend that answers nothing and counts how often the seam step asks for the indexer's
    /// projection. The other five projections are the attention sublayer's, which the seam step
    /// does not run at all, so a call to one of them is a bug in what the step consults the
    /// backend for; panicking names it instead of hiding it.
    struct AskedOnlyIndexer {
        asked: usize,
    }

    impl AttentionProjectionBackend for AskedOnlyIndexer {
        fn wq_a(
            &mut self,
            _config: &ModelConfig,
            _layer: usize,
            _prefix: &str,
            _x: &[f32],
            _src: &dyn WeightSource,
            _limit: usize,
        ) -> Result<Vec<f32>, EngineError> {
            panic!("the seam step must not ask the backend for wq_a")
        }

        fn wq_b(
            &mut self,
            _config: &ModelConfig,
            _layer: usize,
            _prefix: &str,
            _latent: &[f32],
            _src: &dyn WeightSource,
            _limit: usize,
        ) -> Result<Vec<f32>, EngineError> {
            panic!("the seam step must not ask the backend for wq_b")
        }

        fn wkv(
            &mut self,
            _config: &ModelConfig,
            _layer: usize,
            _prefix: &str,
            _x: &[f32],
            _src: &dyn WeightSource,
            _limit: usize,
        ) -> Result<Vec<f32>, EngineError> {
            panic!("the seam step must not ask the backend for wkv")
        }

        fn wo_a(
            &mut self,
            _config: &ModelConfig,
            _prefix: &str,
            _wide: &[f32],
            _src: &dyn WeightSource,
            _limit: usize,
        ) -> Result<Vec<f32>, EngineError> {
            panic!("the seam step must not ask the backend for wo_a")
        }

        fn wo_b(
            &mut self,
            _config: &ModelConfig,
            _prefix: &str,
            _grouped: &[f32],
            _src: &dyn WeightSource,
            _limit: usize,
        ) -> Result<Vec<f32>, EngineError> {
            panic!("the seam step must not ask the backend for wo_b")
        }

        fn indexer_wq_b(
            &mut self,
            _config: &ModelConfig,
            _layer: usize,
            _prefix: &str,
            _latent: &[f32],
            _src: &dyn WeightSource,
            _limit: usize,
        ) -> Result<Option<Vec<f32>>, EngineError> {
            self.asked += 1;
            // None keeps the host native rule, which is what makes this the transparency test.
            Ok(None)
        }
    }

    #[test]
    fn the_seam_step_with_a_backend_is_the_host_seam_step_until_the_backend_answers() {
        let config = ModelConfig::tiny();
        let hand = Hand::for_config(&config);
        let x = chunk(&config, 1, 0);
        let q = zero_latent(&config);
        let run = |backend: &mut dyn AttentionProjectionBackend| {
            reset_shared_attention();
            let mut state = AttnState::for_config(&config);
            let mut seams = Vec::new();
            for position in 0..4 {
                seams.push(
                    attention_seam_step_with_backend(
                        &config, 0, "layers.0", &x, &q, position, &mut state, &hand, LIMIT, backend,
                    )
                    .expect("seam step"),
                );
            }
            (seams, seam_state(&state).0, seam_state(&state).1)
        };
        let host = run(&mut HostAttentionProjectionBackend);
        let mut asked = AskedOnlyIndexer { asked: 0 };
        let with_backend = run(&mut asked);
        assert_eq!(
            host.0, with_backend.0,
            "a backend that answers None changed the seam steps"
        );
        assert_eq!(
            (host.1, host.2),
            (with_backend.1, with_backend.2),
            "a backend that answers None changed the seam state"
        );
        assert!(
            asked.asked > 0,
            "the backend was never asked for the indexer's projection, so the index-source layer did not reach it"
        );
    }

    #[test]
    fn the_compressor_row_split_is_bit_identical_across_thread_counts() {
        let config = ModelConfig::tiny();
        let hand = Hand::for_config(&config);
        let x = chunk(&config, 1, 0);
        let q = zero_latent(&config);
        let run = |threads: usize| {
            reset_shared_attention();
            let mut state = AttnState::for_config(&config);
            let mut seams = Vec::new();
            for position in 0..4 {
                seams.push(
                    with_reference_threads(threads, || {
                        attention_seam_step(
                            &config, 0, "layers.0", &x, &q, position, &mut state, &hand, LIMIT,
                        )
                    })
                    .expect("seam step"),
                );
            }
            (seams, seam_state(&state).0, seam_state(&state).1)
        };
        let (serial_seams, serial_compressed, serial_indexer) = run(1);
        let (wide_seams, wide_compressed, wide_indexer) = run(4);
        assert_eq!(
            serial_seams, wide_seams,
            "the seam steps differ by thread count"
        );
        assert_eq!(
            serial_compressed, wide_compressed,
            "the compressed rows differ by thread count"
        );
        assert_eq!(
            serial_indexer, wide_indexer,
            "the index keys differ by thread count"
        );
    }
}
