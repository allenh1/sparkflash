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

//! The attention half's device stage bodies (S1 P2): the words and launches for
//! `reference_seam`, `ref_select`, `stage_attention`, `attention_body` and `attention_output` of
//! the whole-layer device arm, as pure functions over explicit handles.
//!
//! This module is the launch half only. The `LayerStages` method bodies in
//! `crates/sparkflash-cli/src/tp4_forward.rs` gather the spans and pointers out of `DeviceLayer`
//! and call the functions here; nothing here sees `DeviceLayer`, allocates, loads an image, reads
//! the environment or counts a probe. The arithmetic that decides *whether* a stage may run is in
//! `sparkflash_kernels::attention_operands` and the stage contracts in
//! `crates/sparkflash-cli/src/layer_device.rs`; this module turns a validated set of operands into
//! the pinned argument arrays the kernels declare, and refuses a shape the launch cannot take.
//!
//! The wiring spec and the operand derivation are in `docs/build/s1-p2-attention-wiring.md`. Two
//! rules from it are load-bearing enough to repeat:
//!
//! * the loop's `position` argument is the **0-based token index `p`** - the window includes the
//!   current row, so `window_rows = min(p + 1, 128)` and `window_start = p + 1 - window_rows`;
//! * the body's `selected_count` on a window-only layer (`compress_ratio == 0`, released layers 0
//!   and 1) is `0`, so no store row and no selected id is read; the store spans are still valid
//!   pointers because the sequence entry's ABI has no null branch for them.
//!
//! Nothing in this module has run on a device: there is no GPU in the environment it was written
//! in. The launches are shape-checked and word-tested here; the numerics are the qa-owner's gate.

#![cfg(any(feature = "cuda", test))]
// The `LayerStages` bodies that call these functions are the Lead's wiring step and land in the
// same integration window; until they do, every item below is unreachable from the library target
// and `-D warnings` would flag it. The annotation goes away with that wiring.
#![allow(dead_code)]

use crate::layer_device::DeviceSpan;
use sparkflash_kernels::attention_operands::{
    EXPANDED_ROWS, MAX_ATTENDED, MAX_SELECTED, NO_INDEX, OperandError, OutputGeometry,
    SELECTED_WORD_BYTES, StoreOperands, WINDOW_ROWS, WindowOperands, index_key_block_bytes,
};
use sparkflash_kernels::bf16a_mxfp8w_wo_a_launch::{self as wo_a_launch, Bf16aMxfp8wWoAParams};
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::flash_mla_csa2_seq_launch::{
    FORMAT_CSA_UNION, FlashMlaSeqParams, HEADS, THREADS,
};
use sparkflash_kernels::layer_seams_launch::{self as seams, PrologueBf16Params, RopeRowsParams};
use sparkflash_kernels::mxfp8_dense_launch::Mxfp8DenseParams;
use sparkflash_kernels::mxfp8_gemv_launch;

/// `NO_INDEX` re-exported for a stage body that maps a selected id, so the call site does not
/// depend on which of the two modules spells the sentinel.
pub const NO_SELECTED_ID: u32 = NO_INDEX;

/// The indexer's per-head width: the seq entry's `index_query` row and `index_key` token width.
const INDEX_HEAD_DIM: usize = sparkflash_kernels::mla_ref::INDEX_HEAD_DIM;
/// The indexer blocks the body's key pool holds.
const HSI_BLOCKS: usize = sparkflash_kernels::mla_ref::HSI_BLOCKS;
/// The compressed store row counts the body indexes: `position` maps to `(position + 1) / ratio`.
const STORE_POSITIONS: usize = EXPANDED_ROWS;

/// What one position's attention stages hand to each other.
///
/// `reference_seam` builds it, `ref_select` and `stage_attention` check it, and the two launch
/// stages read it. It is host state only: nothing here is read back from the device, which is what
/// keeps the stage chain free of synchronisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionStepState {
    /// The layer's compress ratio (`config.compress_ratio(layer)`), 0 for a window-only layer.
    pub ratio: usize,
    /// The window this position's body attends.
    pub window: WindowOperands,
    /// How many selected compressed ids the body is handed (0 at ratio 0).
    pub selected_count: u32,
    /// The compressed rows the store has published at this position (`(p + 1) / ratio`).
    pub store_rows: usize,
    /// Whether the store is still empty at this position (`store_rows == 0`).
    pub store_zero: bool,
}

impl AttentionStepState {
    /// The body's attended row count: window rows plus the selected rows it also reads.
    ///
    /// # Errors
    /// [OperandError::SelectedPastTopk] or [OperandError::AttendedPastBound] when the union is past
    /// what the sequence entry holds.
    pub fn attended(&self) -> Result<usize, String> {
        self.window
            .attended(self.selected_count as usize)
            .map_err(|error| refused("attention step", &error))
    }
}

/// The device `reference_seam` decision for one position.
///
/// On a window-only layer (`ratio == 0`, released layers 0 and 1) the seam stage is a pure
/// decision: the reference's `window_width` is the count of non-`NO_INDEX` window ids, which is
/// `min(p + 1, 128)`, and there is no compressor, no index key and no compressed row to publish.
/// The returned state records `selected_count = 0` and `store_rows = 0`.
///
/// On a compressed layer the stage refuses: the pooled latent the append needs comes from
/// `attention.compressor.wkv`/`.wgate`, which are `AttentionHostSeam` tensors outside the device
/// bind on this tree (19 of them, see the wiring doc's part B). The refusal is deliberately a
/// refusal and not a skip - a silently skipped seam would read an unwritten store row.
///
/// # Errors
/// A compressed layer, a ring with no rows, or a position whose row has not been appended yet.
pub fn reference_seam(
    ratio: usize,
    position: usize,
    ring_rows: usize,
    max_positions: usize,
) -> Result<AttentionStepState, String> {
    if ratio != 0 {
        return Err(format!(
            "reference_seam: layer ratio {ratio} is compressed, and the compressor producer \
             (attention.compressor.wkv/.wgate, the pooled latent's norm and rope, and the \
             compressed append) has no device path: those 19 AttentionHostSeam tensors are not \
             in the device bind"
        ));
    }
    let window = WindowOperands::checked(position, ring_rows, max_positions)
        .map_err(|error| refused("reference_seam", &error))?;
    Ok(AttentionStepState {
        ratio,
        window,
        selected_count: 0,
        store_rows: 0,
        store_zero: true,
    })
}

/// The device `ref_select` decision for one position.
///
/// At ratio 0 there is no compressed position to score: the reference's `compressed_indices` returns
/// an empty list and the select kernel is never launched. The stage is a documented no-op that
/// asserts the invariant, not a skipped check.
///
/// A compressed layer refuses here: `ref_select_sm121` takes the indexer's query, key pool and
/// per-query weights, and `attention.indexer.wk`/`.weights_proj` are outside the device bind on
/// this tree.
///
/// # Errors
/// A compressed layer, or a state whose selected count is not the ratio-0 empty list.
pub fn ref_select(ratio: usize, state: &AttentionStepState) -> Result<(), String> {
    if ratio != 0 {
        return Err(format!(
            "ref_select: layer ratio {ratio} is compressed, and the select kernel's operands \
             (the indexer query, the index key pool and the indexer weights_proj) have no device \
             producer: the indexer tensors are AttentionHostSeam entries outside the device bind"
        ));
    }
    if state.selected_count != 0 {
        return Err(format!(
            "ref_select: a {ratio}-ratio layer selected {} compressed ids; a window-only layer has \
             no compressed positions to select from",
            state.selected_count
        ));
    }
    Ok(())
}

/// The device `stage_attention` validation for one position.
///
/// The reference stage assembles its operands from the host cache (`cache::Plan`/`cache::Store`)
/// and uploads the expanded store; the device arm has neither, so this is the checked subset of the
/// reference's rules with the store cursor **derived from the position**:
///
/// * the ring the step was built against is the run's capacity, and the capacity is inside the
///   position budget;
/// * the position's row is in the ring (the `kv_path` append has landed);
/// * the body's window is 1..=128 rows, its selected list at most 512, and the union at most 640;
/// * the store row count the body indexes is `(p + 1) / ratio` at ratio > 0, and empty at ratio 0.
///
/// No device read, no copy and no launch happen here.
///
/// # Errors
/// A capacity that disagrees with the step, a capacity past the position budget, a position past
/// the ring, a window or selected list past the body's bounds, or a store cursor that disagrees with
/// the position.
pub fn stage_attention(
    state: &AttentionStepState,
    capacity_rows: usize,
    max_positions: usize,
) -> Result<(), String> {
    if state.window.ring_rows != capacity_rows {
        return Err(format!(
            "stage_attention: the step's window was built over {} ring rows, the run's capacity is \
             {capacity_rows}",
            state.window.ring_rows
        ));
    }
    WindowOperands::checked(state.window.position, capacity_rows, max_positions)
        .map_err(|error| refused("stage_attention", &error))?;
    if state.window.window_rows == 0 || state.window.window_rows > WINDOW_ROWS {
        return Err(format!(
            "stage_attention: a {} row window at position {}; the body reads 1..={WINDOW_ROWS} \
             rows, the current token included",
            state.window.window_rows, state.window.position
        ));
    }
    let expected = if state.ratio == 0 {
        let zero = state.store_rows == 0 && state.store_zero;
        if !zero {
            return Err(format!(
                "stage_attention: a 0-ratio layer reports {} store rows (zero: {}) at position {}; \
                 a window-only layer publishes no compressed row",
                state.store_rows, state.store_zero, state.window.position
            ));
        }
        None
    } else {
        let operands = StoreOperands::new(state.ratio, state.window.position)
            .map_err(|error| refused("stage_attention", &error))?;
        Some(operands)
    };
    if let Some(operands) = expected
        && (state.store_rows != operands.rows_written || state.store_zero != operands.store_zero)
    {
        return Err(format!(
            "stage_attention: the {}-ratio store reports {} rows at position {} (zero: {}), the \
             position publishes {} (zero: {})",
            state.ratio,
            state.store_rows,
            state.window.position,
            state.store_zero,
            operands.rows_written,
            operands.store_zero
        ));
    }
    state.attended().map(|_| ())
}

/// The sequence entry's operands at one position, per rank.
///
/// Every span is a `DeviceSpan` the caller owns: the length is what this module checks the launch
/// against, and the device pointer is what the kernel takes. `window_codes`/`window_scales` are raw
/// addresses inside the run's window ring (`attention_operands::window_ring_addresses` produces
/// them), not spans, because the ring is one allocation shared by every layer.
#[derive(Debug, Clone, Copy)]
pub struct BodyOperands {
    /// The query row: `query_path`'s wq_b output with the forward rope applied.
    pub q: DeviceSpan,
    /// The compressed store's packed FP4 codes (unread while `selected_count` is 0).
    pub kv_fp4: DeviceSpan,
    /// The compressed store's E4M3 scales (unread while `selected_count` is 0).
    pub kv_scales: DeviceSpan,
    /// The window's first code byte, `ring base + codes_offset(slot(window_start))`.
    pub window_codes: Deviceptr,
    /// The window's first scale byte, `ring base + scales_offset(slot(window_start))`.
    pub window_scales: Deviceptr,
    /// The window rows supplied, oldest first: `min(p + 1, 128)`.
    pub window_rows: u32,
    /// The selected compressed ids (valid pointer, count 0 on a window-only layer).
    pub selected: DeviceSpan,
    /// How many selected ids the body reads.
    pub selected_count: u32,
    /// The run-wide index key pool (zero-filled until an index-key producer lands).
    pub index_key: DeviceSpan,
    /// The indexer query row the entry reads: index head 0 only.
    pub index_query: DeviceSpan,
    /// The layer's per-head sink logits, `heads_per_rank` f32 values.
    pub sink: DeviceSpan,
    /// The block score scratch the entry writes every launch.
    pub block_scores: DeviceSpan,
    /// The block best-token scratch the entry writes every launch.
    pub block_best: DeviceSpan,
    /// The body's output row, `heads_per_rank` rows of `head_width` bf16.
    pub out: DeviceSpan,
    /// The 0-based token position `p`, as the kernel's `position` word.
    pub position: u32,
    /// The per-rank head count: the launch's `grid.x`.
    pub heads_per_rank: u32,
    /// One head row's width (512 at the released geometry).
    pub head_width: u32,
    /// The trace ring's device address, or 0.
    pub trace_ring: u64,
    /// The layer id the trace event records.
    pub layer_id: u32,
    /// The step sequence the trace event records.
    pub step_seq: u64,
}

/// The 19 argument words of the sequence entry, checked against the operands they came from.
///
/// # Errors
/// A window outside 1..=128 rows, a selected list past 512, a union past 640, a head count past the
/// kernel's fixed 64 or with no heads at all, an output row too short for the head rows, a selected
/// span too short for its count, an index key pool too short for the position's blocks, an index
/// query span shorter than one index head, a sink span shorter than the per-rank head count, or a
/// block scratch shorter than the pool the entry writes.
pub fn body_params(operands: &BodyOperands) -> Result<FlashMlaSeqParams, String> {
    let heads = operands.heads_per_rank as usize;
    if heads == 0 {
        return Err("attention body: the launch has no query heads".to_owned());
    }
    if heads > HEADS {
        return Err(format!(
            "attention body: {heads} heads per rank, the sequence entry's fixed grid serves {HEADS}"
        ));
    }
    if operands.head_width == 0 {
        return Err("attention body: the head row is zero values wide".to_owned());
    }
    let rows = operands.window_rows as usize;
    if rows == 0 || rows > WINDOW_ROWS {
        return Err(format!(
            "attention body: a {rows} row window; the entry reads 1..={WINDOW_ROWS} rows, the \
             current token included"
        ));
    }
    let selected = operands.selected_count as usize;
    if selected > MAX_SELECTED {
        return Err(format!(
            "attention body: {selected} selected ids, the union is sized for {MAX_SELECTED}"
        ));
    }
    let attended = rows
        .checked_add(selected)
        .ok_or_else(|| "attention body: the attended row count overflows".to_owned())?;
    if attended > MAX_ATTENDED {
        return Err(format!(
            "attention body: {rows} window rows plus {selected} selected rows is {attended}, the \
             entry's union holds {MAX_ATTENDED}"
        ));
    }
    let body_row = heads
        .checked_mul(operands.head_width as usize)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| "attention body: the head rows' byte width overflows".to_owned())?;
    if operands.q.len < body_row {
        return Err(format!(
            "attention body: workspace.q is {} B, {heads} heads of {} need {body_row} B",
            operands.q.len, operands.head_width
        ));
    }
    if operands.out.len < body_row {
        return Err(format!(
            "attention body: workspace.attn_body_bf16 is {} B, {heads} heads of {} need {body_row} B",
            operands.out.len, operands.head_width
        ));
    }
    let selected_bytes = selected
        .checked_mul(SELECTED_WORD_BYTES)
        .ok_or_else(|| "attention body: the selected list's byte width overflows".to_owned())?;
    if operands.selected.len < selected_bytes {
        return Err(format!(
            "attention body: the selected span is {} B, {selected} ids need {selected_bytes} B",
            operands.selected.len
        ));
    }
    let key_bytes = index_key_block_bytes(operands.position as usize)
        .map_err(|error| refused("attention body", &error))?;
    if operands.index_key.len < key_bytes {
        return Err(format!(
            "attention body: the index key pool is {} B, position {} reads {key_bytes} B",
            operands.index_key.len, operands.position
        ));
    }
    let query_bytes = INDEX_HEAD_DIM * 2;
    if operands.index_query.len < query_bytes {
        return Err(format!(
            "attention body: the index query span is {} B, index head 0 needs {query_bytes} B",
            operands.index_query.len
        ));
    }
    let sink_bytes = heads * 4;
    if operands.sink.len < sink_bytes {
        return Err(format!(
            "attention body: the sink span is {} B, {heads} heads need {sink_bytes} B",
            operands.sink.len
        ));
    }
    for (name, span) in [
        ("block scores", operands.block_scores),
        ("block best", operands.block_best),
    ] {
        let need = HSI_BLOCKS * 4;
        if span.len < need {
            return Err(format!(
                "attention body: the {name} span is {} B, the {HSI_BLOCKS} block slots need {need} B",
                span.len
            ));
        }
    }
    for (name, pointer) in [
        ("the window's codes", operands.window_codes),
        ("the window's scales", operands.window_scales),
        (
            "the compressed store's codes",
            Deviceptr(operands.kv_fp4.device),
        ),
        (
            "the compressed store's scales",
            Deviceptr(operands.kv_scales.device),
        ),
    ] {
        if pointer.0 == 0 {
            return Err(format!(
                "attention body: {name} is a null pointer; the sequence entry has no null branch \
                 for its operands"
            ));
        }
    }
    Ok(FlashMlaSeqParams::new(
        Deviceptr(operands.q.device),
        Deviceptr(operands.kv_fp4.device),
        Deviceptr(operands.kv_scales.device),
        operands.window_codes,
        operands.window_scales,
        FORMAT_CSA_UNION,
        u64::from(operands.window_rows),
        Deviceptr(operands.selected.device),
        Deviceptr(operands.index_key.device),
        Deviceptr(operands.index_query.device),
        Deviceptr(operands.sink.device),
        Deviceptr(operands.block_scores.device),
        Deviceptr(operands.block_best.device),
        Deviceptr(operands.out.device),
        u64::from(operands.position),
        operands.selected_count,
        operands.trace_ring,
        operands.layer_id,
        operands.step_seq,
    ))
}

/// Launches the sequence entry once over the per-rank heads.
///
/// The grid is `(heads_per_rank, 1, 1)` and the block `(256, 1, 1)`: the pinned launcher
/// (`flash_mla_csa2_seq_launch::launch`) is fixed at the world-1 grid `(64, 1, 1)` and
/// `launch_batched` is the six-position entry, so the per-rank call is this `cuda.launch` - the same
/// one the host arm makes.
///
/// # Errors
/// [`body_params`]'s refusals, or the driver's launch error.
pub fn launch_attention_body(
    cuda: &Cuda,
    kernel: Kernel,
    operands: &BodyOperands,
) -> Result<(), String> {
    let params = body_params(operands)?;
    let grid = (operands.heads_per_rank, 1, 1);
    let block = (THREADS, 1, 1);
    cuda.launch(kernel, grid, block, params.as_slice())
        .map_err(|error| format!("attention body: the sequence entry launch: {error}"))
}

/// The inverse-rope launch's argument words: the body's last launch on a window-only layer.
///
/// It is `rope_rotate_rows_bf16` with `sin_sign = -1`, which is bit-for-bit
/// `engine::linalg::rope_rotate_inverse` (the twin test in `layer_seams_ref` pins it). `src` and
/// `out` are the same row: each pair reads before it writes.
///
/// # Errors
/// The launcher's own geometry refusals (zero rows, a zero or odd rope tail, a tail wider than the
/// row, a sign that is not ±1).
#[allow(clippy::too_many_arguments)]
pub fn inverse_rope_params(
    row: Deviceptr,
    cos: Deviceptr,
    sin: Deviceptr,
    rows: u32,
    head_width: u32,
    rope_head_dim: u32,
    trace_ring: u64,
    layer_id: u32,
    step_seq: u64,
) -> Result<RopeRowsParams, String> {
    let params = RopeRowsParams {
        src: row,
        cos,
        sin,
        out: row,
        rows,
        head_width,
        rope_head_dim,
        sin_sign: -1,
        trace_ring,
        layer_id,
        step_seq,
    };
    params
        .validate()
        .map_err(|error| format!("attention body: the inverse rope: {error}"))?;
    Ok(params)
}

/// The forward-rope launch's argument words: the compressor latent's rotation.
///
/// It is the same `rope_rotate_rows_bf16` the body uses with `sin_sign = -1`, called with `sin_sign =
/// +1` - the forward rotation `engine::linalg::rope_rotate` performs. `src` and `out` are the same
/// row, and each pair reads before it writes, which is why the launcher can rotate in place; the
/// compressor path still passes distinct latitudes because its input is f32.
///
/// # Errors
/// The launcher's own geometry refusals (zero rows, a zero or odd rope tail, a tail wider than the
/// row, a sign that is not ±1).
#[allow(clippy::too_many_arguments)]
pub fn forward_rope_params(
    row: Deviceptr,
    cos: Deviceptr,
    sin: Deviceptr,
    rows: u32,
    head_width: u32,
    rope_head_dim: u32,
    trace_ring: u64,
    layer_id: u32,
    step_seq: u64,
) -> Result<RopeRowsParams, String> {
    let params = RopeRowsParams {
        src: row,
        cos,
        sin,
        out: row,
        rows,
        head_width,
        rope_head_dim,
        sin_sign: 1,
        trace_ring,
        layer_id,
        step_seq,
    };
    params
        .validate()
        .map_err(|error| format!("compressor latent: the forward rope: {error}"))?;
    Ok(params)
}

/// Launches the forward rope over the compressor's latent row.
///
/// # Errors
/// [`forward_rope_params`]'s refusals, or the launcher's driver error.
#[allow(clippy::too_many_arguments)]
pub fn launch_forward_rope(
    cuda: &Cuda,
    kernel: Kernel,
    row: Deviceptr,
    cos: Deviceptr,
    sin: Deviceptr,
    rows: u32,
    head_width: u32,
    rope_head_dim: u32,
    trace_ring: u64,
    layer_id: u32,
    step_seq: u64,
) -> Result<(), String> {
    let params = forward_rope_params(
        row,
        cos,
        sin,
        rows,
        head_width,
        rope_head_dim,
        trace_ring,
        layer_id,
        step_seq,
    )?;
    seams::launch_rope_rows(cuda, kernel, &params)
        .map_err(|error| format!("compressor latent: the forward rope launch: {error}"))
}

/// Launches the inverse rope over the body's per-rank head rows.
///
/// # Errors
/// [`inverse_rope_params`]'s refusals, or the launcher's driver error.
#[allow(clippy::too_many_arguments)]
pub fn launch_inverse_rope(
    cuda: &Cuda,
    kernel: Kernel,
    row: Deviceptr,
    cos: Deviceptr,
    sin: Deviceptr,
    rows: u32,
    head_width: u32,
    rope_head_dim: u32,
    trace_ring: u64,
    layer_id: u32,
    step_seq: u64,
) -> Result<(), String> {
    let params = inverse_rope_params(
        row,
        cos,
        sin,
        rows,
        head_width,
        rope_head_dim,
        trace_ring,
        layer_id,
        step_seq,
    )?;
    seams::launch_rope_rows(cuda, kernel, &params)
        .map_err(|error| format!("attention body: the inverse rope launch: {error}"))
}

/// The three launches of `attention_output`, per rank: wo_a, the wo_b activation quantiser, then
/// the wo_b GEMV. All device-to-device; the collective is stage 12's, not this stage's.
#[derive(Debug, Clone, Copy)]
pub struct OutputOperands {
    /// The body's rotated row: `heads_per_rank * head_width` bf16, wo_a's activation.
    pub attn_bf16: DeviceSpan,
    /// wo_a's output row: `groups * o_lora_rank` bf16.
    pub lo_out: DeviceSpan,
    /// The wo_b activation's E4M3 codes (`groups * o_lora_rank` bytes).
    pub act_codes: DeviceSpan,
    /// The wo_b activation's E8M0 scales (`ceil(groups * o_lora_rank / 32)` bytes).
    pub act_scales: DeviceSpan,
    /// The wo_b output row: `hidden` bf16, the all-reduce's input.
    pub wo_b_out: DeviceSpan,
    /// The plan's wo_a shard: `(weight, scales)` from the bind, never null.
    pub wo_a: (Deviceptr, Deviceptr),
    /// The plan's wo_b shard: `(weight, scales)` from the bind, never null.
    pub wo_b: (Deviceptr, Deviceptr),
    /// The plan's stored wo_a shard shape `(cols, inner)`, as the bind reports it.
    pub wo_a_shape: (usize, usize),
    /// The plan's stored wo_b shard shape `(cols, inner)`, as the bind reports it.
    pub wo_b_shape: (usize, usize),
    /// The per-rank geometry the three launches derive their shapes from.
    pub geometry: OutputGeometry,
    /// The trace ring's device address, or 0.
    pub trace_ring: u64,
    /// The layer id the trace events record.
    pub layer_id: u32,
    /// The step sequence the trace events record.
    pub step_seq: u64,
}

/// The checks all three `attention_output` launches share: the plan's shards, the row widths and
/// the weight pointers.
///
/// # Errors
/// A plan wo_a or wo_b shape that disagrees with the per-rank geometry, a null weight or scale
/// pointer, a row span too short for its consumers, or a wo_b activation width the quantiser cannot
/// express.
pub fn output_checks(operands: &OutputOperands) -> Result<(), String> {
    let geometry = &operands.geometry;
    geometry
        .check_wo_a(operands.wo_a_shape)
        .map_err(|error| refused("attention_output", &error))?;
    geometry
        .check_wo_b(operands.wo_b_shape)
        .map_err(|error| refused("attention_output", &error))?;
    geometry
        .check_activation_bytes(operands.attn_bf16.len)
        .map_err(|error| refused("attention_output", &error))?;
    for (name, pair) in [("wo_a", operands.wo_a), ("wo_b", operands.wo_b)] {
        if pair.0.0 == 0 || pair.1.0 == 0 {
            return Err(format!(
                "attention_output: {name} bound a null pointer; refusing to launch"
            ));
        }
    }
    for (name, span, need) in [
        ("workspace.attn_lo", operands.lo_out, geometry.lo_bytes()),
        (
            "workspace.act_codes",
            operands.act_codes,
            geometry.lo_code_bytes(),
        ),
        (
            "workspace.act_scales",
            operands.act_scales,
            geometry.lo_scale_bytes(),
        ),
        (
            "workspace.attn_out_bf16",
            operands.wo_b_out,
            geometry.output_bytes(),
        ),
    ] {
        if span.len < need {
            return Err(format!(
                "attention_output: {name} is {} B, the launch needs {need} B",
                span.len
            ));
        }
    }
    if !geometry.prologue_cols().is_multiple_of(32) {
        return Err(format!(
            "attention_output: the wo_b activation is {} values, the quantiser's 32-value blocks \
             need a multiple of 32",
            geometry.prologue_cols()
        ));
    }
    Ok(())
}

/// The wo_a launch's 8 words.
///
/// # Errors
/// [`output_checks`]'s refusals, a per-rank group count that does not fit the ABI's u32, or a
/// hidden width that does not fit the wo_b GEMV's u32 columns.
pub fn wo_a_params(operands: &OutputOperands) -> Result<Bf16aMxfp8wWoAParams, String> {
    output_checks(operands)?;
    let groups = u32::try_from(operands.geometry.groups).map_err(|_| {
        format!(
            "attention_output: {} groups do not fit u32",
            operands.geometry.groups
        )
    })?;
    Ok(Bf16aMxfp8wWoAParams::new(
        Deviceptr(operands.attn_bf16.device),
        operands.wo_a.0,
        operands.wo_a.1,
        Deviceptr(operands.lo_out.device),
        groups,
        operands.trace_ring,
        operands.layer_id,
        operands.step_seq,
    ))
}

/// The wo_b activation quantiser's 8 words: one row of `groups * o_lora_rank` values.
///
/// # Errors
/// [`output_checks`]'s refusals, or a column count that does not fit the ABI's u32.
pub fn prologue_bf16_params(operands: &OutputOperands) -> Result<PrologueBf16Params, String> {
    output_checks(operands)?;
    let cols = u32::try_from(operands.geometry.prologue_cols()).map_err(|_| {
        format!(
            "attention_output: {} activation columns do not fit u32",
            operands.geometry.prologue_cols()
        )
    })?;
    Ok(PrologueBf16Params {
        normed: Deviceptr(operands.lo_out.device),
        codes: Deviceptr(operands.act_codes.device),
        scales: Deviceptr(operands.act_scales.device),
        cols,
        rows: 1,
        trace_ring: operands.trace_ring,
        layer_id: operands.layer_id,
        step_seq: operands.step_seq,
    })
}

/// The wo_b decode GEMV's 11 words: one row of `hidden` columns over a `groups * o_lora_rank`
/// reduction.
///
/// # Errors
/// [`output_checks`]'s refusals, a hidden width or reduction that does not fit the ABI's u32, or a
/// reduction the decode GEMV cannot hold.
pub fn gemv_params(operands: &OutputOperands) -> Result<Mxfp8DenseParams, String> {
    output_checks(operands)?;
    let geometry = &operands.geometry;
    let cols = u32::try_from(geometry.hidden).map_err(|_| {
        format!(
            "attention_output: {} columns do not fit u32",
            geometry.hidden
        )
    })?;
    let inner = u32::try_from(geometry.groups * geometry.o_lora_rank).map_err(|_| {
        format!(
            "attention_output: the {} value reduction does not fit u32",
            geometry.groups * geometry.o_lora_rank
        )
    })?;
    if inner > mxfp8_gemv_launch::MAX_INNER {
        return Err(format!(
            "attention_output: the reduction is {inner} values, the decode GEMV holds {}",
            mxfp8_gemv_launch::MAX_INNER
        ));
    }
    Ok(Mxfp8DenseParams::new(
        Deviceptr(operands.act_codes.device),
        Deviceptr(operands.act_scales.device),
        operands.wo_b.0,
        operands.wo_b.1,
        Deviceptr(operands.wo_b_out.device),
        1,
        cols,
        inner,
        operands.trace_ring,
        operands.layer_id,
        operands.step_seq,
    ))
}

/// Launches the three `attention_output` kernels in order, device to device.
///
/// `wo_a_kernel` is `b12x_bf16a_mxfp8w_wo_a_sm121`, `prologue_kernel` is
/// `w8a8_prologue_bf16_e4m3` (the seam image's `prologue_bf16` symbol) and `gemv_kernel` is the
/// decode GEMV (`b12x_mxfp8_gemv_m1_sm121`). The wo_b weight and its scale grid are the TPM-sharded
/// plan bytes, so the launch produces **this rank's partial** row; stage 12's collective owns the
/// cross-rank reduce and stage 13's seam owns the residual.
///
/// # Errors
/// [`output_checks`]'s refusals, the launchers' geometry refusals, or a driver error.
pub fn launch_attention_output(
    cuda: &Cuda,
    wo_a_kernel: Kernel,
    prologue_kernel: Kernel,
    gemv_kernel: Kernel,
    operands: &OutputOperands,
) -> Result<(), String> {
    output_checks(operands)?;
    let wo_a = wo_a_params(operands)?;
    wo_a_launch::launch(cuda, wo_a_kernel, &wo_a)
        .map_err(|error| format!("attention_output: the wo_a launch: {error}"))?;
    let prologue = prologue_bf16_params(operands)?;
    seams::launch_prologue_bf16(cuda, prologue_kernel, &prologue)
        .map_err(|error| format!("attention_output: the wo_b quantiser launch: {error}"))?;
    let gemv = gemv_params(operands)?;
    mxfp8_gemv_launch::launch(cuda, gemv_kernel, &gemv)
        .map_err(|error| format!("attention_output: the wo_b GEMV launch: {error}"))
}

/// Formats one operand refusal with the stage it stopped, so a stage refusal reads as one sentence
/// in the arm's frontier record.
fn refused(stage: &str, error: &OperandError) -> String {
    format!("{stage}: {error}")
}

/// The device cache's published rows at a position: `(p + 1) / ratio`, or `None` at ratio 0.
///
/// A helper for the wiring, not a stage: the body never reads the count back, it derives it here.
///
/// # Errors
/// [OperandError::RatioZero] is impossible (it returns `None`); [OperandError::RowsPastStore] when
/// the position publishes past the expanded store.
pub fn store_rows_at(ratio: usize, position: usize) -> Result<Option<usize>, String> {
    if ratio == 0 {
        return Ok(None);
    }
    let operands =
        StoreOperands::new(ratio, position).map_err(|error| refused("attention store", &error))?;
    if operands.rows_written > STORE_POSITIONS {
        return Err(format!(
            "attention store: {} rows at position {position} past the {STORE_POSITIONS}-row store",
            operands.rows_written
        ));
    }
    Ok(Some(operands.rows_written))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_kernels::attention_operands::{
        OutputGeometry, STORE_CODE_BYTES, STORE_SCALE_BYTES,
    };
    use sparkflash_kernels::mxfp8_dense_launch;

    fn span(device: u64, len: usize) -> DeviceSpan {
        DeviceSpan::new(device, len)
    }

    fn released_geometry() -> OutputGeometry {
        OutputGeometry::new(8, 4, 16, 512, 1024, 5120).expect("the released per-rank geometry")
    }

    fn body() -> BodyOperands {
        BodyOperands {
            q: span(0x1000, 16_384),
            kv_fp4: span(0x2000, STORE_CODE_BYTES),
            kv_scales: span(0x3000, STORE_SCALE_BYTES),
            window_codes: Deviceptr(0x4000),
            window_scales: Deviceptr(0x5000),
            window_rows: 1,
            selected: span(0x6000, 512 * 4),
            selected_count: 0,
            index_key: span(0x7000, 4_194_304),
            index_query: span(0x8000, 8192),
            sink: span(0x9000, 64),
            block_scores: span(0xa000, 8192),
            block_best: span(0xb000, 8192),
            out: span(0xc000, 16_384),
            position: 0,
            heads_per_rank: 16,
            head_width: 512,
            trace_ring: 0,
            layer_id: 0,
            step_seq: 0,
        }
    }

    fn output() -> OutputOperands {
        let geometry = released_geometry();
        OutputOperands {
            attn_bf16: span(0x1000, geometry.activation_bytes()),
            lo_out: span(0x2000, geometry.lo_bytes()),
            act_codes: span(0x3000, geometry.lo_code_bytes()),
            act_scales: span(0x4000, geometry.lo_scale_bytes()),
            wo_b_out: span(0x5000, geometry.output_bytes()),
            wo_a: (Deviceptr(0x6000), Deviceptr(0x7000)),
            wo_b: (Deviceptr(0x8000), Deviceptr(0x9000)),
            wo_a_shape: geometry.wo_a_shape(),
            wo_b_shape: geometry.wo_b_shape(),
            geometry,
            trace_ring: 0,
            layer_id: 3,
            step_seq: 7,
        }
    }

    #[test]
    fn reference_seam_on_a_window_only_layer_is_a_pure_window_decision() {
        let state = reference_seam(0, 0, 512, 2048).expect("position 0");
        assert_eq!(
            (
                state.window.window_rows,
                state.window.window_start,
                state.window.slot
            ),
            (1, 0, 0),
            "the current token is in the window"
        );
        assert_eq!((state.selected_count, state.store_rows), (0, 0));
        assert!(
            state.store_zero,
            "a window-only layer publishes no store row"
        );

        let short = reference_seam(0, 127, 512, 2048).expect("position 127");
        assert_eq!(
            (short.window.window_rows, short.window.window_start),
            (128, 0)
        );
        let full = reference_seam(0, 300, 512, 2048).expect("position 300");
        assert_eq!(
            (full.window.window_rows, full.window.window_start),
            (128, 173),
            "the window is the last 128 rows including the current one"
        );
    }

    #[test]
    fn reference_seam_refuses_a_compressed_layer_by_name() {
        let error = reference_seam(2, 0, 512, 2048).expect_err("ratio 2 is blocked");
        assert!(
            error.contains("compress") && error.contains("compressor"),
            "the refusal names the missing producer: {error}"
        );
        assert!(error.contains("19"), "and the class size: {error}");
    }

    #[test]
    fn reference_seam_refuses_a_position_whose_row_is_not_in_the_ring() {
        let error = reference_seam(0, 512, 512, 2048).expect_err("position == ring rows");
        assert!(error.contains("512"), "{error}");
        let error = reference_seam(0, 0, 0, 2048).expect_err("no ring rows");
        assert!(error.contains("zero"), "{error}");
        let error = reference_seam(0, 0, 2048, 512).expect_err("ring past the budget");
        assert!(error.contains("2048") && error.contains("512"), "{error}");
    }

    #[test]
    fn ref_select_at_ratio_zero_is_a_no_op_and_refuses_a_compressed_layer() {
        let state = reference_seam(0, 0, 512, 2048).expect("position 0");
        ref_select(0, &state).expect("ratio 0 selects nothing");
        let error = ref_select(2, &state).expect_err("ratio 2 is blocked");
        assert!(error.contains("weights_proj"), "{error}");
    }

    #[test]
    fn ref_select_refuses_a_selected_list_on_a_window_only_layer() {
        let mut state = reference_seam(0, 0, 512, 2048).expect("position 0");
        state.selected_count = 4;
        let error = ref_select(0, &state).expect_err("no compressed positions exist");
        assert!(error.contains('4'), "{error}");
    }

    #[test]
    fn stage_attention_accepts_the_layer_zero_state_at_every_window_size() {
        for position in [0, 1, 127, 128, 300, 511] {
            let state = reference_seam(0, position, 512, 2048).expect("position in the ring");
            stage_attention(&state, 512, 2048).expect("the validation passes");
            assert_eq!(state.attended().expect("bounded"), state.window.window_rows);
        }
    }

    #[test]
    fn stage_attention_refuses_a_capacity_that_disagrees_with_the_step() {
        let state = reference_seam(0, 0, 512, 2048).expect("position 0");
        let error = stage_attention(&state, 2048, 2048).expect_err("different ring");
        assert!(error.contains("512") && error.contains("2048"), "{error}");
    }

    #[test]
    fn stage_attention_refuses_a_ring_wider_than_the_position_budget() {
        let state = reference_seam(0, 0, 2048, 2048).expect("position 0 in a 2048-row ring");
        let error = stage_attention(&state, 2048, 512).expect_err("ring past the budget");
        assert!(error.contains("2048") && error.contains("512"), "{error}");
    }

    #[test]
    fn stage_attention_refuses_a_non_empty_store_on_a_window_only_layer() {
        let mut state = reference_seam(0, 0, 512, 2048).expect("position 0");
        state.store_rows = 3;
        state.store_zero = false;
        let error = stage_attention(&state, 512, 2048).expect_err("no compressed row exists");
        assert!(error.contains('3'), "{error}");
    }

    #[test]
    fn stage_attention_checks_the_compressed_store_cursor() {
        let mut state = reference_seam(0, 3, 512, 2048).expect("position 3");
        state.ratio = 2;
        stage_attention(&state, 512, 2048).expect_err("a ratio-2 position has two rows");
        state.store_rows = 2;
        state.store_zero = false;
        stage_attention(&state, 512, 2048).expect("position 3 publishes two ratio-2 rows");
    }

    #[test]
    fn stage_attention_refuses_a_selected_list_past_the_topk() {
        let mut state = reference_seam(0, 0, 512, 2048).expect("position 0");
        // A crafted state, not one the ratio-0 stages build: the refusal is still exercised. With
        // window <= 128 the union bound (640) cannot be exceeded without also exceeding the top-k,
        // so the top-k refusal is the reachable one; the union check stays as defence in depth.
        state.selected_count = 513;
        let error = stage_attention(&state, 512, 2048).expect_err("past the top-k");
        assert!(error.contains("past") && error.contains("512"), "{error}");
    }

    #[test]
    fn body_words_are_the_pinned_nineteen() {
        let operands = body();
        let params = body_params(&operands).expect("the layer-0 operand set");
        let words = params.as_slice();
        assert_eq!(
            words.len(),
            sparkflash_kernels::flash_mla_csa2_seq_launch::ARG_WORDS,
            "the entry's ABI is 19 words"
        );
        assert_eq!(
            words,
            [
                0x1000u64,
                0x2000,
                0x3000,
                0x4000,
                0x5000,
                FORMAT_CSA_UNION,
                1,
                0x6000,
                0x7000,
                0x8000,
                0x9000,
                0xa000,
                0xb000,
                0xc000,
                0,
                0,
                0,
                0,
                0,
            ],
            "the words are the declaration's order: selected_count after position"
        );
        assert_eq!(FORMAT_CSA_UNION, 2);
    }

    #[test]
    fn body_words_follow_the_selected_count_and_trace_tail() {
        let mut operands = body();
        operands.selected_count = 512;
        operands.window_rows = 1;
        operands.trace_ring = 0x7777;
        operands.layer_id = 5;
        operands.step_seq = 9;
        let params = body_params(&operands).expect("one window row plus 512 selected rows");
        let words = params.as_slice();
        assert_eq!(
            &words[15..],
            &[
                u64::from(operands.selected_count),
                0x7777,
                u64::from(operands.layer_id),
                operands.step_seq
            ],
            "selected_count is word 15, then the trace trio"
        );
    }

    #[test]
    fn body_refuses_a_window_outside_the_kernels_bound() {
        let mut operands = body();
        operands.window_rows = 0;
        let error = body_params(&operands).expect_err("no window row");
        assert!(error.contains('0'), "{error}");
        operands.window_rows = 129;
        let error = body_params(&operands).expect_err("129 rows");
        assert!(error.contains("129") && error.contains("128"), "{error}");
    }

    #[test]
    fn body_accepts_the_exact_union_and_refuses_one_more_selected_id() {
        let mut operands = body();
        operands.window_rows = 128;
        operands.selected_count = 512;
        body_params(&operands).expect("128 + 512 is the exact 640-row union bound");
        operands.selected_count = 513;
        let error = body_params(&operands).expect_err("past the top-k");
        assert!(error.contains("513") && error.contains("512"), "{error}");
    }

    #[test]
    fn body_refuses_thin_spans_row_by_row() {
        let mut operands = body();
        operands.q.len = 16_383;
        let error = body_params(&operands).expect_err("short q");
        assert!(error.contains("workspace.q"), "{error}");
        operands = body();
        operands.out.len = 100;
        let error = body_params(&operands).expect_err("short out");
        assert!(error.contains("attn_body_bf16"), "{error}");
        operands = body();
        operands.selected_count = 8;
        operands.selected.len = 31;
        let error = body_params(&operands).expect_err("short selected");
        assert!(error.contains("selected"), "{error}");
        operands = body();
        operands.sink.len = 63;
        let error = body_params(&operands).expect_err("short sink");
        assert!(error.contains("sink"), "{error}");
        operands = body();
        operands.block_scores.len = 8191;
        let error = body_params(&operands).expect_err("short block scores");
        assert!(error.contains("block scores"), "{error}");
        operands = body();
        operands.block_best.len = 0;
        let error = body_params(&operands).expect_err("empty block best");
        assert!(error.contains("block best"), "{error}");
    }

    #[test]
    fn body_refuses_an_index_pool_that_the_position_would_read_past() {
        let mut operands = body();
        operands.index_key.len = 256;
        let error = body_params(&operands).expect_err("one block is not enough at position 0");
        assert!(error.contains("256"), "{error}");
        operands.position = 16_383;
        operands.index_key.len = body().index_key.len;
        body_params(&operands).expect("the 2048-block pool covers the last whole block");
        operands.position = 16_384;
        let error = body_params(&operands).expect_err("past the pool's 2048 blocks");
        assert!(error.contains("16384") || error.contains("2049"), "{error}");
    }

    #[test]
    fn body_refuses_a_null_operand() {
        let mut operands = body();
        operands.kv_fp4 = span(0, STORE_CODE_BYTES);
        let error = body_params(&operands).expect_err("null store codes");
        assert!(error.contains("null"), "{error}");
        operands = body();
        operands.window_codes = Deviceptr(0);
        let error = body_params(&operands).expect_err("null window codes");
        assert!(error.contains("null"), "{error}");
    }

    #[test]
    fn body_refuses_a_head_count_the_fixed_grid_does_not_serve() {
        let mut operands = body();
        operands.heads_per_rank = 0;
        let error = body_params(&operands).expect_err("no heads");
        assert!(error.contains("no query heads"), "{error}");
        operands.heads_per_rank = 65;
        let error = body_params(&operands).expect_err("past the fixed grid");
        assert!(error.contains("64"), "{error}");
    }

    #[test]
    fn inverse_rope_words_carry_the_inverse_sign() {
        let params = inverse_rope_params(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            16,
            512,
            64,
            0,
            3,
            7,
        )
        .expect("the released geometry");
        let words = params.words();
        assert_eq!(words.len(), 11, "the rope row ABI is 11 words");
        assert_eq!(
            words,
            [
                0x1000,
                0x2000,
                0x3000,
                0x1000,
                16,
                512,
                64,
                u64::MAX,
                0,
                3,
                7
            ],
            "src, cos, sin, out == src, rows, head_width, rope_head_dim, -1, trace"
        );
        assert_eq!(
            words[7],
            u64::MAX,
            "sin_sign = -1 is the inverse rotation, stored as i64 then u64"
        );
    }

    #[test]
    fn inverse_rope_refuses_geometry_the_kernel_cannot_rotate() {
        let bad = [
            (0u32, 512u32, 64u32),
            (16, 512, 0),
            (16, 512, 63),
            (16, 512, 520),
        ];
        for (rows, head_width, rope_head_dim) in bad {
            let error = inverse_rope_params(
                Deviceptr(0x1000),
                Deviceptr(0x2000),
                Deviceptr(0x3000),
                rows,
                head_width,
                rope_head_dim,
                0,
                0,
                0,
            )
            .expect_err("geometry refused");
            assert!(!error.is_empty(), "each refusal names its numbers: {error}");
        }
    }

    #[test]
    fn output_words_are_the_pinned_eight_eight_and_eleven() {
        let operands = output();
        let wo_a = wo_a_params(&operands).expect("the released wo_a");
        assert_eq!(wo_a.as_slice().len(), wo_a_launch::ARG_WORDS);
        assert_eq!(
            wo_a.as_slice(),
            [0x1000u64, 0x6000, 0x7000, 0x2000, 2, 0, 3, 7],
            "attn_bf16, weight, scales, lo_out, groups = 2, trace, layer, step"
        );
        assert_eq!(wo_a.groups(), 2, "the per-rank group count at world 4");

        let prologue = prologue_bf16_params(&operands).expect("the released prologue");
        assert_eq!(
            prologue.words(),
            [0x2000u64, 0x3000, 0x4000, 2_048, 1, 0, 3, 7],
            "lo row, codes, scales, cols = 2048, rows = 1, trace"
        );

        let gemv = gemv_params(&operands).expect("the released wo_b");
        assert_eq!(gemv.as_slice().len(), mxfp8_dense_launch::ARG_WORDS);
        assert_eq!(
            gemv.as_slice(),
            [
                0x3000u64, 0x4000, 0x8000, 0x9000, 0x5000, 1, 5_120, 2_048, 0, 3, 7
            ],
            "codes, scales, weight, weight scales, out, rows = 1, cols = 5120, inner = 2048"
        );
    }

    #[test]
    fn output_refuses_a_plan_shard_that_disagrees_with_the_per_rank_geometry() {
        let mut operands = output();
        operands.wo_a_shape = (1024, 8192);
        let error = output_checks(&operands).expect_err("the world-1 wo_a shard");
        assert!(error.contains("wo_a"), "{error}");
        operands = output();
        operands.wo_b_shape = (5120, 1024);
        let error = output_checks(&operands).expect_err("the world-1 wo_b shard");
        assert!(error.contains("wo_b"), "{error}");
    }

    #[test]
    fn output_refuses_a_body_row_that_is_not_the_per_rank_heads() {
        let mut operands = output();
        operands.attn_bf16.len = 16_383;
        let error = output_checks(&operands).expect_err("short activation");
        assert!(
            error.contains("16383") && error.contains("16384"),
            "{error}"
        );
    }

    #[test]
    fn output_refuses_a_null_weight_pair_and_a_thin_row() {
        let mut operands = output();
        operands.wo_a = (Deviceptr(0), Deviceptr(0x7000));
        let error = output_checks(&operands).expect_err("null wo_a");
        assert!(error.contains("null"), "{error}");
        operands = output();
        operands.lo_out.len = 4_095;
        let error = output_checks(&operands).expect_err("short lo row");
        assert!(error.contains("attn_lo"), "{error}");
        operands = output();
        operands.act_scales.len = 63;
        let error = output_checks(&operands).expect_err("short scale row");
        assert!(error.contains("act_scales"), "{error}");
        operands = output();
        operands.wo_b_out.len = 10_239;
        let error = output_checks(&operands).expect_err("short wo_b row");
        assert!(error.contains("attn_out_bf16"), "{error}");
    }

    #[test]
    fn output_refuses_a_reduction_the_decode_gemv_cannot_hold() {
        // A per-rank geometry whose reduction (`groups * o_lora_rank` = 2 x 3104 = 6208) is past the
        // GEMV's 5120-value shared-memory bound, with a quantiser width that is still a whole
        // number of 32-value blocks so the earlier checks pass: the refusal must be the inner one.
        let geometry =
            OutputGeometry::new(8, 4, 16, 512, 3104, 5120).expect("a wide-rank geometry");
        let mut operands = output();
        operands.attn_bf16 = span(0x1000, geometry.activation_bytes());
        operands.lo_out = span(0x2000, geometry.lo_bytes());
        operands.act_codes = span(0x3000, geometry.lo_code_bytes());
        operands.act_scales = span(0x4000, geometry.lo_scale_bytes());
        operands.wo_b_out = span(0x5000, geometry.output_bytes());
        operands.wo_a_shape = geometry.wo_a_shape();
        operands.wo_b_shape = geometry.wo_b_shape();
        operands.geometry = geometry;
        let error = gemv_params(&operands).expect_err("6208 is past the GEMV's bound");
        assert!(error.contains("6208"), "{error}");
    }

    #[test]
    fn store_rows_are_derived_from_the_position() {
        assert_eq!(store_rows_at(0, 300).expect("ratio 0"), None);
        assert_eq!(store_rows_at(1, 0).expect("ratio 1"), Some(1));
        assert_eq!(store_rows_at(2, 0).expect("ratio 2"), Some(0));
        assert_eq!(store_rows_at(2, 1).expect("ratio 2"), Some(1));
        assert_eq!(store_rows_at(2, 512).expect("ratio 2"), Some(256));
        let error = store_rows_at(2, 513).expect_err("ratio 2 publishes row 512 at 513");
        assert!(error.contains("store") && error.contains("257"), "{error}");
    }
}
