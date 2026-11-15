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

//! The chunked prefill: [crate::device_plan::Pass::Prefill] executed over a row count.
//!
//! The schedule is the decode one; only the width differs. Every dense projection takes the
//! token image's tensor-core tile over the chunk's rows, the attention entry reads the prefix
//! the step's own cache stores wrote, the four short convolutions read their taps across the
//! chunk, and the routed band is launched once per expert over the rows that picked it.
//!
//! Two things a chunk cannot simply widen:
//!
//! * **The convolution's seam.** A row's filter reads the three rows before it, and for a
//!   chunk's first rows those are the rows the previous launch ended on. The four
//!   convolution inputs therefore carry three history rows in front of their first row, filled
//!   by [crate::device_plan]'s companion kernel [Shared::conv_history] from the decode
//!   history, and written back the same way. A cold start's history is zeros, which is the
//!   release's "no such sample", so one mechanism serves both cases.
//! * **The block-summary stack.** The stack is a token's residual chain, not a history: every
//!   row of a chunk builds its own from layer zero. So the stack is indexed by row here, one
//!   stack a row, and the boundary layer of every row pushes into it in one strided launch.
//!
//! The arithmetic is otherwise the row-at-a-time walk's: the schedules' steps are in the same
//! order, the same kernels compute the same sums, and the seam above is a copy of the numbers
//! rather than a re-derivation of them. That is what makes "the chunked prefill and the
//! row-at-a-time one pick the same token" a test rather than a hope.

use sparkflash_model::backend::BackendError;

use crate::device_plan::{BandOp, Collective, ConvSlot, ExpertBody, LayerSchedule, Step};

use super::*;

/// A device refusal from any error the driver or a kernel reports.
pub(super) fn device_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::new(super::DEVICE, error.to_string())
}

/// One all-reduce of an f32 buffer, in place.
///
/// With a device leg the partials are exchanged into the peers' slots and summed on the stream
/// by q4e_sum_slots; without one the buffer makes a staged round trip through the host exchange.
/// Both sum in ascending rank order from zero, so the two legs give the same bits.
pub(super) fn exchange_sum(
    shared: &Shared,
    pointer: Deviceptr,
    elements: usize,
) -> Result<(), BackendError> {
    let world = shared.exchange.world();
    if world == 1 {
        return Ok(());
    }
    if let Some(slots) = shared.exchange_slots {
        let widest = (shared.prefill_chunk * shared.hidden()).max(shared.plan.rank_vocab());
        if elements > widest || !elements.is_multiple_of(4) {
            return Err(device_error(format!(
                "a device-leg sum of {elements} values: the slots hold {widest} a rank, \
                 in whole groups of four"
            )));
        }
        let bytes = elements * std::mem::size_of::<f32>();
        let rank = shared.exchange.rank();
        let sum = |own: u64, count: usize| {
            let grid = (count / 4).div_ceil(SUM_BLOCK as usize) as u32;
            shared
                .launch(
                    shared.kernels.sum_slots,
                    (grid, 1, 1),
                    (SUM_BLOCK, 1, 1),
                    &[own, slots.0, count as u64, rank as u64, world as u64],
                )
                .map_err(device_error)
        };
        if bytes >= TWO_SHOT_FROM && elements.is_multiple_of(4 * world) {
            // Two shots: this rank sums chunk `rank` (every peer's piece of it lands in the
            // slots), then the summed chunks are shared in place. Each element is the same fold
            // in the same order, so the bits are the one-shot's at half the bytes a rank.
            let chunk = elements / world;
            let mine = pointer.0 + (rank * chunk * std::mem::size_of::<f32>()) as u64;
            shared
                .exchange
                .scatter_on_device(0, pointer.0, slots.0, bytes)
                .map_err(device_error)?;
            sum(mine, chunk)?;
            return shared
                .exchange
                .exchange_on_device(0, mine, pointer.0, chunk * std::mem::size_of::<f32>())
                .map_err(device_error);
        }
        shared
            .exchange
            .exchange_on_device(0, pointer.0, slots.0, bytes)
            .map_err(device_error)?;
        return sum(pointer.0, elements);
    }
    let mut bytes = row_bytes(&vec![0f32; elements]);
    shared
        .cuda
        .copy_dtoh(&mut bytes, &pointer)
        .map_err(device_error)?;
    let mut values = from_bytes(&bytes);
    shared
        .exchange
        .all_reduce_f32(&mut values)
        .map_err(device_error)?;
    shared
        .cuda
        .copy_htod(&pointer, &row_bytes(&values))
        .map_err(device_error)?;
    Ok(())
}

/// The block q4e_sum_slots is written for (the token image's Q4E_BLOCK).
const SUM_BLOCK: u32 = 256;

/// The narrowest sum that takes two shots rather than one: about 57 rows of the residual. Below
/// it the second shot's launch costs more than the bytes it saves, and the decode step's sums
/// (one row) stay on the one-shot exchange the direct-RDMA leg carries.
const TWO_SHOT_FROM: usize = 1 << 20;

/// One all-gather of each rank's `width` f32 slice into `gathered`, rank-major.
///
/// The device leg copies this rank's slice into its own slot and exchanges it into the peers';
/// the host leg is a staged round trip. Either moves the values unchanged.
pub(super) fn exchange_gather(
    shared: &Shared,
    slice: Deviceptr,
    gathered: Deviceptr,
    width: usize,
) -> Result<(), BackendError> {
    let world = shared.exchange.world();
    let bytes_a_rank = width * std::mem::size_of::<f32>();
    if world == 1 || shared.exchange.has_device_leg() {
        let own = Deviceptr(gathered.0 + (shared.exchange.rank() * bytes_a_rank) as u64);
        shared
            .cuda
            .copy_dtod(&own, &slice, bytes_a_rank)
            .map_err(device_error)?;
        if world == 1 {
            return Ok(());
        }
        return shared
            .exchange
            .exchange_on_device(0, slice.0, gathered.0, bytes_a_rank)
            .map_err(device_error);
    }
    let mut bytes = row_bytes(&vec![0f32; width]);
    shared
        .cuda
        .copy_dtoh(&mut bytes, &slice)
        .map_err(device_error)?;
    let mut values = vec![0f32; width * world];
    shared
        .exchange
        .all_gather_f32(&from_bytes(&bytes), &mut values)
        .map_err(device_error)?;
    shared
        .cuda
        .copy_htod(&gathered, &row_bytes(&values))
        .map_err(device_error)
}

/// The bytes of an i32 vector, little-endian.
fn word_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// One dense projection of the chunk's rows.
///
/// The tensor walk takes the wide tile for every projection whose shape it reads (all but the
/// twelve-row attention sink and the convolution taps, which stay on the row GEMV at any walk).
/// The chunk walk's batched GEMV is the row kernel's own, so a row of a chunk is the bits a
/// row-at-a-time step would produce; the reordered tiled kernel is the comparison the note's
/// table quotes. The GEMV takes the chunk's groups of eight in one launch along the grid's y:
/// those two projections are one or two blocks wide, and a launch a group left a 4,096-row chunk
/// 512 serial launches a layer on them - 8.5 s of a 9.5 s tensor prefill.
#[allow(clippy::too_many_arguments)]
fn project(
    shared: &Shared,
    walk: PrefillWalk,
    weight: Deviceptr,
    input: Deviceptr,
    out: Deviceptr,
    width: usize,
    inner: usize,
    rows: usize,
) -> Result<(), BackendError> {
    if walk == PrefillWalk::Gemm {
        return shared
            .gemm(weight, input, out, width, inner, rows)
            .map_err(device_error);
    }
    if Shared::wide_fits(walk, width, inner) {
        return shared
            .wide(weight, input, out, width, inner, rows)
            .map_err(device_error);
    }
    // The whole groups of eight in one launch along the grid's y, then the remainder: each group
    // is the walk a launch of its own would run, so the split moves no bit.
    let byte = std::mem::size_of::<f32>();
    let groups = rows / BATCH_ROWS;
    if groups > 0 {
        shared
            .gemv_batch(weight, input, out, width, inner, BATCH_ROWS, groups)
            .map_err(device_error)?;
    }
    let base = groups * BATCH_ROWS;
    if base < rows {
        shared
            .gemv_batch(
                weight,
                input.offset(base * inner * byte),
                out.offset(base * width * byte),
                width,
                inner,
                rows - base,
                1,
            )
            .map_err(device_error)?;
    }
    Ok(())
}

/// Read one row's root-mean-square and first two elements, for the trace.
///
/// A debug read: it synchronises, and only runs when the trace is on. The row-at-a-time walk
/// prints the same three numbers one row per token, which is what makes the two paths' traces
/// comparable line for line.
fn probe(
    shared: &Shared,
    label: &str,
    base: Deviceptr,
    rows: usize,
    stride: usize,
) -> Result<(), BackendError> {
    if !trace_on() {
        return Ok(());
    }
    let mut bytes = vec![0u8; rows * stride * std::mem::size_of::<f32>()];
    shared
        .cuda
        .copy_dtoh(&mut bytes, &base)
        .map_err(device_error)?;
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    for row in 0..rows {
        let slice = &values[row * stride..(row + 1) * stride];
        let rms = (slice.iter().map(|value| value * value).sum::<f32>() / stride as f32).sqrt();
        eprintln!(
            "[inkling] {label} row {row} rms={rms:.5} first={:.5} second={:.5}",
            slice[0], slice[1]
        );
    }
    Ok(())
}

/// Read the band buffers of the rows this launch owns, for the trace.
///
/// The band is indexed by the (row, band slot) pair the router selected, so the pairs are the
/// only way to name a row's own slice of it.
fn probe_band(
    shared: &Shared,
    label: &str,
    base: Deviceptr,
    entries: &[i32],
    stride: usize,
) -> Result<(), BackendError> {
    if !trace_on() {
        return Ok(());
    }
    for entry in 0..entries.len() / 2 {
        if entries[2 * entry] != 0 {
            continue;
        }
        let slot = entries[2 * entry + 1] as usize;
        let at = base.offset(slot * stride * std::mem::size_of::<f32>());
        probe(shared, &format!("{label} slot {slot}"), at, 1, stride)?;
    }
    Ok(())
}

/// Whose rows one walk of the chunk schedule runs.
///
/// The schedule's wide stages (the norms, every projection, the router, the routed band, the
/// residuals, the block-summary stacks and the collectives) read a row as a row, whoever's it
/// is. Only four stages read a sequence's own state - the four convolutions' histories, the cache
/// store, the attention and the position its scale is taken at - and those are what this names.
pub(super) enum Rows<'s, 'a> {
    /// A prompt chunk: consecutive positions of one sequence, from its next position on.
    Chunk(&'s mut State),
    /// Fused lanes: the next position of each lane's own sequence, one row a lane in the order
    /// given. The sequence stages run each lane's row through the decode step's own launch on
    /// that lane's state, so a lane's row is the bits its decode step alone would give it.
    Lanes(&'s mut [&'a mut State]),
}

/// Rows of a chunk the text model takes from an image rather than from the embedding table.
///
/// The release replaces each image placeholder's embedding *after* the embedding norm, so the
/// walk copies `count` f32 rows from `source` over chunk rows `row..row + count` once the
/// prologue has normalised them; every rank holds the same rows (the tower's rank broadcast them
/// through the group's sum), so the replacement is the same on every rank.
#[derive(Debug, Clone, Copy)]
pub(super) struct Injection {
    /// The chunk's first row the run replaces.
    pub row: usize,
    /// The first of the run's rows, `hidden` f32 each.
    pub source: Deviceptr,
    /// The rows in the run.
    pub count: usize,
}

/// Run the chunk schedule over `tokens` and answer the logits: the last row's for a chunk, every
/// row's for fused lanes.
///
/// `walk` picks the dense projections' and the band's kernels: a prompt passes this rank's
/// [PrefillWalk], and fused lanes pass [PrefillWalk::Chunk], whose sums are the decode step's.
/// `images` are the chunk's image rows ([Injection]); fused lanes step text and pass none.
pub(super) fn walk_rows(
    shared: &Shared,
    mut sequences: Rows<'_, '_>,
    chunk: &mut Chunk,
    tokens: &[u32],
    walk: PrefillWalk,
    images: &[Injection],
) -> Result<Vec<Vec<f32>>, BackendError> {
    let config = &shared.config;
    let plan = &shared.plan;
    let schedule = shared.prefill.as_ref().ok_or_else(|| {
        device_error("this rank has no chunk schedule: rebuild with the prefill entries")
    })?;
    let rows = tokens.len();
    if rows == 0 {
        return Err(device_error("a chunk needs at least one row"));
    }
    if rows > chunk.rows {
        return Err(device_error(format!(
            "a chunk of {rows} rows is wider than the {}-row buffer",
            chunk.rows
        )));
    }
    // A chunk answers its last row; lanes answer every row, one a lane.
    let answers = match &sequences {
        Rows::Chunk(_) => 1,
        Rows::Lanes(states) => {
            if states.len() != rows {
                return Err(device_error(format!(
                    "{} lanes were given {rows} tokens: a lane steps one token",
                    states.len()
                )));
            }
            rows
        }
    };
    if answers > chunk.answers {
        return Err(device_error(format!(
            "{answers} answered rows need more than the buffer's {} logit rows",
            chunk.answers
        )));
    }
    let hidden = shared.hidden();
    let kv_width = shared.kv_width();
    let heads = shared.heads();
    let head_dim = config.head_dim;
    let kv_heads = plan.rank_kv_heads().max(1);
    let groups = heads / kv_heads;
    let history = config.conv_history().max(1);
    let position = match &sequences {
        Rows::Chunk(state) => state.position,
        Rows::Lanes(_) => 0,
    };
    let scratch = &chunk.scratch;
    let weights = &shared.weights;
    let weight = |index: usize| weights.get(index).pointer;
    let run = |result: Result<(), String>| result.map_err(device_error);
    let summary_stride = chunk.summary_stride;
    let mut blocks = 0usize;

    for step in schedule.prologue() {
        match (step.op, step.target) {
            (Op::Embed, _) => {
                // The embedding is vocabulary-parallel: exactly one rank holds a token's row, so
                // each owned token is widened where it lands and the gather below sums the rows
                // with the zeros the other ranks contribute.
                let rank_vocab = plan.rank_vocab();
                let first = plan.tp().rank() * rank_vocab;
                run(shared
                    .cuda
                    .memset_d8_async(scratch.x, 0, rows * hidden * std::mem::size_of::<f32>())
                    .map_err(|error| error.to_string()))?;
                let table = weight(step.weights[0]);
                for (row, token) in tokens.iter().enumerate() {
                    let Some(local) = (*token as usize)
                        .checked_sub(first)
                        .filter(|local| *local < rank_vocab)
                    else {
                        continue;
                    };
                    run(shared.launch(
                        shared.kernels.embed,
                        (flat(hidden), 1, 1),
                        (TOKEN_BLOCK, 1, 1),
                        &[
                            table.0,
                            scratch
                                .x
                                .offset(row * hidden * std::mem::size_of::<f32>())
                                .0,
                            local as u64,
                            hidden as u64,
                            hidden as u64,
                            0,
                        ],
                    ))?;
                }
            }
            (Op::Collective(Collective::EmbeddingRow), _) => {
                exchange_sum(shared, scratch.x, rows * hidden)?;
            }
            (Op::RmsNorm, Target::Norm(NormRole::Embedding)) => {
                let gain = weight(step.weights[0]);
                run(shared.rmsnorm_rows(scratch.x, gain, scratch.x, rows, hidden, hidden, hidden))?;
            }
            (op, target) => {
                return Err(device_error(format!(
                    "the chunk prologue's {op:?} at {target:?} has no device dispatch"
                )));
            }
        }
    }
    let stride = hidden * std::mem::size_of::<f32>();
    for image in images {
        if image.row + image.count > rows {
            return Err(device_error(format!(
                "image rows {}..{} fall outside a {rows}-row chunk",
                image.row,
                image.row + image.count
            )));
        }
        run(shared.cuda.copy_dtod_async(
            &scratch.x.offset(image.row * stride),
            &image.source,
            image.count * stride,
        ))?;
    }
    probe(shared, "chunk embed_norm", scratch.x, rows, hidden)?;

    for (index, layer) in schedule.layers().iter().enumerate() {
        let sliding = layer.sliding;
        let cache_rows = layer.kv_rows;
        let clips = shared.clip[index];
        let mut previous_conv: Option<ConvSlot> = None;
        let mut cursor = 0usize;
        while cursor < layer.steps.len() {
            let step = &layer.steps[cursor];
            match (step.op, step.target) {
                (Op::BlockResidual, Target::Input(ResidualInput::Attention)) => {
                    let queries = weight(step.weights[0]);
                    run(shared.block_residual(
                        chunk.summaries,
                        scratch.x,
                        queries,
                        scratch.rows,
                        blocks,
                        if sliding { 1 } else { 3 },
                        rows,
                        summary_stride,
                    ))?;
                }
                (Op::RmsNorm, Target::Norm(role)) => {
                    let gain = weight(step.weights[0]);
                    let (source, destination, src_stride) = match (role, sliding) {
                        (NormRole::Stream, _) => (scratch.x, scratch.q_in, hidden),
                        (NormRole::Attention, _) => (scratch.rows, scratch.q_in, hidden),
                        (NormRole::Query, false) => (scratch.rows, scratch.q_in, 3 * hidden),
                        (NormRole::Key, false) => (
                            scratch.rows.offset(hidden * std::mem::size_of::<f32>()),
                            scratch.k_in,
                            3 * hidden,
                        ),
                        (NormRole::Value, false) => (
                            scratch.rows.offset(2 * hidden * std::mem::size_of::<f32>()),
                            scratch.v_in,
                            3 * hidden,
                        ),
                        (NormRole::Mlp, _) => (scratch.rows, scratch.q_in, hidden),
                        (other, _) => {
                            return Err(device_error(format!(
                                "layer {index}: norm role {other:?} is not an input norm"
                            )));
                        }
                    };
                    run(shared.rmsnorm_rows(
                        source,
                        gain,
                        destination,
                        rows,
                        hidden,
                        src_stride,
                        hidden,
                    ))?;
                    if index == 0 && matches!(role, NormRole::Mlp) {
                        probe(shared, "chunk mlp_norm", destination, rows, hidden)?;
                    }
                }
                (Op::Gemv, Target::Project(projection)) => {
                    let matrix = weight(step.weights[0]);
                    // A projection's inner width is the width its weight multiplies: the stream
                    // for every projection but the attention output's, which contracts the heads.
                    let attention_width = heads * head_dim;
                    let (input, out, width, inner) = match projection {
                        Projection::Query => (scratch.q_in, scratch.q, attention_width, hidden),
                        Projection::Gate => (scratch.q_in, scratch.gate, attention_width, hidden),
                        Projection::Key => (
                            if sliding { scratch.q_in } else { scratch.k_in },
                            scratch.k,
                            kv_width,
                            hidden,
                        ),
                        Projection::Value => (
                            if sliding { scratch.q_in } else { scratch.v_in },
                            scratch.v,
                            kv_width,
                            hidden,
                        ),
                        Projection::Relative => {
                            (scratch.q_in, scratch.rel, heads * config.d_rel, hidden)
                        }
                        Projection::Sink => (scratch.q_in, scratch.sink, heads, hidden),
                        Projection::AttentionOut => {
                            (scratch.attn, scratch.delta, hidden, attention_width)
                        }
                        Projection::Router => (
                            scratch.q_in,
                            scratch.router_logits,
                            config.n_routed_experts,
                            hidden,
                        ),
                        Projection::Head => {
                            return Err(device_error("the head is an epilogue step"));
                        }
                    };
                    project(shared, walk, matrix, input, out, width, inner, rows)?;
                    if index == 0 && matches!(projection, Projection::AttentionOut) {
                        probe(shared, "chunk attention_out", out, rows, width)?;
                    }
                }
                (Op::ShortConv, Target::Conv(slot)) => {
                    let taps = weight(step.weights[0]);
                    let (input, out, channels) = match slot {
                        ConvSlot::Key => (scratch.k, scratch.k_next, kv_width),
                        ConvSlot::Value => (scratch.v, scratch.v_next, kv_width),
                        ConvSlot::Attention => (scratch.delta, scratch.delta_conv, hidden),
                        ConvSlot::Mlp => (scratch.mixed, scratch.mixed_conv, hidden),
                    };
                    let slot_index = conv_slot(slot);
                    match &sequences {
                        Rows::Chunk(state) => {
                            // The seam: this slot's history in the row layout the chunk reads,
                            // and the tail this launch leaves back in the layout the next decode
                            // step reads.
                            run(shared.conv_history(
                                state.conv[index][slot_index],
                                chunk.conv_head(slot_index),
                                channels,
                                true,
                            ))?;
                            // The entry indexes from the head, so the rows before the chunk's
                            // first row are the samples it filters against.
                            let _ = input;
                            run(shared.short_conv_rows(
                                chunk.conv_head(slot_index),
                                taps,
                                out,
                                rows,
                                channels,
                                history,
                            ))?;
                            run(shared.conv_history(
                                chunk
                                    .conv_head(slot_index)
                                    .offset(rows * channels * std::mem::size_of::<f32>()),
                                state.conv[index][slot_index],
                                channels,
                                false,
                            ))?;
                        }
                        Rows::Lanes(states) => {
                            // A lane's previous samples are its own history, not the row above
                            // it, so each row takes the decode step's launch on its lane's state.
                            for (row, state) in states.iter().enumerate() {
                                let at = row * channels * std::mem::size_of::<f32>();
                                run(shared.launch(
                                    shared.kernels.short_conv,
                                    (flat(channels), 1, 1),
                                    (ELEMENT_BLOCK, 1, 1),
                                    &[
                                        input.offset(at).0,
                                        taps.0,
                                        state.conv[index][slot_index].0,
                                        out.offset(at).0,
                                        channels as u64,
                                        shared.kernel_width() as u64,
                                    ],
                                ))?;
                            }
                        }
                    }
                    previous_conv = Some(slot);
                }
                (Op::ShortConvDynamic, Target::Conv(slot)) => {
                    let static_taps = weight(step.weights[0]);
                    let projection = weight(step.weights[1]);
                    let (input, out, source) = match slot {
                        ConvSlot::Key => (scratch.k, scratch.k_next, scratch.k_in),
                        ConvSlot::Value => (scratch.v, scratch.v_next, scratch.v_in),
                        other => {
                            return Err(device_error(format!(
                                "layer {index}: a dynamic convolution has no {other:?} slot"
                            )));
                        }
                    };
                    // The taps are a projection of this slot's own input, one kernel width a
                    // head, and they scale the row they were projected from - never the history.
                    let taps = kv_heads * shared.kernel_width();
                    project(
                        shared,
                        walk,
                        projection,
                        source,
                        scratch.taps,
                        taps,
                        hidden,
                        rows,
                    )?;
                    let slot_index = conv_slot(slot);
                    match &sequences {
                        Rows::Chunk(state) => {
                            run(shared.conv_history(
                                state.conv[index][slot_index],
                                chunk.conv_head(slot_index),
                                kv_width,
                                true,
                            ))?;
                            let _ = input;
                            run(shared.short_conv_dynamic_rows(
                                chunk.conv_head(slot_index),
                                static_taps,
                                scratch.taps,
                                out,
                                rows,
                                kv_width,
                                history,
                            ))?;
                            run(shared.conv_history(
                                chunk
                                    .conv_head(slot_index)
                                    .offset(rows * kv_width * std::mem::size_of::<f32>()),
                                state.conv[index][slot_index],
                                kv_width,
                                false,
                            ))?;
                        }
                        Rows::Lanes(states) => {
                            let byte = std::mem::size_of::<f32>();
                            for (row, state) in states.iter().enumerate() {
                                let at = row * kv_width * byte;
                                run(shared.launch(
                                    shared.kernels.short_conv_dynamic,
                                    (flat(kv_width), 1, 1),
                                    (ELEMENT_BLOCK, 1, 1),
                                    &[
                                        input.offset(at).0,
                                        static_taps.0,
                                        scratch.taps.offset(row * taps * byte).0,
                                        state.conv[index][slot_index].0,
                                        out.offset(at).0,
                                        kv_width as u64,
                                        shared.kernel_width() as u64,
                                        head_dim as u64,
                                    ],
                                ))?;
                            }
                        }
                    }
                    previous_conv = Some(slot);
                }
                (Op::KvStore, _) => match &sequences {
                    Rows::Chunk(state) => run(shared.launch(
                        shared.kernels.kv_store,
                        (flat(kv_width), rows as u32, 1),
                        (ELEMENT_BLOCK, 1, 1),
                        &[
                            scratch.k_next.0,
                            scratch.v_next.0,
                            state.keys[index].0,
                            state.values[index].0,
                            rows as u64,
                            position as u64,
                            kv_width as u64,
                            // A sliding layer's cache is its window and wraps; a full layer's is
                            // the context and is addressed absolutely.
                            if sliding { cache_rows as u64 } else { 0 },
                        ],
                    ))?,
                    Rows::Lanes(states) => {
                        // Each lane's row lands in its own cache at its own position, as the
                        // decode step stores it: the slot wrapped here, no ring for the entry.
                        let byte = std::mem::size_of::<f32>();
                        for (row, state) in states.iter().enumerate() {
                            let at = row * kv_width * byte;
                            run(shared.launch(
                                shared.kernels.kv_store,
                                (flat(kv_width), 1, 1),
                                (ELEMENT_BLOCK, 1, 1),
                                &[
                                    scratch.k_next.offset(at).0,
                                    scratch.v_next.offset(at).0,
                                    state.keys[index].0,
                                    state.values[index].0,
                                    1,
                                    (state.position % cache_rows.max(1)) as u64,
                                    kv_width as u64,
                                    0,
                                ],
                            ))?;
                        }
                    }
                },
                (Op::Attention, _) => {
                    let bank = weight(step.weights[0]);
                    match &sequences {
                        Rows::Chunk(state) => {
                            // The log-scaling factor is a function of the absolute position, so a
                            // chunk's rows differ. It is the plan's own rule, evaluated here,
                            // which is what keeps a chunk's score the score the row-at-a-time
                            // walk computes.
                            let taus: Vec<f32> =
                                (0..rows).map(|row| layer.tau.at(position + row)).collect();
                            run(shared
                                .cuda
                                .copy_htod(&scratch.taus, &row_bytes(&taus))
                                .map_err(|error| error.to_string()))?;
                            run(shared.attention_chunk(
                                scratch,
                                index,
                                scratch.q,
                                scratch.gate,
                                state.keys[index],
                                state.values[index],
                                state.tables[index],
                                scratch.rel,
                                bank,
                                scratch.sink,
                                clips,
                                scratch.attn_bf16,
                                scratch.taus,
                                rows,
                                position,
                                kv_heads,
                                groups,
                                if sliding { layer.window } else { 0 },
                                if sliding { cache_rows } else { 0 },
                                layer.rel_extent,
                                layer.tau.at(position),
                            ))?;
                        }
                        Rows::Lanes(states) => {
                            // Each lane's row against its own cache, through the decode step's
                            // call: the lane's own one-row arena holds the split walk's passes,
                            // so the pass a key falls in is the one its decode step would use.
                            let byte = std::mem::size_of::<f32>();
                            let query_width = heads * head_dim;
                            for (row, state) in states.iter().enumerate() {
                                let at = state.position;
                                run(shared.attention_rows(
                                    &state.scratch,
                                    index,
                                    scratch.q.offset(row * query_width * byte),
                                    scratch.gate.offset(row * query_width * byte),
                                    state.keys[index],
                                    state.values[index],
                                    state.tables[index],
                                    scratch.rel.offset(row * heads * config.d_rel * byte),
                                    bank,
                                    scratch.sink.offset(row * heads * byte),
                                    clips,
                                    scratch.attn_bf16.offset(row * query_width * 2),
                                    Deviceptr(0),
                                    1,
                                    at,
                                    plan.rank_kv_heads(),
                                    groups,
                                    if sliding { layer.window } else { 0 },
                                    if sliding { cache_rows } else { 0 },
                                    layer.rel_extent,
                                    layer.tau.at(at),
                                ))?;
                            }
                        }
                    }
                }
                (Op::Widen, _) => {
                    run(shared.launch(
                        shared.kernels.widen,
                        (flat(rows * heads * head_dim), 1, 1),
                        (ELEMENT_BLOCK, 1, 1),
                        &[
                            scratch.attn_bf16.0,
                            scratch.attn.0,
                            (rows * heads * head_dim) as u64,
                        ],
                    ))?;
                }
                (Op::ResidualAdd, _) => {
                    let delta = match previous_conv {
                        Some(ConvSlot::Mlp) => scratch.mixed_conv,
                        _ => scratch.delta_conv,
                    };
                    run(shared.launch(
                        shared.kernels.residual_add,
                        (flat(hidden), rows as u32, 1),
                        (ELEMENT_BLOCK, 1, 1),
                        &[
                            scratch.x.0,
                            delta.0,
                            scratch.x.0,
                            rows as u64,
                            hidden as u64,
                        ],
                    ))?;
                    if index == 0 {
                        let label = if previous_conv == Some(ConvSlot::Mlp) {
                            "chunk band_res"
                        } else {
                            "chunk attn_res"
                        };
                        probe(shared, label, scratch.x, rows, hidden)?;
                    }
                    previous_conv = None;
                }
                (Op::SummaryPush, _) => {
                    if blocks >= chunk.summary_rows {
                        return Err(device_error(format!(
                            "layer {index}: the summary stack is full ({} rows) and the plan                              pushes another",
                            chunk.summary_rows
                        )));
                    }
                    run(shared.summary_push(
                        scratch.x,
                        chunk.summaries,
                        rows,
                        blocks,
                        summary_stride,
                    ))?;
                    blocks += 1;
                    // At a boundary the sublayer output *is* the stream: there is no additive
                    // residual, and the pre-attention stream is the new summary, so the stream
                    // becomes the attention convolution's output. The row-at-a-time walk's
                    // convolution writes in place, so its own rule (`x <- delta`) is this copy;
                    // taking the projection instead would drop the convolution.
                    run(shared
                        .cuda
                        .copy_dtod(
                            &scratch.x,
                            &scratch.delta_conv,
                            rows * hidden * std::mem::size_of::<f32>(),
                        )
                        .map_err(|error| error.to_string()))?;
                }
                (Op::BlockResidual, Target::Input(ResidualInput::Mlp)) => {
                    let queries = weight(step.weights[0]);
                    run(shared.block_residual(
                        chunk.summaries,
                        scratch.x,
                        queries,
                        scratch.rows,
                        blocks,
                        1,
                        rows,
                        summary_stride,
                    ))?;
                }
                (Op::Router, _) => {
                    let bias = weight(step.weights[0]);
                    run(shared.router_rows(
                        scratch.router_logits,
                        bias,
                        scratch.indices,
                        scratch.shares,
                        rows,
                    ))?;
                    if index == 0 {
                        probe(
                            shared,
                            "chunk router_logits",
                            scratch.router_logits,
                            rows,
                            config.n_routed_experts,
                        )?;
                    }
                }
                (Op::ExpertGateUp, Target::Band(BandOp::GateUp)) => {
                    let activation = &layer.steps[cursor + 1];
                    let down = &layer.steps[cursor + 2];
                    if walk == PrefillWalk::Tensor && layer.experts == ExpertBody::NvFp4 {
                        run_band_mma(shared, chunk, layer, rows)?;
                    } else {
                        run_band_chunk(shared, chunk, layer, step, activation, down, rows)?;
                    }
                    cursor += 4;
                    continue;
                }
                (Op::Collective(Collective::AttentionDelta), _) => {
                    exchange_sum(shared, scratch.delta, rows * hidden)?;
                }
                (Op::Collective(Collective::ExpertDelta), _) => {
                    // The mix is local until this point: the probe reads the rank's own share of
                    // the sum, which is the number the row-at-a-time walk's own mix prints.
                    if index == 0 {
                        probe(shared, "chunk band", scratch.mixed, rows, hidden)?;
                    }
                    exchange_sum(shared, scratch.mixed, rows * hidden)?;
                }
                (Op::Collective(_), _) => {}
                (op, target) => {
                    return Err(device_error(format!(
                        "layer {index}: the chunk schedule's {op:?} at {target:?} has no device                          dispatch"
                    )));
                }
            }
            cursor += 1;
        }
        if trace_on() {
            let mut bytes = vec![0u8; rows * hidden * std::mem::size_of::<f32>()];
            run(shared
                .cuda
                .copy_dtoh(&mut bytes, &scratch.x)
                .map_err(|error| error.to_string()))?;
            let values = from_bytes(&bytes);
            for row in 0..rows {
                let slice = &values[row * hidden..(row + 1) * hidden];
                let rms =
                    (slice.iter().map(|value| value * value).sum::<f32>() / hidden as f32).sqrt();
                eprintln!(
                    "[inkling] chunk rank {} layer {index} row {row} x0={:.5} x1={:.5} rms={rms:.5}",
                    plan.tp().rank(),
                    slice[0],
                    slice[1]
                );
            }
        }
    }

    // The epilogue belongs to the answered rows: a chunk's last row, whose head is what the sampler
    // reads, or every lane's. Each answered row's block residual is that row's own chain.
    let first = rows - answers;
    let byte = std::mem::size_of::<f32>();
    let rank_vocab = plan.rank_vocab();
    let padded = rank_vocab * shared.exchange.world();
    let unpadded = config.unpadded_vocab_size.min(padded).max(1);
    for step in schedule.epilogue() {
        match (step.op, step.target) {
            (Op::BlockResidual, Target::Input(ResidualInput::Output)) => {
                let queries = weight(step.weights[0]);
                run(shared.block_residual(
                    chunk.summaries.offset(first * summary_stride * byte),
                    scratch.x.offset(first * hidden * byte),
                    queries,
                    scratch.final_row,
                    blocks,
                    1,
                    answers,
                    summary_stride,
                ))?;
            }
            (Op::RmsNorm, Target::Norm(NormRole::Final)) => {
                let gain = weight(step.weights[0]);
                run(shared.rmsnorm(scratch.final_row, gain, scratch.final_row, answers, hidden))?;
            }
            (Op::Gemv, Target::Project(Projection::Head)) => {
                let matrix = weight(step.weights[0]);
                match &sequences {
                    Rows::Chunk(_) => run(shared.gemv(
                        matrix,
                        scratch.final_row,
                        scratch.logits,
                        rank_vocab,
                        hidden,
                    ))?,
                    // The lanes' rows through the batched GEMV, whose row is the one-row
                    // launch's bits.
                    Rows::Lanes(_) => project(
                        shared,
                        PrefillWalk::Chunk,
                        matrix,
                        scratch.final_row,
                        scratch.logits,
                        rank_vocab,
                        hidden,
                        answers,
                    )?,
                }
            }
            (Op::Collective(Collective::LogitSlice), _) => {
                // A row's slices gather into that row's padded vocabulary, which is the layout
                // the softcap trims; one gather a row keeps each round the decode step's width.
                for answer in 0..answers {
                    exchange_gather(
                        shared,
                        scratch.logits.offset(answer * rank_vocab * byte),
                        scratch.logits_full.offset(answer * padded * byte),
                        rank_vocab,
                    )?;
                }
            }
            (Op::Collective(_), _) => {}
            (Op::LogitSoftcap, _) => {
                run(shared.launch(
                    shared.kernels.logit_softcap,
                    (answers as u32, 1, 1),
                    (ELEMENT_BLOCK, 1, 1),
                    &[
                        scratch.logits_full.0,
                        scratch.logits_bf16.0,
                        answers as u64,
                        padded as u64,
                        unpadded as u64,
                        bits(config.final_logit_softcap.unwrap_or(0.0) as f32),
                    ],
                ))?;
            }
            (op, target) => {
                return Err(device_error(format!(
                    "the chunk epilogue's {op:?} at {target:?} has no device dispatch"
                )));
            }
        }
    }

    match &mut sequences {
        Rows::Chunk(state) => {
            state.blocks = blocks;
            state.position += rows;
        }
        Rows::Lanes(states) => {
            for state in states.iter_mut() {
                state.blocks = blocks;
                state.position += 1;
            }
        }
    }
    run(shared.cuda.synchronize().map_err(|error| error.to_string()))?;
    // As the decode step: a doorbell round that was given up is reported before a read-back.
    run(shared.exchange.check_device())?;
    let mut bytes = vec![0u8; answers * unpadded * 2];
    run(shared
        .cuda
        .copy_dtoh(&mut bytes, &scratch.logits_bf16)
        .map_err(|error| error.to_string()))?;
    let values: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
        .collect();
    Ok(values.chunks(unpadded).map(<[f32]>::to_vec).collect())
}

/// The convolution state slot a schedule's conv target names.
fn conv_slot(slot: ConvSlot) -> usize {
    match slot {
        ConvSlot::Key => 0,
        ConvSlot::Value => 1,
        ConvSlot::Attention => 2,
        ConvSlot::Mlp => 3,
    }
}

/// The blocks of a chunk expert projection over `rows` output rows: INKLING_GROUP_ROWS a block.
fn pair_grid(rows: usize) -> u32 {
    u32::try_from(rows.div_ceil(PAIR_ROWS)).unwrap_or(u32::MAX)
}

/// The output rows one block of inkling_*_expert_pairs computes (INKLING_GROUP_ROWS).
const PAIR_ROWS: usize = 16;
/// The column run one thread of the pair walk reads a step (INKLING_GROUP).
const PAIR_RUN: usize = 16;
/// The most rows a launch grid's second dimension holds.
const GRID_ROWS: usize = 65_535;

/// The (token, slot) rows one item of the grouped MMA holds (Q4P_MOE_ROWS).
const MOE_ROWS: usize = 32;
/// The K a stage of the grouped MMA reads (Q4P_BK), the intermediate columns a gate_up CTA writes
/// and the output columns a down CTA writes.
const MOE_K_STAGE: usize = 64;
const MOE_GATE_UP_COLUMNS: usize = 64;
const MOE_DOWN_COLUMNS: usize = 128;
/// The threads of a grouped-MMA CTA and of the activation split.
const MOE_THREADS: u32 = 256;
/// The most experts the plan and the MMA tables index (MOE_PLAN_MAX_EXPERTS, Q4P_MAX_EXPERTS).
const MOE_MAX_EXPERTS: usize = 512;
/// moe_plan's mode for an id outside [0, experts): left out of the items (MOE_PLAN_DROP).
const MOE_PLAN_DROP: u64 = 2;

/// The most items the shared plan can cut from `slots` slots over `owned` experts: a run per
/// expert and every MOE_ROWS rows of one. The grouped MMA's grid is this tall; past the count
/// the plan wrote, a CTA returns before touching anything.
pub(super) fn moe_items(slots: usize, owned: usize) -> usize {
    slots.div_ceil(MOE_ROWS) + owned.min(slots)
}

/// The routed band over a chunk on the tensor cores, for an NVFP4 layer under the `tensor` walk.
///
/// The kernels are Qwen's grouped W4A16 MMA as Qwen runs it: q4p_split_bf16 turns the expert
/// input into a bf16 head and remainder, the shared plan (moe_plan) sorts the slots this rank owns
/// into items of up to 32 rows an expert, q4p_moe_gate_up_capped multiplies an item by w13 (its
/// gate and up rows interleaved) into the softcapped SwiGLU, and q4p_moe_down by w2 into the same
/// `[slot][hidden]` partials the pair walk writes. The router's choice never leaves the device:
/// inkling_band_shares writes the mix's shares and the plan's local ids from it. The down pass's
/// routing weight is one, so the mix applies the share exactly as the pair walk's does.
///
/// The sums are the MMA's: E2M1 x E4M3 is exact in bf16, the activation carries 16 significant
/// bits in its two planes, and the accumulation is f32 in the tile's order. That is a different
/// rounding from the pair walk's f32 chain, which is why `chunk` stays the walk with the decode
/// step's bits.
fn run_band_mma(
    shared: &Shared,
    chunk: &Chunk,
    layer: &LayerSchedule,
    rows: usize,
) -> Result<(), BackendError> {
    let config = &shared.config;
    let scratch = &chunk.scratch;
    let top_k = config.num_experts_per_tok;
    let intermediate = config.moe_intermediate_size;
    let hidden = config.hidden_size;
    let (first, owned) = shared.plan.expert_range();
    let meta = shared.experts[layer.index].as_ref().ok_or_else(|| {
        device_error(format!(
            "layer {}: an NVFP4 band with no expert scales",
            layer.index
        ))
    })?;
    if !hidden.is_multiple_of(MOE_DOWN_COLUMNS)
        || !hidden.is_multiple_of(MOE_K_STAGE)
        || !intermediate.is_multiple_of(MOE_GATE_UP_COLUMNS)
        || !intermediate.is_multiple_of(MOE_K_STAGE)
        || owned > MOE_MAX_EXPERTS
    {
        return Err(device_error(format!(
            "layer {}: the grouped MMA's tiles do not cover hidden {hidden}, intermediate \
             {intermediate} and {owned} experts a rank; PREFILL=chunk walks this band",
            layer.index
        )));
    }
    let slots = rows * top_k;
    let elements = rows * hidden;
    let items = moe_items(slots, owned);
    shared
        .launch(
            shared.kernels.band_shares,
            (
                u32::try_from(slots.div_ceil(SHARES_BLOCK as usize)).unwrap_or(u32::MAX),
                1,
                1,
            ),
            (SHARES_BLOCK, 1, 1),
            &[
                scratch.indices.0,
                scratch.shares.0,
                scratch.scales.0,
                slots as u64,
                first as u64,
                owned as u64,
                scratch.local.0,
            ],
        )
        .map_err(device_error)?;
    // The expert input's two bf16 planes go in the band buffer, which the pair walk's f32
    // gate/up rows need and this walk does not.
    shared
        .launch(
            shared.kernels.split_bf16,
            (
                (elements / 2).div_ceil(MOE_THREADS as usize).clamp(1, 4096) as u32,
                1,
                1,
            ),
            (MOE_THREADS, 1, 1),
            &[scratch.q_in.0, scratch.band.0, elements as u64],
        )
        .map_err(device_error)?;
    shared
        .launch(
            shared.kernels.moe_plan,
            (1, 1, 1),
            (
                sparkflash_model::active_profile().plan.block_threads(),
                1,
                1,
            ),
            &[
                scratch.local.0,
                scratch.plan.0,
                scratch.items.0,
                scratch.item_count.0,
                0,
                slots as u64,
                owned as u64,
                top_k as u64,
                MOE_ROWS as u64,
                items as u64,
                0,
                // Another rank's slot is left out; its share is zero, so the mix skips its row.
                MOE_PLAN_DROP,
                0,
                0,
                0,
                0,
                0,
                hidden as u64,
            ],
        )
        .map_err(device_error)?;
    let grid_items = u32::try_from(items).map_err(device_error)?;
    // The SwiGLU rows' two bf16 planes fill the activation buffer's f32 rows exactly.
    let activated_plane = (slots * intermediate) as u64;
    shared
        .launch(
            shared.kernels.moe_gate_up,
            ((intermediate / MOE_GATE_UP_COLUMNS) as u32, grid_items, 1),
            (MOE_THREADS, 1, 1),
            &[
                meta.mma_table.0,
                meta.mma_scales.0,
                scratch.plan.0,
                scratch.items.0,
                scratch.item_count.0,
                scratch.band.0,
                elements as u64,
                scratch.activation.0,
                activated_plane,
                top_k as u64,
                hidden as u64,
                intermediate as u64,
                bits(config.mlp_gate_softcap as f32),
                bits(config.mlp_up_softcap as f32),
            ],
        )
        .map_err(device_error)?;
    shared
        .launch(
            shared.kernels.moe_down,
            ((hidden / MOE_DOWN_COLUMNS) as u32, grid_items, 1),
            (MOE_THREADS, 1, 1),
            &[
                meta.mma_table.0,
                meta.mma_scales.0,
                scratch.plan.0,
                scratch.items.0,
                scratch.item_count.0,
                scratch.activation.0,
                activated_plane,
                shared.route_ones.0,
                scratch.partials.0,
                top_k as u64,
                hidden as u64,
                intermediate as u64,
                // Every slot reads the same [top_k] ones.
                0,
            ],
        )
        .map_err(device_error)?;
    // The mix is local; the all-reduce that follows it completes the sum over the ranks.
    shared
        .launch(
            shared.kernels.expert_mix,
            (flat(hidden), rows as u32, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                scratch.mixed.0,
                scratch.partials.0,
                scratch.scales.0,
                rows as u64,
                top_k as u64,
                hidden as u64,
            ],
        )
        .map_err(device_error)?;
    Ok(())
}

/// The routed band over a chunk: one launch an expert over the rows that selected it.
fn run_band_chunk(
    shared: &Shared,
    chunk: &Chunk,
    layer: &LayerSchedule,
    gate_up: &Step,
    activation: &Step,
    down: &Step,
    rows: usize,
) -> Result<(), BackendError> {
    let config = &shared.config;
    let scratch = &chunk.scratch;
    let top_k = config.num_experts_per_tok;
    let intermediate = config.moe_intermediate_size;
    let hidden = config.hidden_size;
    let weights = &shared.weights;
    let weight = |index: usize| weights.get(index).pointer;
    let _ = activation;

    let mut index_bytes = vec![0u8; rows * top_k * std::mem::size_of::<i32>()];
    shared
        .cuda
        .copy_dtoh(&mut index_bytes, &scratch.indices)
        .map_err(device_error)?;
    let mut share_bytes = vec![0u8; rows * top_k * std::mem::size_of::<f32>()];
    shared
        .cuda
        .copy_dtoh(&mut share_bytes, &scratch.shares)
        .map_err(device_error)?;
    if trace_on() && layer.index == 0 {
        for row in 0..rows {
            let slots: Vec<i32> = (0..top_k)
                .map(|slot| {
                    let at = (row * top_k + slot) * std::mem::size_of::<i32>();
                    i32::from_le_bytes(index_bytes[at..at + 4].try_into().unwrap_or([0; 4]))
                })
                .collect();
            let shares: Vec<f32> = (0..top_k)
                .map(|slot| {
                    let at = (row * top_k + slot) * std::mem::size_of::<f32>();
                    f32::from_le_bytes(share_bytes[at..at + 4].try_into().unwrap_or([0; 4]))
                })
                .collect();
            eprintln!("[inkling] chunk router layer 0 row {row} slots {slots:?} shares {shares:?}");
        }
    }

    // The router is replicated, so every rank selected the same experts; each rank computes the
    // ones it owns and the other slots contribute a zero share to the sum over ranks.
    let (first, owned) = shared.plan.expert_range();
    let mut groups: Vec<Vec<i32>> = vec![Vec::new(); owned.max(1)];
    let mut shares = vec![0f32; rows * top_k];
    for row in 0..rows {
        for slot in 0..top_k {
            let at = (row * top_k + slot) * std::mem::size_of::<i32>();
            let expert = i32::from_le_bytes(
                index_bytes[at..at + std::mem::size_of::<i32>()]
                    .try_into()
                    .unwrap_or([0; 4]),
            )
            .max(0) as usize;
            if expert < first || expert >= first + owned {
                continue;
            }
            groups[expert - first].push(row as i32);
            groups[expert - first].push((row * top_k + slot) as i32);
            shares[row * top_k + slot] = f32::from_le_bytes(
                share_bytes[at..at + std::mem::size_of::<f32>()]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
        }
    }
    // One entry a (row, band slot) pair, grouped by expert, so a launch reads its rows and its
    // destinations from the device without the host guessing at either.
    let mut entries: Vec<i32> = Vec::with_capacity(rows * top_k * 2);
    let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(owned.max(1));
    for pairs in &groups {
        if pairs.is_empty() {
            bounds.push((0, 0));
            continue;
        }
        bounds.push((entries.len() / 2, pairs.len() / 2));
        entries.extend_from_slice(pairs);
    }
    if !entries.is_empty() {
        shared
            .cuda
            .copy_htod(&scratch.plan, &word_bytes(&entries))
            .map_err(device_error)?;
    }
    shared
        .cuda
        .copy_htod(&scratch.scales, &row_bytes(&shares))
        .map_err(device_error)?;

    if !hidden.is_multiple_of(PAIR_RUN)
        || !intermediate.is_multiple_of(PAIR_RUN)
        || !MXFP8_BLOCK.is_multiple_of(PAIR_RUN)
    {
        return Err(device_error(format!(
            "layer {}: the chunk band's widths ({hidden}, {intermediate}, MXFP8 block \
             {MXFP8_BLOCK}) are not whole runs of the {PAIR_RUN} columns a thread reads",
            layer.index
        )));
    }
    let gate_up_codes = weight(gate_up.weights[0]);
    let gate_up_scales = gate_up.weights.get(1).map(|index| weight(*index));
    let down_codes = weight(down.weights[0]);
    let down_scales = down.weights.get(1).map(|index| weight(*index));
    let meta = shared.experts[layer.index].as_ref();
    let expert_bytes = codes_bytes(2 * intermediate, hidden, layer.experts);
    let expert_scale_bytes = scales_bytes(2 * intermediate, hidden, layer.experts);
    let down_bytes = codes_bytes(hidden, intermediate, layer.experts);
    let down_scale_bytes = scales_bytes(hidden, intermediate, layer.experts);

    for (local, &(start, count)) in bounds.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let plan = scratch.plan.offset(start * 2 * std::mem::size_of::<i32>());
        match layer.experts {
            ExpertBody::Bf16 => {
                // An unquantised band is a dense projection a expert, so the row's own input and
                // its own band slot are the only things a plan can say.
                for entry in start..start + count {
                    let row = entries[2 * entry] as usize;
                    let slot = entries[2 * entry + 1] as usize;
                    shared
                        .gemv(
                            gate_up_codes.offset(local * expert_bytes),
                            scratch
                                .q_in
                                .offset(row * hidden * std::mem::size_of::<f32>()),
                            scratch
                                .band
                                .offset(slot * 2 * intermediate * std::mem::size_of::<f32>()),
                            2 * intermediate,
                            hidden,
                        )
                        .map_err(device_error)?;
                }
            }
            ExpertBody::NvFp4 | ExpertBody::MxFp8 => {
                let (entry, last) = match layer.experts {
                    ExpertBody::NvFp4 => (
                        shared.kernels.expert_nvfp4_pairs,
                        bits(meta.map_or(1.0, |scale| scale.gate_up[local])),
                    ),
                    _ => (shared.kernels.expert_mxfp8_pairs, MXFP8_BLOCK as u64),
                };
                let scale_pointer = gate_up_scales.map_or(Deviceptr(0), |pointer| {
                    pointer.offset(local * expert_scale_bytes)
                });
                shared
                    .launch(
                        entry,
                        (pair_grid(2 * intermediate), 1, 1),
                        (ELEMENT_BLOCK, 1, 1),
                        &[
                            gate_up_codes.offset(local * expert_bytes).0,
                            scale_pointer.0,
                            scratch.q_in.0,
                            scratch.band.0,
                            2 * intermediate as u64,
                            hidden as u64,
                            last,
                            plan.0,
                            count as u64,
                        ],
                    )
                    .map_err(device_error)?;
            }
        }
    }

    if layer.index == 0 {
        probe_band(
            shared,
            "chunk gate_up",
            scratch.band,
            &entries,
            2 * intermediate,
        )?;
    }

    // The activation is elementwise over the whole band, whatever selected it. A slot is a grid
    // row, and a wide chunk has more slots than a grid has rows, so the band goes in runs of
    // GRID_ROWS slots: the same element is the same arithmetic whichever launch carries it.
    let slots = rows * top_k;
    let byte = std::mem::size_of::<f32>();
    let mut first_slot = 0usize;
    while first_slot < slots {
        let take = (slots - first_slot).min(GRID_ROWS);
        shared
            .launch(
                shared.kernels.swiglu,
                (flat(intermediate), take as u32, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    scratch.band.offset(first_slot * 2 * intermediate * byte).0,
                    scratch
                        .activation
                        .offset(first_slot * intermediate * byte)
                        .0,
                    intermediate as u64,
                    bits(config.mlp_gate_softcap as f32),
                    bits(config.mlp_up_softcap as f32),
                ],
            )
            .map_err(device_error)?;
        first_slot += take;
    }

    if layer.index == 0 {
        probe_band(
            shared,
            "chunk swiglu",
            scratch.activation,
            &entries,
            intermediate,
        )?;
    }

    // The down projection reads the band slot it was activated in, not the token's row of the
    // stream: the plan's pairs are (input row, output slot), and for this pass the input is the
    // same slot the output lands on. The grouping is the gate/up pass's, so the pairs are its
    // entries in its order with the row replaced by the slot.
    let down_pairs: Vec<i32> = entries
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|pair| [pair[1], pair[1]])
        .collect();
    if !down_pairs.is_empty() {
        shared
            .cuda
            .copy_htod(&scratch.plan, &word_bytes(&down_pairs))
            .map_err(device_error)?;
    }

    for (local, &(start, count)) in bounds.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let plan = scratch.plan.offset(start * 2 * std::mem::size_of::<i32>());
        match layer.experts {
            ExpertBody::Bf16 => {
                for entry in start..start + count {
                    let row = entries[2 * entry] as usize;
                    let slot = entries[2 * entry + 1] as usize;
                    shared
                        .gemv(
                            down_codes.offset(local * down_bytes),
                            scratch
                                .activation
                                .offset(slot * intermediate * std::mem::size_of::<f32>()),
                            scratch
                                .partials
                                .offset(slot * hidden * std::mem::size_of::<f32>()),
                            hidden,
                            intermediate,
                        )
                        .map_err(device_error)?;
                    let _ = row;
                }
            }
            ExpertBody::NvFp4 | ExpertBody::MxFp8 => {
                let (entry, last) = match layer.experts {
                    ExpertBody::NvFp4 => (
                        shared.kernels.expert_nvfp4_pairs,
                        bits(meta.map_or(1.0, |scale| scale.down[local])),
                    ),
                    _ => (shared.kernels.expert_mxfp8_pairs, MXFP8_BLOCK as u64),
                };
                let scale_pointer = down_scales.map_or(Deviceptr(0), |pointer| {
                    pointer.offset(local * down_scale_bytes)
                });
                shared
                    .launch(
                        entry,
                        (pair_grid(hidden), 1, 1),
                        (ELEMENT_BLOCK, 1, 1),
                        &[
                            down_codes.offset(local * down_bytes).0,
                            scale_pointer.0,
                            scratch.activation.0,
                            scratch.partials.0,
                            hidden as u64,
                            intermediate as u64,
                            last,
                            plan.0,
                            count as u64,
                        ],
                    )
                    .map_err(device_error)?;
            }
        }
    }

    if layer.index == 0 {
        probe_band(shared, "chunk down", scratch.partials, &entries, hidden)?;
    }

    // The mix is local; the all-reduce that follows it completes the sum over the ranks.
    shared
        .launch(
            shared.kernels.expert_mix,
            (flat(hidden), rows as u32, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                scratch.mixed.0,
                scratch.partials.0,
                scratch.scales.0,
                rows as u64,
                top_k as u64,
                hidden as u64,
            ],
        )
        .map_err(device_error)?;
    Ok(())
}
