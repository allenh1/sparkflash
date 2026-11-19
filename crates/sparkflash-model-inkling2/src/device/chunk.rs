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
//!   history, and written back the same way; a verify keeps its rows instead and its round's
//!   settle writes the history back from them. A cold start's history is zeros, which is the
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
use sparkflash_ops::moe::{MoePlanCall, grouped};
use sparkflash_ops::projection::{Route, WidePolicy, route_bf16};

use crate::device_plan::{BandOp, Collective, ConvSlot, ExpertBody, LayerSchedule, Step, TauRule};

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
        if takes_two_shots(elements, world) {
            return two_shots(shared, None, pointer, slots, elements);
        }
        shared
            .exchange
            .exchange_on_device(0, pointer.0, slots.0, elements * std::mem::size_of::<f32>())
            .map_err(device_error)?;
        return sum_slots(shared, None, pointer.0, slots, elements);
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

/// Whether [exchange_sum] of `elements` takes two shots ([two_shots]) rather than one.
fn takes_two_shots(elements: usize, world: usize) -> bool {
    elements * std::mem::size_of::<f32>() >= TWO_SHOT_FROM && elements.is_multiple_of(4 * world)
}

/// [exchange_sum]'s two shots of `elements` at `pointer`, on the run's stream or, under `side`,
/// on that stream behind everything queued on the run's so far: this rank sums chunk `rank`
/// (every peer's piece of it lands in the slots), then the summed chunks are shared in place. Each
/// element is the same fold in the same order, so the bits are the one-shot's at half the bytes a
/// rank.
fn two_shots(
    shared: &Shared,
    side: Option<&SideStream>,
    pointer: Deviceptr,
    slots: Deviceptr,
    elements: usize,
) -> Result<(), BackendError> {
    let (rank, world) = (shared.exchange.rank(), shared.exchange.world());
    let float = std::mem::size_of::<f32>();
    let chunk = elements / world;
    let mine = pointer.0 + (rank * chunk * float) as u64;
    if let Some(side) = side {
        shared.cuda.side_follow_run(side).map_err(device_error)?;
    }
    let stream = side.map_or(0, SideStream::raw);
    shared
        .exchange
        .scatter_on_device(stream, pointer.0, slots.0, elements * float)
        .map_err(device_error)?;
    sum_slots(shared, side, mine, slots, chunk)?;
    shared
        .exchange
        .exchange_on_device(stream, mine, pointer.0, chunk * float)
        .map_err(device_error)
}

/// q4e_sum_slots over `count` values at `own`: the slots' pieces and its own, in ascending rank
/// order from zero, on the run's stream or on `side`.
fn sum_slots(
    shared: &Shared,
    side: Option<&SideStream>,
    own: u64,
    slots: Deviceptr,
    count: usize,
) -> Result<(), BackendError> {
    let grid = ((count / 4).div_ceil(SUM_BLOCK as usize) as u32, 1, 1);
    let args = [
        own,
        slots.0,
        count as u64,
        shared.exchange.rank() as u64,
        shared.exchange.world() as u64,
    ];
    let entry = shared.kernels.sum_slots;
    match side {
        None => shared.launch(entry, grid, (SUM_BLOCK, 1, 1), &args),
        // The side stream already follows the run's (two_shots), and outside a graph segment
        // side_launch always launches.
        Some(side) => shared
            .cuda
            .side_launch(side, entry.kernel, grid, (SUM_BLOCK, 1, 1), &args),
    }
    .map_err(device_error)
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

/// The row expert split's all-gather ([ExpertSplit::Rows]).
///
/// Every rank wrote its `rows` runs of `piece` bytes at `base + rank * rows * piece`; the device
/// leg brings the peers' runs in beside it, and with `joined` the join lays each row's runs side
/// by side there, rank-major, so a row every rank computed a run of reads as one row. One row's
/// runs already are the row, which is why a one-row gather into its destination needs no join.
/// Both are copies: the words arrive as they left.
pub(super) fn gather_runs(
    shared: &Shared,
    base: Deviceptr,
    joined: Option<Deviceptr>,
    rows: usize,
    piece: usize,
) -> Result<(), BackendError> {
    let (rank, world) = (shared.exchange.rank(), shared.exchange.world());
    let run = rows * piece;
    if world > 1 {
        shared
            .exchange
            .exchange_on_device(0, base.0 + (rank * run) as u64, base.0, run)
            .map_err(device_error)?;
    }
    let Some(joined) = joined else {
        return Ok(());
    };
    // The open refuses a split whose pieces are not whole 16-byte words.
    let words = piece / 16;
    shared
        .launch(
            shared.kernels.join_pieces,
            (flat(rows * world * words), 1, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[joined.0, base.0, rows as u64, words as u64, world as u64],
        )
        .map_err(device_error)
}

/// Where a band's down projection lands and where this rank's mix rows wait for the exchange.
///
/// The expert-parallel splits project into `partials` and mix the residual-wide rows into
/// `mixed`, which the exchange sums. The row split projects into the band (its activation has
/// read it by then) and mixes its own rows into its run of a staging buffer: `mixed` itself for
/// one row, whose runs are the row, and otherwise `partials`, which the activation's join has
/// read by then and which the exchange's join reads back into `mixed`.
pub(super) struct BandOut {
    /// The down projection's output, [slot][rows].
    pub(super) down: Deviceptr,
    /// The mix's output, [row][width] at this rank's run.
    pub(super) mix: Deviceptr,
    /// The gather's base, under the row split.
    pub(super) staging: Deviceptr,
    /// The mix's width: the residual, or this rank's rows of it.
    pub(super) width: usize,
    /// Whether the exchange gathers (the row split) rather than sums.
    pub(super) gathers: bool,
}

impl BandOut {
    pub(super) fn of(shared: &Shared, scratch: &Scratch, rows: usize) -> Self {
        let down = shared.plan.expert_down();
        if shared.config.expert_split != ExpertSplit::Rows {
            return Self {
                down: scratch.partials,
                mix: scratch.mixed,
                staging: scratch.mixed,
                width: shared.hidden(),
                gathers: false,
            };
        }
        let staging = if rows == 1 {
            scratch.mixed
        } else {
            scratch.partials
        };
        let run = rows * down.rows * std::mem::size_of::<f32>();
        Self {
            down: scratch.band,
            mix: staging.offset(shared.exchange.rank() * run),
            staging,
            width: down.rows,
            gathers: true,
        }
    }
}

/// The band's mix over `rows` rows of `partials` [row][slot][width] into `out`.
///
/// Under the row split one rank holds every pick's rows, so the mix folds each expert-parallel
/// owner's picks from zero and those in rank order (inkling_expert_mix's grouped form): the
/// expert-parallel rank's local mix and its exchange's sum, term for term.
pub(super) fn band_mix(
    shared: &Shared,
    scratch: &Scratch,
    rows: usize,
) -> Result<(), BackendError> {
    let out = BandOut::of(shared, scratch, rows);
    let (indices, owner_experts, owners) = if out.gathers {
        (
            scratch.indices.0,
            shared.plan.parallel_experts(),
            shared.exchange.world(),
        )
    } else {
        (0, 0, 0)
    };
    shared
        .launch(
            shared.kernels.expert_mix,
            (flat(out.width), rows as u32, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                out.mix.0,
                out.down.0,
                scratch.scales.0,
                rows as u64,
                shared.config.num_experts_per_tok as u64,
                out.width as u64,
                indices,
                owner_experts as u64,
                owners as u64,
            ],
        )
        .map_err(device_error)
}

/// The band's exchange: the sum of the ranks' mixes, or under the row split the gather of
/// their rows into `mixed`.
pub(super) fn band_exchange(
    shared: &Shared,
    scratch: &Scratch,
    rows: usize,
) -> Result<(), BackendError> {
    let out = BandOut::of(shared, scratch, rows);
    if !out.gathers {
        return exchange_sum(shared, scratch.mixed, rows * shared.hidden());
    }
    let joined = (rows > 1).then_some(scratch.mixed);
    gather_runs(
        shared,
        out.staging,
        joined,
        rows,
        out.width * std::mem::size_of::<f32>(),
    )
}

/// Whether the decode band's few-row routed-expert body takes a layer at `rows` rows.
///
/// The body is the prompt tile's chain over one to eight rows: the arm's `BATCH_ROWS`, the
/// precisions it stores (NVFP4 and MXFP8), both activation planes (`PLANES=both`), the grouped
/// table, the shapes the bodies cover (whole 32-column K runs and whole eight-column writes;
/// the NVFP4 bodies walk 128-column K blocks) and a band whose down projection does not gather.
/// Everything else keeps the arm's row walk.
#[allow(clippy::too_many_arguments)]
pub(super) fn expert_rows_body_fits(
    rows: usize,
    experts: ExpertBody,
    planes: ActivationPlanes,
    gathers: bool,
    table: bool,
    hidden: usize,
    intermediate: usize,
    down: (usize, usize),
) -> bool {
    let (down_rows, down_columns) = down;
    (1..=super::BATCH_ROWS).contains(&rows)
        && !gathers
        && matches!(experts, ExpertBody::NvFp4 | ExpertBody::MxFp8)
        && planes == ActivationPlanes::Both
        && table
        && hidden.is_multiple_of(32)
        && intermediate.is_multiple_of(32)
        && down_rows.is_multiple_of(8)
        && down_columns.is_multiple_of(32)
        && (if experts == ExpertBody::MxFp8 {
            // The MXFP8 body walks the tile's 64-column K stages.
            hidden.is_multiple_of(64) && intermediate.is_multiple_of(64)
        } else {
            hidden.is_multiple_of(128) && intermediate.is_multiple_of(128)
        })
}

/// The most rows the chunk walk steers on the device ([run_steered_band]): a verify's anchor and
/// its drafts, fused lanes, or a shared walk of four lanes' verifies at the deepest drafter
/// depth (four times sixteen rows). A wider chunk groups its pairs by expert on the host
/// ([run_band_chunk]), whose bits the steered band's equal; the steered band's buffers are the
/// chunk's, sized by its rows.
pub(super) const STEERED_BAND_ROWS: usize = 64;

/// The fewest rows whose steered band is planned by expert ([run_steered_band]). Four rows' 32
/// top-8 picks name about 30 experts, so reading each once saves next to nothing and the plan's
/// launches cost more: at 93k a four-row verify measured the same either way and two or four
/// fused lanes 5-9% slower planned, while an eight-row verify (DSpark at depth 7) measured 7%
/// faster planned.
pub(super) const PLANNED_BAND_ROWS: usize = 5;

/// The routed band of a quantised layer over `rows` rows of `input` [row][hidden], with the
/// router's choice left on the device.
///
/// A decode step's one row picks top-k distinct experts, and its launches cover the slots at once
/// (grid y, slot `row * top_k + k`): a block whose slot names an expert another rank holds exits,
/// [ExpertBody::NvFp4] and [ExpertBody::MxFp8] read the selection through `indices` and the NVFP4
/// scale_2 through the layer's device vector, and the mix's shares come from inkling_band_shares.
/// From [PLANNED_BAND_ROWS] rows on (a deep verify, several fused lanes) rows share enough experts
/// that moe_plan sorts this rank's slots by expert on the device and the item walk reads an expert
/// once for every slot that picked it; fewer rows stay on the band, an expert a slot. A slot's
/// every output is the pair walk's sum, term for term, either way, so the bits are the
/// host-grouped walk's; what is gone is the readback of the selection that made every layer wait
/// for the host, and a launch an expert.
pub(super) fn run_steered_band(
    shared: &Shared,
    scratch: &Scratch,
    layer: usize,
    gate_up: &Step,
    down: &Step,
    input: Deviceptr,
    rows: usize,
) -> Result<(), BackendError> {
    let config = &shared.config;
    let schedule = &shared.plan.layers()[layer];
    let top_k = config.num_experts_per_tok;
    let slots = rows * top_k;
    let intermediate = shared.plan.expert_intermediate();
    let hidden = config.hidden_size;
    let (first, owned) = shared.plan.expert_range();
    let byte = std::mem::size_of::<f32>();
    let weight = |step: &Step, at: usize| {
        step.weights
            .get(at)
            .map_or(Deviceptr(0), |index| shared.weights.get(*index).pointer)
    };
    let scale_2 = shared.experts[layer].as_ref();
    let planned = rows >= PLANNED_BAND_ROWS;
    let max_items = owned.min(slots);
    let out = BandOut::of(shared, scratch, rows);
    let down_shape = shared.plan.expert_down();
    // A decode step, a verify or a few lanes: one to eight rows take the grouped experts'
    // few-row bodies, the prompt tile's chain over the same plan. A wider band, a shape the
    // bodies do not cover or a row-split gather keeps the row walk below.
    let body = expert_rows_body_fits(
        rows,
        schedule.experts,
        shared.planes,
        out.gathers,
        shared.mma[layer].is_some(),
        hidden,
        intermediate,
        (down_shape.rows, down_shape.columns),
    );
    if body {
        let meta = shared.mma[layer]
            .ok_or_else(|| device_error(format!("layer {layer}: no grouped-MMA table")))?;
        let mx8 = schedule.experts == ExpertBody::MxFp8;
        let items = grouped::items(slots, owned);
        let grid_items = u32::try_from(items).map_err(device_error)?;
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
        let plan = MoePlanCall::items(
            scratch.local.0,
            scratch.plan.0,
            (scratch.items.0, scratch.item_count.0),
            (slots, owned, top_k),
            0,
            max_items,
        );
        let (grid, block) = MoePlanCall::launch();
        shared
            .launch(shared.kernels.moe_plan, grid, block, &plan.args())
            .map_err(device_error)?;
        // The expert input's two bf16 planes, as the tile's split stage reads them.
        let elements = rows * hidden;
        shared
            .launch(
                shared.kernels.split_bf16,
                (
                    (elements / 2)
                        .div_ceil(grouped::THREADS as usize)
                        .clamp(1, 4096) as u32,
                    1,
                    1,
                ),
                (grouped::THREADS, 1, 1),
                &[input.0, scratch.band.0, elements as u64],
            )
            .map_err(device_error)?;
        // The NVFP4 rows take the few-row body (q4p_moe_rows_gate_up_capped); the catalogue
        // has no MXFP8 few-row form, so those rows take the tile body itself, which computes
        // the same row (the item holds at most eight and the rest of its 128 are zero).
        let (gate_up, gate_up_columns, gate_up_threads) = if mx8 {
            (shared.kernels.moe_gate_up_mx8, 64usize, grouped::THREADS)
        } else {
            (shared.kernels.moe_rows_gate_up, 32usize, 128u32)
        };
        shared
            .launch(
                gate_up,
                ((intermediate / gate_up_columns) as u32, grid_items, 1),
                (gate_up_threads, 1, 1),
                &[
                    meta.table.0,
                    meta.scales.0,
                    scratch.plan.0,
                    scratch.items.0,
                    scratch.item_count.0,
                    scratch.band.0,
                    elements as u64,
                    scratch.activation.0,
                    (slots * intermediate) as u64,
                    top_k as u64,
                    hidden as u64,
                    intermediate as u64,
                    bits(config.mlp_gate_softcap as f32),
                    bits(config.mlp_up_softcap as f32),
                ],
            )
            .map_err(device_error)?;
        let (down, down_columns, down_threads) = if mx8 {
            (shared.kernels.moe_down_mx8, 128usize, grouped::THREADS)
        } else {
            (shared.kernels.moe_rows_down, 64usize, 128u32)
        };
        shared
            .launch(
                down,
                ((down_shape.rows / down_columns) as u32, grid_items, 1),
                (down_threads, 1, 1),
                &[
                    meta.table.0,
                    meta.scales.0,
                    scratch.plan.0,
                    scratch.items.0,
                    scratch.item_count.0,
                    scratch.activation.0,
                    (slots * intermediate) as u64,
                    shared.route_ones.0,
                    out.down.0,
                    top_k as u64,
                    hidden as u64,
                    intermediate as u64,
                    0,
                ],
            )
            .map_err(device_error)?;
        return band_mix(shared, scratch, rows);
    }
    if planned {
        // The shares, and this rank's local id a slot (-1 for another rank's), then the plan: one
        // item an expert this rank holds, its slots in ascending order; another rank's slots are
        // left out, and their zero shares keep the mix off their rows.
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
        // One item an expert, whatever its slot count.
        let plan = MoePlanCall::items(
            scratch.local.0,
            scratch.plan.0,
            (scratch.items.0, scratch.item_count.0),
            (slots, owned, top_k),
            0,
            max_items,
        );
        let (grid, block) = MoePlanCall::launch();
        shared
            .launch(shared.kernels.moe_plan, grid, block, &plan.args())
            .map_err(device_error)?;
    }
    // `row_slots` is the slots that read one input row: a token's top-k for the gate/up
    // projection, which reads its row of the stream, and 1 for the down projection, which reads
    // the slot's own activation.
    let projection = |step: &Step,
                      input: Deviceptr,
                      out: Deviceptr,
                      outputs: usize,
                      columns: usize,
                      row_slots: usize,
                      factor: Option<Deviceptr>|
     -> Result<(), BackendError> {
        let (band, items, tail) = match schedule.experts {
            ExpertBody::NvFp4 => (
                shared.kernels.expert_nvfp4_band,
                shared.kernels.expert_nvfp4_items,
                factor
                    .ok_or_else(|| {
                        device_error(format!(
                            "layer {layer}: an NVFP4 band has no scale_2 vector"
                        ))
                    })?
                    .0,
            ),
            ExpertBody::MxFp8 => (
                shared.kernels.expert_mxfp8_band,
                shared.kernels.expert_mxfp8_items,
                MXFP8_BLOCK as u64,
            ),
            ExpertBody::Bf16 => {
                return Err(device_error(format!(
                    "layer {layer}: a BF16 expert band is steered on the host"
                )));
            }
        };
        if !columns.is_multiple_of(BAND_RUN) || !MXFP8_BLOCK.is_multiple_of(BAND_RUN) {
            return Err(device_error(format!(
                "layer {layer}: a band row of {columns} columns (MXFP8 block {MXFP8_BLOCK}) is \
                 not whole runs of the {BAND_RUN} columns a thread reads"
            )));
        }
        if planned {
            return shared
                .launch(
                    items,
                    (
                        pair_grid(outputs),
                        u32::try_from(max_items).map_err(device_error)?,
                        1,
                    ),
                    (ELEMENT_BLOCK, 1, 1),
                    &[
                        weight(step, 0).0,
                        weight(step, 1).0,
                        input.0,
                        out.0,
                        outputs as u64,
                        columns as u64,
                        tail,
                        scratch.plan.0,
                        scratch.items.0,
                        scratch.item_count.0,
                        row_slots as u64,
                    ],
                )
                .map_err(device_error);
        }
        shared
            .launch(
                band,
                (
                    outputs.div_ceil(BAND_ROWS) as u32,
                    u32::try_from(slots).map_err(device_error)?,
                    1,
                ),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    weight(step, 0).0,
                    weight(step, 1).0,
                    input.0,
                    out.0,
                    outputs as u64,
                    columns as u64,
                    tail,
                    scratch.indices.0,
                    first as u64,
                    owned as u64,
                    row_slots as u64,
                ],
            )
            .map_err(device_error)
    };
    projection(
        gate_up,
        input,
        scratch.band,
        2 * intermediate,
        hidden,
        top_k,
        scale_2.map(|scale| scale.device_gate_up),
    )?;
    // Every slot's activation, owned or not: a slot another rank holds reads a stale band row,
    // and its share below is zero. Under the row split it is this rank's run of every slot's
    // columns, which the gather joins into whole rows before the down projection.
    let out = BandOut::of(shared, scratch, rows);
    let activated = if out.gathers {
        scratch
            .partials
            .offset(shared.exchange.rank() * slots * intermediate * byte)
    } else {
        scratch.activation
    };
    shared
        .launch(
            shared.kernels.swiglu,
            (
                flat(intermediate),
                u32::try_from(slots).map_err(device_error)?,
                1,
            ),
            (ELEMENT_BLOCK, 1, 1),
            &[
                scratch.band.0,
                activated.0,
                intermediate as u64,
                bits(config.mlp_gate_softcap as f32),
                bits(config.mlp_up_softcap as f32),
            ],
        )
        .map_err(device_error)?;
    if out.gathers {
        gather_runs(
            shared,
            scratch.partials,
            Some(scratch.activation),
            slots,
            intermediate * byte,
        )?;
    }
    let shape = shared.plan.expert_down();
    projection(
        down,
        scratch.activation,
        out.down,
        shape.rows,
        shape.columns,
        1,
        scale_2.map(|scale| scale.device_down),
    )?;
    if planned {
        return band_mix(shared, scratch, rows);
    }
    shared
        .launch(
            shared.kernels.band_shares,
            (1, 1, 1),
            (SHARES_BLOCK, 1, 1),
            &[
                scratch.indices.0,
                scratch.shares.0,
                scratch.scales.0,
                slots as u64,
                first as u64,
                owned as u64,
                // The steered band reads no plan.
                0,
            ],
        )
        .map_err(device_error)?;
    band_mix(shared, scratch, rows)
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
pub(super) fn project(
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
    // The tensor walk takes the wide tile on every shape it covers, at any rows; the walk is the
    // caller's and not this rank's prompt walk, because a step of fused lanes walks exactly
    // whatever a prompt walks.
    let policy = (walk == PrefillWalk::Tensor).then_some(WidePolicy::INKLING_TENSOR);
    if route_bf16(policy, width, inner, rows) == Route::Wide {
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
    /// A speculative verify: an anchor and its drafts at consecutive positions of one sequence,
    /// every row answered, each row the bits its own decode step would give it. The walk is
    /// [PrefillWalk::Chunk], whose projections and band are the decode step's sums; the
    /// convolutions are a chunk's, which take the decode entry's order; the attention is the
    /// decode step's split walk over a split arena, each row against its own prefix; and the head
    /// is the batched GEMV, whose row is the one-row launch's. Each layer's convolution inputs
    /// are left in the [VerifyTails], so the session can keep any leading run of the rows.
    Verify(&'s mut State, VerifyTails),
    /// Several sequences' verifies in one walk ([sparkflash_runtime::run_round]): each segment is
    /// one sequence's anchor and drafts at its consecutive positions, the segments' rows one after
    /// another in the order given. Each segment's sequence stages are the verify's on its own
    /// state, tails and drafter taps, so each row is the bits its own decode step would give it.
    /// A segment's convolutions read and leave their rows in its own tails, so the shared input
    /// rows are never written over.
    Segments(&'s mut [Segment<'a>]),
}

/// One sequence's rows in a [Rows::Segments] walk.
pub(super) struct Segment<'a> {
    pub state: &'a mut State,
    /// The sequence's verify tails: its convolutions' history and rows, for its settle.
    pub tails: VerifyTails,
    /// The anchor and the drafts verified with it.
    pub rows: usize,
    /// The sequence's drafter taps, when it drafts with DSpark.
    pub taps: Option<TapSink>,
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

/// A walk's block residuals chained ([RESIDUAL_CHAIN_ENV]): every call's queries in walk order, so
/// each call scores the next call's against the summary rows while its mix reads them, and what
/// the carry ([Scratch::residual_stats]) holds now.
struct ResidualChain {
    /// Every residual call's queries and their count, in the order the walk makes them.
    calls: Vec<(Deviceptr, usize)>,
    /// The call the walk makes next.
    next: usize,
    /// The queries the carry is scored against, and over how many summary rows.
    carried: Option<(Deviceptr, usize)>,
}

impl ResidualChain {
    /// The calls `schedule` makes: each layer's attention and MLP inputs (three queries on a full
    /// layer's attention, one elsewhere), then the epilogue's output.
    fn new(schedule: &InklingDevicePlan, weights: &Weights) -> Self {
        let mut calls = Vec::new();
        for layer in schedule.layers() {
            for step in &layer.steps {
                match (step.op, step.target) {
                    (Op::BlockResidual, Target::Input(ResidualInput::Attention)) => calls.push((
                        weights.get(step.weights[0]).pointer,
                        if layer.sliding { 1 } else { 3 },
                    )),
                    (Op::BlockResidual, Target::Input(ResidualInput::Mlp)) => {
                        calls.push((weights.get(step.weights[0]).pointer, 1));
                    }
                    _ => {}
                }
            }
        }
        for step in schedule.epilogue() {
            if let (Op::BlockResidual, Target::Input(ResidualInput::Output)) =
                (step.op, step.target)
            {
                calls.push((weights.get(step.weights[0]).pointer, 1));
            }
        }
        Self {
            calls,
            next: 0,
            carried: None,
        }
    }

    /// One residual call over `rows` rows. A call the walk makes in its order scores the next
    /// call's queries; one it repeats (the drafter's tap of the output) neither scores nor moves
    /// the chain, and takes the carry when its queries are the ones the carry was scored against.
    #[allow(clippy::too_many_arguments)]
    fn residual(
        &mut self,
        shared: &Shared,
        summaries: Deviceptr,
        partial: Deviceptr,
        queries: Deviceptr,
        count: usize,
        out: Deviceptr,
        carry: Deviceptr,
        blocks: usize,
        rows: usize,
        summary_stride: usize,
    ) -> Result<(), String> {
        let carried = match self.carried {
            Some((scored, over)) if scored == queries => over.min(blocks),
            _ => 0,
        };
        let in_order = self.calls.get(self.next) == Some(&(queries, count));
        let next = if in_order {
            self.calls.get(self.next + 1).copied()
        } else {
            None
        };
        shared.block_residual_chain(
            summaries,
            partial,
            queries,
            out,
            carry,
            blocks,
            count,
            carried,
            next,
            rows,
            summary_stride,
        )?;
        if in_order {
            self.next += 1;
        }
        if let Some((queries, _)) = next {
            self.carried = Some((queries, blocks));
        }
        Ok(())
    }
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
    taps: Option<TapSink>,
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
    // A chunk answers its last row; lanes answer every row, one a lane, and a verify every row.
    let answers = match &sequences {
        Rows::Chunk(_) => 1,
        Rows::Verify(_, tails) => {
            if rows > tails.rows {
                return Err(device_error(format!(
                    "a verify of {rows} rows is deeper than the {} rows its tails hold",
                    tails.rows
                )));
            }
            if walk != PrefillWalk::Chunk {
                return Err(device_error(
                    "a verify's rows are the decode step's sums: it walks PrefillWalk::Chunk",
                ));
            }
            if tails.history != config.conv_history().max(1) {
                return Err(device_error(format!(
                    "the verify's tails hold {} history rows and the convolutions read {}",
                    tails.history,
                    config.conv_history().max(1)
                )));
            }
            rows
        }
        Rows::Lanes(states) => {
            if states.len() != rows {
                return Err(device_error(format!(
                    "{} lanes were given {rows} tokens: a lane steps one token",
                    states.len()
                )));
            }
            rows
        }
        Rows::Segments(segments) => {
            let given: usize = segments.iter().map(|segment| segment.rows).sum();
            if given != rows {
                return Err(device_error(format!(
                    "segments of {given} rows were given {rows} tokens"
                )));
            }
            if walk != PrefillWalk::Chunk {
                return Err(device_error(
                    "a verify's rows are the decode step's sums: segments walk PrefillWalk::Chunk",
                ));
            }
            for segment in segments.iter() {
                let tails = &segment.tails;
                if segment.rows == 0 || segment.rows > tails.rows {
                    return Err(device_error(format!(
                        "a segment of {} rows, and its tails hold 1..={}",
                        segment.rows, tails.rows
                    )));
                }
                if tails.history != config.conv_history().max(1) {
                    return Err(device_error(format!(
                        "a segment's tails hold {} history rows and the convolutions read {}",
                        tails.history,
                        config.conv_history().max(1)
                    )));
                }
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
        Rows::Chunk(state) | Rows::Verify(state, _) => state.position,
        Rows::Lanes(_) | Rows::Segments(_) => 0,
    };
    // Each row's absolute position, for the scale tables: a segment's rows are at its own.
    let row_positions: Vec<usize> = match &sequences {
        Rows::Segments(segments) => segments
            .iter()
            .flat_map(|segment| (0..segment.rows).map(|row| segment.state.position + row))
            .collect(),
        _ => (0..rows).map(|row| position + row).collect(),
    };
    // Each segment's first row in the walk.
    let offsets: Vec<usize> = match &sequences {
        Rows::Segments(segments) => segments
            .iter()
            .scan(0, |first, segment| {
                let at = *first;
                *first += segment.rows;
                Some(at)
            })
            .collect(),
        _ => Vec::new(),
    };
    // Where the drafter's taps go: the walk's one sink over every row, or each segment's own over
    // its rows.
    let sinks: Vec<(TapSink, usize, usize)> = match &sequences {
        Rows::Segments(segments) => segments
            .iter()
            .zip(&offsets)
            .filter_map(|(segment, &first)| segment.taps.map(|sink| (sink, first, segment.rows)))
            .collect(),
        _ => taps.map(|sink| (sink, 0, rows)).into_iter().collect(),
    };
    let scratch = &chunk.scratch;
    let weights = &shared.weights;
    let weight = |index: usize| weights.get(index).pointer;
    let run = |result: Result<(), String>| result.map_err(device_error);
    let summary_stride = chunk.summary_stride;
    let mut blocks = 0usize;
    // The walk's block residuals chained ([RESIDUAL_CHAIN_ENV]) over more rows than the two-launch
    // form takes; a decode-sized walk keeps the per-call forms. `first` is the residual's first
    // row, where its share of the carry starts.
    let mut chain = (shared.residual_chain && rows > RESIDUAL_SPLIT_PAIRS)
        .then(|| ResidualChain::new(schedule, weights));
    #[allow(clippy::too_many_arguments)]
    let residual = |chain: &mut Option<ResidualChain>,
                    summaries: Deviceptr,
                    partial: Deviceptr,
                    queries: Deviceptr,
                    count: usize,
                    out: Deviceptr,
                    blocks: usize,
                    first: usize,
                    rows: usize|
     -> Result<(), BackendError> {
        let stats = scratch
            .residual_stats
            .offset(first * RESIDUAL_STATS_ROW * std::mem::size_of::<f32>());
        run(match chain {
            Some(chain) => chain.residual(
                shared,
                summaries,
                partial,
                queries,
                count,
                out,
                stats,
                blocks,
                rows,
                summary_stride,
            ),
            None => shared.block_residual(
                summaries,
                partial,
                queries,
                out,
                scratch.residual_stats,
                blocks,
                count,
                rows,
                summary_stride,
            ),
        })
    };

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
                    run(shared.call(
                        shared.kernels.embed,
                        sparkflash_ops::calls::embed(
                            table.0,
                            scratch
                                .x
                                .offset(row * hidden * std::mem::size_of::<f32>())
                                .0,
                            local as u32,
                            (hidden, hidden),
                        ),
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

    // A layer's per-row scale table: null for a layer that does not scale (its kernels take the
    // scalar `tau`, 1 at every position), otherwise the rule's factor at each row's position. The
    // full layers share one rule, so the table goes up once a walk: an upload from pageable memory
    // waits for the stream, and one a layer left the GPU idle between layers.
    let mut uploaded_tau: Option<TauRule> = None;
    let mut tau_table = |rule: TauRule| -> Result<Deviceptr, BackendError> {
        if rule == TauRule::One {
            return Ok(Deviceptr(0));
        }
        if uploaded_tau != Some(rule) {
            let taus: Vec<f32> = row_positions.iter().map(|&at| rule.at(at)).collect();
            run(shared
                .cuda
                .copy_htod(&scratch.taus, &row_bytes(&taus))
                .map_err(|error| error.to_string()))?;
            uploaded_tau = Some(rule);
        }
        Ok(scratch.taus)
    };

    for (index, layer) in schedule.layers().iter().enumerate() {
        // The drafter's taps of this layer: every row's stream as the layer starts on it.
        for &(sink, first, count) in &sinks {
            if let Some(slot) = sink.slot(layer.index) {
                sink.capture(
                    shared,
                    scratch
                        .x
                        .offset(first * hidden * std::mem::size_of::<f32>()),
                    count,
                    slot,
                )?;
            }
        }
        let sliding = layer.sliding;
        let cache_rows = layer.kv_rows;
        let clips = shared.clip[index];
        let mut previous_conv: Option<ConvSlot> = None;
        // Whether this layer's band already gathered its mix rows into `mixed` (run_band_halves),
        // which is then what its ExpertDelta collective would have done.
        let mut band_gathered = false;
        let mut cursor = 0usize;
        while cursor < layer.steps.len() {
            let step = &layer.steps[cursor];
            match (step.op, step.target) {
                (Op::BlockResidual, Target::Input(ResidualInput::Attention)) => {
                    let queries = weight(step.weights[0]);
                    residual(
                        &mut chain,
                        chunk.summaries,
                        scratch.x,
                        queries,
                        if sliding { 1 } else { 3 },
                        scratch.rows,
                        blocks,
                        0,
                        rows,
                    )?;
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
                    let io = |projection: Projection| match projection {
                        Projection::Query => Ok((scratch.q_in, scratch.q, attention_width, hidden)),
                        Projection::Gate => {
                            Ok((scratch.q_in, scratch.gate, attention_width, hidden))
                        }
                        Projection::Key => Ok((
                            if sliding { scratch.q_in } else { scratch.k_in },
                            scratch.k,
                            kv_width,
                            hidden,
                        )),
                        Projection::Value => Ok((
                            if sliding { scratch.q_in } else { scratch.v_in },
                            scratch.v,
                            kv_width,
                            hidden,
                        )),
                        Projection::Relative => {
                            Ok((scratch.q_in, scratch.rel, heads * config.d_rel, hidden))
                        }
                        Projection::Sink => Ok((scratch.q_in, scratch.sink, heads, hidden)),
                        Projection::AttentionOut => {
                            Ok((scratch.attn, scratch.delta, hidden, attention_width))
                        }
                        Projection::Router => Ok((
                            scratch.q_in,
                            scratch.router_logits,
                            config.n_routed_experts,
                            hidden,
                        )),
                        Projection::Head => Err(device_error("the head is an epilogue step")),
                    };
                    let (input, out, width, inner) = io(projection)?;
                    // A few rows on the batched GEMV (a verify, a handful of lanes) take the
                    // projections that follow over the same input with this one, up to four a
                    // launch, as the decode step does: a sliding layer's six attention
                    // projections are two launches rather than six. Each row's sums are its own
                    // launch's bits (Shared::gemv_fused).
                    if walk == PrefillWalk::Chunk && rows <= BATCH_ROWS {
                        let mut parts = [(Deviceptr(0), Deviceptr(0), 0usize); GEMV_FUSED];
                        parts[0] = (matrix, out, width);
                        let mut fused = 1;
                        while fused < GEMV_FUSED {
                            let Some(next) = layer.steps.get(cursor + fused) else {
                                break;
                            };
                            let (Op::Gemv, Target::Project(following)) = (next.op, next.target)
                            else {
                                break;
                            };
                            let (next_input, next_out, next_width, next_inner) = io(following)?;
                            if next_input.0 != input.0 || next_inner != inner {
                                break;
                            }
                            parts[fused] = (weight(next.weights[0]), next_out, next_width);
                            fused += 1;
                        }
                        if fused > 1 {
                            run(shared.gemv_fused(&parts[..fused], input, inner, rows))?;
                            cursor += fused;
                            continue;
                        }
                    }
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
                        Rows::Chunk(state) | Rows::Verify(state, _) => {
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
                            // A verify keeps its rows instead, and the round's settle restores the
                            // history from them.
                            if !keep_tail(shared, &sequences, chunk, index, slot_index, rows)? {
                                run(shared.conv_history(
                                    chunk
                                        .conv_head(slot_index)
                                        .offset(rows * channels * std::mem::size_of::<f32>()),
                                    state.conv[index][slot_index],
                                    channels,
                                    false,
                                ))?;
                            }
                        }
                        Rows::Segments(segments) => {
                            for (segment, &first) in segments.iter().zip(&offsets) {
                                let tail = segment.tails.at(index, slot_index);
                                segment_input(
                                    shared, chunk, segment, index, slot_index, first, channels,
                                )?;
                                run(shared.short_conv_rows(
                                    tail,
                                    taps,
                                    out.offset(first * channels * std::mem::size_of::<f32>()),
                                    segment.rows,
                                    channels,
                                    history,
                                ))?;
                            }
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
                        Rows::Chunk(state) | Rows::Verify(state, _) => {
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
                            if !keep_tail(shared, &sequences, chunk, index, slot_index, rows)? {
                                run(shared.conv_history(
                                    chunk
                                        .conv_head(slot_index)
                                        .offset(rows * kv_width * std::mem::size_of::<f32>()),
                                    state.conv[index][slot_index],
                                    kv_width,
                                    false,
                                ))?;
                            }
                        }
                        Rows::Segments(segments) => {
                            let byte = std::mem::size_of::<f32>();
                            for (segment, &first) in segments.iter().zip(&offsets) {
                                let tail = segment.tails.at(index, slot_index);
                                segment_input(
                                    shared, chunk, segment, index, slot_index, first, kv_width,
                                )?;
                                run(shared.short_conv_dynamic_rows(
                                    tail,
                                    static_taps,
                                    scratch.taps.offset(first * taps * byte),
                                    out.offset(first * kv_width * byte),
                                    segment.rows,
                                    kv_width,
                                    history,
                                ))?;
                            }
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
                    // A verify's rows land where its decode steps would store them; the rows it
                    // does not keep are past the position the session leaves, and the next rows
                    // stored there overwrite them before any row reads them.
                    Rows::Chunk(state) | Rows::Verify(state, _) => run(shared.launch(
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
                    Rows::Segments(segments) => {
                        // Each segment's rows land where its verify alone would store them.
                        let byte = std::mem::size_of::<f32>();
                        for (segment, &first) in segments.iter().zip(&offsets) {
                            let at = first * kv_width * byte;
                            let state = &segment.state;
                            run(shared.launch(
                                shared.kernels.kv_store,
                                (flat(kv_width), segment.rows as u32, 1),
                                (ELEMENT_BLOCK, 1, 1),
                                &[
                                    scratch.k_next.offset(at).0,
                                    scratch.v_next.offset(at).0,
                                    state.keys[index].0,
                                    state.values[index].0,
                                    segment.rows as u64,
                                    state.position as u64,
                                    kv_width as u64,
                                    if sliding { cache_rows as u64 } else { 0 },
                                ],
                            ))?;
                        }
                    }
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
                            let taus = tau_table(layer.tau)?;
                            // Under the overlap the output, its projection and its delta's sum
                            // run as two halves of the rows (attention_half), and the walk goes on
                            // past them.
                            let split = attention_split(shared, walk, layer, cursor, rows);
                            let overlap = match (split, shared.overlap.as_ref()) {
                                (Some(_), Some(overlap)) => Some(overlap.lock().map_err(|_| {
                                    device_error("the band overlap's lock is poisoned")
                                })?),
                                _ => None,
                            };
                            let mut half = |first_row: usize, count: usize| match (
                                overlap.as_deref(),
                                split,
                            ) {
                                (Some(overlap), Some((_, projection))) => attention_half(
                                    shared,
                                    scratch,
                                    walk,
                                    weight(projection.weights[0]),
                                    &overlap.side,
                                    first_row,
                                    count,
                                )
                                .map_err(|error| error.to_string()),
                                _ => Err("an attention half with no overlap".to_owned()),
                            };
                            let halves = split.map(|(rows, _)| rows).zip(Some(
                                &mut half as &mut dyn FnMut(usize, usize) -> Result<(), String>,
                            ));
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
                                taus,
                                rows,
                                position,
                                kv_heads,
                                groups,
                                if sliding { layer.window } else { 0 },
                                if sliding { cache_rows } else { 0 },
                                layer.rel_extent,
                                layer.tau.at(position),
                                halves,
                            ))?;
                            if let Some(overlap) = overlap.as_deref() {
                                // The convolution reads the summed delta.
                                run(shared.cuda.side_mark(&overlap.side, &overlap.summed))?;
                                run(shared.cuda.run_wait(&overlap.summed))?;
                                cursor += 4;
                                continue;
                            }
                        }
                        Rows::Verify(state, _) => {
                            // The decode step's attention, each row against its own prefix: the
                            // split arena's passes are the decode step's, and the rows' scales are
                            // the plan's rule at each row's position, as the decode step takes it.
                            let taus = tau_table(layer.tau)?;
                            run(shared.attention_rows(
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
                                taus,
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
                        Rows::Segments(segments) => {
                            // Each segment's rows as its verify alone runs them: the decode
                            // step's attention, each row against its own prefix, the segments
                            // one after another through the walk's split arena.
                            let taus = tau_table(layer.tau)?;
                            let byte = std::mem::size_of::<f32>();
                            let query_width = heads * head_dim;
                            for (segment, &first) in segments.iter().zip(&offsets) {
                                let state = &segment.state;
                                let at = state.position;
                                let segment_taus = if taus.0 == 0 {
                                    taus
                                } else {
                                    taus.offset(first * byte)
                                };
                                run(shared.attention_rows(
                                    scratch,
                                    index,
                                    scratch.q.offset(first * query_width * byte),
                                    scratch.gate.offset(first * query_width * byte),
                                    state.keys[index],
                                    state.values[index],
                                    state.tables[index],
                                    scratch.rel.offset(first * heads * config.d_rel * byte),
                                    bank,
                                    scratch.sink.offset(first * heads * byte),
                                    clips,
                                    scratch.attn_bf16.offset(first * query_width * 2),
                                    segment_taus,
                                    segment.rows,
                                    at,
                                    kv_heads,
                                    groups,
                                    if sliding { layer.window } else { 0 },
                                    if sliding { cache_rows } else { 0 },
                                    layer.rel_extent,
                                    layer.tau.at(at),
                                ))?;
                            }
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
                    residual(
                        &mut chain,
                        chunk.summaries,
                        scratch.x,
                        queries,
                        1,
                        scratch.rows,
                        blocks,
                        0,
                        rows,
                    )?;
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
                    let mma = match layer.experts {
                        ExpertBody::NvFp4 => true,
                        ExpertBody::MxFp8 => shared.mxfp8_walk == Mxfp8Walk::Mma,
                        ExpertBody::Bf16 => false,
                    };
                    if walk == PrefillWalk::Tensor && mma {
                        band_gathered = run_band_mma(shared, chunk, layer, rows)?;
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
                    // the sum, which is the number the row-at-a-time walk's own mix prints (the
                    // row split's rows wait in a staging run, so it reads the exchanged row).
                    let gathers = BandOut::of(shared, scratch, rows).gathers;
                    if index == 0 && !gathers {
                        probe(shared, "chunk band", scratch.mixed, rows, hidden)?;
                    }
                    if !band_gathered {
                        band_exchange(shared, scratch, rows)?;
                    }
                    if index == 0 && gathers {
                        probe(shared, "chunk band", scratch.mixed, rows, hidden)?;
                    }
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
    let mut output_queries: Option<Deviceptr> = None;
    for step in schedule.epilogue() {
        match (step.op, step.target) {
            (Op::BlockResidual, Target::Input(ResidualInput::Output)) => {
                let queries = weight(step.weights[0]);
                output_queries = Some(queries);
                residual(
                    &mut chain,
                    chunk.summaries.offset(first * summary_stride * byte),
                    scratch.x.offset(first * hidden * byte),
                    queries,
                    1,
                    scratch.final_row,
                    blocks,
                    first,
                    answers,
                )?;
            }
            (Op::RmsNorm, Target::Norm(NormRole::Final)) => {
                let gain = weight(step.weights[0]);
                run(shared.rmsnorm(scratch.final_row, gain, scratch.final_row, answers, hidden))?;
                if let Some(&(sink, _, _)) = sinks.first() {
                    if answers == rows {
                        for &(sink, first, count) in &sinks {
                            sink.capture(
                                shared,
                                scratch.final_row.offset(first * hidden * byte),
                                count,
                                FINAL_TAP,
                            )?;
                        }
                    } else {
                        // A chunk answers its last row, and the drafter's last tap is every row's
                        // normed output: the same per-row chain over all of them, into a buffer
                        // the epilogue no longer reads.
                        let queries = output_queries.ok_or_else(|| {
                            device_error("the epilogue norms before its output residual")
                        })?;
                        residual(
                            &mut chain,
                            chunk.summaries,
                            scratch.x,
                            queries,
                            1,
                            scratch.q_in,
                            blocks,
                            0,
                            rows,
                        )?;
                        run(shared.rmsnorm(scratch.q_in, gain, scratch.q_in, rows, hidden))?;
                        sink.capture(shared, scratch.q_in, rows, FINAL_TAP)?;
                    }
                }
            }
            (Op::Gemv, Target::Project(Projection::Head)) => {
                let matrix = weight(step.weights[0]);
                // The head's one-to-eight rows take the wide tile's few-row body
                // (q4p_dense_rows), so a decode or verify row is the prompt tile's row; a wider
                // walk and a shape the body does not cover keep the row GEMV.
                let body = if answers <= BATCH_ROWS {
                    shared
                        .dense_rows(
                            matrix,
                            scratch.final_row,
                            scratch.logits,
                            rank_vocab,
                            hidden,
                            answers,
                        )
                        .map_err(|_| ())
                } else {
                    Err(())
                };
                if body.is_err() {
                    match &sequences {
                        Rows::Chunk(_) => run(shared.gemv(
                            matrix,
                            scratch.final_row,
                            scratch.logits,
                            rank_vocab,
                            hidden,
                        ))?,
                        // The lanes' or a verify's rows through the batched GEMV, whose row is
                        // the one-row launch's bits.
                        Rows::Lanes(_) | Rows::Verify(..) | Rows::Segments(_) => project(
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
                    (flat(unpadded), answers as u32, 1),
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
        Rows::Chunk(state) | Rows::Verify(state, _) => {
            state.blocks = blocks;
            state.position += rows;
        }
        Rows::Lanes(states) => {
            for state in states.iter_mut() {
                state.blocks = blocks;
                state.position += 1;
            }
        }
        Rows::Segments(segments) => {
            for segment in segments.iter_mut() {
                segment.state.blocks = blocks;
                segment.state.position += segment.rows;
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

/// A verify's copy of one layer's convolution input as the launch read it: the history rows
/// before its first row, then its own rows. Nothing for any other walk. Answers whether it kept
/// them: a verify leaves the sequence's history as it was, for the round's settle to restore
/// from these rows, and every other walk writes its tail back to the history itself.
fn keep_tail(
    shared: &Shared,
    sequences: &Rows<'_, '_>,
    chunk: &Chunk,
    layer: usize,
    slot: usize,
    rows: usize,
) -> Result<bool, BackendError> {
    let Rows::Verify(_, tails) = sequences else {
        return Ok(false);
    };
    shared
        .cuda
        .copy_dtod_async(
            &tails.at(layer, slot),
            &chunk.conv_head(slot),
            (tails.history + rows) * tails.channels(slot) * std::mem::size_of::<f32>(),
        )
        .map_err(device_error)?;
    Ok(true)
}

/// A segment's convolution input in its own tails, as a verify's launch reads it: the sequence's
/// history rows, then the segment's rows of the walk's input (`first` on). The launch then reads
/// the tails in place, and they are what the sequence's settle restores from.
fn segment_input(
    shared: &Shared,
    chunk: &Chunk,
    segment: &Segment<'_>,
    layer: usize,
    slot: usize,
    first: usize,
    channels: usize,
) -> Result<(), BackendError> {
    let byte = std::mem::size_of::<f32>();
    let tail = segment.tails.at(layer, slot);
    shared
        .conv_history(segment.state.conv[layer][slot], tail, channels, true)
        .map_err(device_error)?;
    let history = segment.tails.history;
    shared
        .cuda
        .copy_dtod_async(
            &tail.offset(history * channels * byte),
            &chunk
                .conv_head(slot)
                .offset((history + first) * channels * byte),
            segment.rows * channels * byte,
        )
        .map_err(device_error)
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

// The grouped MMA's tile, item count and plan are the shared rule (sparkflash_ops::moe).

/// The routed band over a chunk on the tensor cores, for an NVFP4 layer under the `tensor` walk
/// and an MXFP8 one under [Mxfp8Walk::Mma].
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
/// step's bits. The MXFP8 layers take the same body over their E4M3 bytes and ue8m0 codes
/// (q4p_moe_gate_up_capped_mx8, q4p_moe_down_mx8), each weight the pair walk's f32 product, exact
/// in bf16 under the codes the open admitted, with ones for the outer scales.
fn run_band_mma(
    shared: &Shared,
    chunk: &Chunk,
    layer: &LayerSchedule,
    rows: usize,
) -> Result<bool, BackendError> {
    let config = &shared.config;
    let scratch = &chunk.scratch;
    let top_k = config.num_experts_per_tok;
    let intermediate = shared.plan.expert_intermediate();
    let hidden = config.hidden_size;
    let (first, owned) = shared.plan.expert_range();
    let down = shared.plan.expert_down();
    let meta = shared.mma[layer.index].ok_or_else(|| {
        device_error(format!(
            "layer {}: a {:?} band with no grouped-MMA table",
            layer.index, layer.experts
        ))
    })?;
    let mx8 = layer.experts == ExpertBody::MxFp8;
    let gate_up_kernel = if mx8 {
        shared.kernels.moe_gate_up_mx8
    } else {
        shared.kernels.moe_gate_up
    };
    if !grouped::fits(hidden, intermediate, (down.rows, down.columns), owned) {
        return Err(device_error(format!(
            "layer {}: the grouped MMA's tiles do not cover hidden {hidden}, intermediate \
             {intermediate}, a down projection of {} by {} and {owned} experts a rank; \
             PREFILL=chunk walks this band",
            layer.index, down.rows, down.columns
        )));
    }
    let slots = rows * top_k;
    let elements = rows * hidden;
    let items = grouped::items(slots, owned);
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
                (elements / 2)
                    .div_ceil(grouped::THREADS as usize)
                    .clamp(1, 4096) as u32,
                1,
                1,
            ),
            (grouped::THREADS, 1, 1),
            &[scratch.q_in.0, scratch.band.0, elements as u64],
        )
        .map_err(device_error)?;
    let out = BandOut::of(shared, scratch, rows);
    if out.gathers
        && rows >= OVERLAP_ROWS
        && let Some(overlap) = shared.overlap.as_ref()
    {
        let overlap = overlap
            .lock()
            .map_err(|_| device_error("the band overlap's lock is poisoned"))?;
        let band = HalvesBand {
            meta,
            gate_up: gate_up_kernel,
            down: if mx8 {
                shared.kernels.moe_down_runs_mx8
            } else {
                shared.kernels.moe_down_runs
            },
        };
        run_band_halves(shared, scratch, &band, &overlap, rows)?;
        return Ok(true);
    }
    // Another rank's slot is left out; its share is zero, so the mix skips its row.
    let plan = MoePlanCall::items(
        scratch.local.0,
        scratch.plan.0,
        (scratch.items.0, scratch.item_count.0),
        (slots, owned, top_k),
        grouped::ITEM_ROWS,
        items,
    );
    let (grid, block) = MoePlanCall::launch();
    shared
        .launch(shared.kernels.moe_plan, grid, block, &plan.args())
        .map_err(device_error)?;
    let grid_items = u32::try_from(items).map_err(device_error)?;
    // The SwiGLU rows' two bf16 planes fill the activation buffer's f32 rows exactly. Under the
    // row split they are this rank's run of every slot's columns, its planes together, so the
    // gather moves them as planes * slots rows of a run each (the head-only operand writes and
    // moves one plane).
    let planes = shared.planes.count();
    let run = planes * slots * intermediate * std::mem::size_of::<u16>();
    let activated = if out.gathers {
        scratch.partials.offset(shared.exchange.rank() * run)
    } else {
        scratch.activation
    };
    shared
        .launch(
            gate_up_kernel,
            (
                (intermediate / grouped::GATE_UP_COLUMNS) as u32,
                grid_items,
                1,
            ),
            (grouped::THREADS, 1, 1),
            &[
                meta.table.0,
                meta.scales.0,
                scratch.plan.0,
                scratch.items.0,
                scratch.item_count.0,
                scratch.band.0,
                elements as u64,
                activated.0,
                (slots * intermediate) as u64,
                top_k as u64,
                hidden as u64,
                intermediate as u64,
                bits(config.mlp_gate_softcap as f32),
                bits(config.mlp_up_softcap as f32),
            ],
        )
        .map_err(device_error)?;
    // Under the row split the down pass reads the gathered runs where they land (rank r's run of
    // both planes at r * run, a slot's row `intermediate` wide): the stages copy the elements a
    // joined row holds at their columns, so no join copies the activation first.
    let grid = ((down.rows / grouped::DOWN_COLUMNS) as u32, grid_items, 1);
    let head = [
        meta.table.0,
        meta.scales.0,
        scratch.plan.0,
        scratch.items.0,
        scratch.item_count.0,
    ];
    let tail = [
        shared.route_ones.0,
        out.down.0,
        top_k as u64,
        down.rows as u64,
        down.columns as u64,
        // Every slot reads the same [top_k] ones.
        0,
    ];
    if out.gathers {
        gather_runs(
            shared,
            scratch.partials,
            None,
            planes * slots,
            intermediate * std::mem::size_of::<u16>(),
        )?;
        let kernel = if mx8 {
            shared.kernels.moe_down_runs_mx8
        } else {
            shared.kernels.moe_down_runs
        };
        let source = [scratch.partials.0, (slots * intermediate) as u64];
        let runs = [intermediate as u64, (planes * slots * intermediate) as u64];
        let args: Vec<u64> = [&head[..], &source, &tail, &runs].concat();
        shared
            .launch(kernel, grid, (grouped::THREADS, 1, 1), &args)
            .map_err(device_error)?;
    } else {
        let kernel = if mx8 {
            shared.kernels.moe_down_mx8
        } else {
            shared.kernels.moe_down
        };
        let source = [scratch.activation.0, (slots * down.columns) as u64];
        let args: Vec<u64> = [&head[..], &source, &tail].concat();
        shared
            .launch(kernel, grid, (grouped::THREADS, 1, 1), &args)
            .map_err(device_error)?;
    }
    // The mix is local; the exchange that follows it completes the sum over the ranks.
    band_mix(shared, scratch, rows)?;
    Ok(false)
}

/// The fewest rows whose row-split band runs as two halves ([run_band_halves]): below it a
/// half's gathers are too short to hide anything behind.
const OVERLAP_ROWS: usize = 512;

/// Where a chunk's attention at `cursor` splits into the halves [attention_half] runs, and the
/// output projection's step; `None` runs it whole. The halves run under the overlap (the row
/// split's tensor walk on the device leg), the tensor-core attention, a chunk of at least
/// [OVERLAP_ROWS] rows, and an attention the widen, the output projection and the delta's sum
/// follow, as the plan lays every layer out. The trace reads the projection back before its sum,
/// so it keeps the whole form.
fn attention_split<'l>(
    shared: &Shared,
    walk: PrefillWalk,
    layer: &'l LayerSchedule,
    cursor: usize,
    rows: usize,
) -> Option<(usize, &'l Step)> {
    if shared.overlap.is_none()
        || shared.exchange_slots.is_none()
        || walk != PrefillWalk::Tensor
        || shared.attention != AttentionWalk::Fast
        || trace_on()
    {
        return None;
    }
    let [widen, projection, sum] = layer.steps.get(cursor + 1..cursor + 4)? else {
        return None;
    };
    let laid_out = matches!(widen.op, Op::Widen)
        && matches!(
            (projection.op, projection.target),
            (Op::Gemv, Target::Project(Projection::AttentionOut))
        )
        && matches!(sum.op, Op::Collective(Collective::AttentionDelta));
    if !laid_out {
        return None;
    }
    Some((
        split_rows(rows, shared.hidden(), shared.exchange.world())?,
        projection,
    ))
}

/// [attention_split]'s row: half the chunk's rows, down to an edge of the tensor-core walk's
/// blocks, where each half's sum takes [exchange_sum]'s two shots.
fn split_rows(rows: usize, hidden: usize, world: usize) -> Option<usize> {
    if rows < OVERLAP_ROWS {
        return None;
    }
    let split = rows / 2 / MMA_Q * MMA_Q;
    (takes_two_shots(split * hidden, world) && takes_two_shots((rows - split) * hidden, world))
        .then_some(split)
}

/// One half of a chunk's attention output, once its MMA is queued (Shared::attention_mma under
/// halves): the widen and the output projection of its rows on the run's stream, then its delta's
/// two-shot sum on the overlap's side stream behind them, so half 0's sum crosses the wire under
/// half 1's attention. The widen is per element, every output of the projection is its own row's
/// dot in its own K order, and every element's sum is the one fold in the one order whichever rows
/// share its launch, so the halves give every bit the whole form gives. Half 1's slots are past
/// half 0's, so none of its pieces lands where half 0's sum still reads. The walk marks the side
/// stream after the second half and the run's stream waits for it before the convolution.
fn attention_half(
    shared: &Shared,
    scratch: &Scratch,
    walk: PrefillWalk,
    matrix: Deviceptr,
    side: &SideStream,
    first_row: usize,
    rows: usize,
) -> Result<(), BackendError> {
    let hidden = shared.hidden();
    let width = shared.heads() * shared.config.head_dim;
    let float = std::mem::size_of::<f32>();
    let attended = scratch.attn.offset(first_row * width * float);
    shared
        .launch(
            shared.kernels.widen,
            (flat(rows * width), 1, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                scratch
                    .attn_bf16
                    .offset(first_row * width * std::mem::size_of::<u16>())
                    .0,
                attended.0,
                (rows * width) as u64,
            ],
        )
        .map_err(device_error)?;
    let delta = scratch.delta.offset(first_row * hidden * float);
    project(shared, walk, matrix, attended, delta, hidden, width, rows)?;
    let slots = shared
        .exchange_slots
        .ok_or_else(|| device_error("an attention half's sum has no device-leg slots"))?;
    two_shots(
        shared,
        Some(side),
        delta,
        slots.offset(first_row * hidden * float),
        rows * hidden,
    )
}

/// The kernels and table one layer's halves run.
struct HalvesBand {
    meta: MmaTable,
    gate_up: Entry,
    /// The down pass over the gathered runs (q4p_moe_down_runs or its MXFP8 form).
    down: Entry,
}

/// One half's places in the chunk's scratch ([run_band_halves]).
struct HalfPlace {
    /// 0 or 1: which of the two item counts the half's plan writes.
    half: usize,
    first_row: usize,
    rows: usize,
    first_slot: usize,
    slots: usize,
    items: usize,
    first_item: usize,
    /// The half's activation runs in `partials`, [rank][plane][slot][intermediate].
    runs: Deviceptr,
    /// The half's mix rows in `activation`, [rank][row][down rows].
    mixes: Deviceptr,
}

/// The row split's band over a chunk as two halves of its rows, each half's two gathers on the
/// overlap's side stream while the run's stream computes the other half.
///
/// The run's stream: gate_up of half 0, gate_up of half 1, down and mix of half 0 once its
/// activation is gathered, down and mix of half 1 once its activation is, then each half's join
/// once its mix rows are gathered. The side stream, in the same order on every rank: half 0's
/// activation gather (after its gate_up), half 1's (after its gate_up), half 0's mix gather
/// (after its mix), half 1's (after its mix). So half 0's activation crosses the wire under half
/// 1's gate_up, half 1's under half 0's down and mix, and half 0's mix rows under half 1's down.
/// Every collective of the band is on the side stream, the side stream follows the run's stream
/// up to each gather, and the run's stream waits for the last of them before the next layer's
/// exchange, so the group still issues its collectives one at a time in one order.
///
/// A half is planned and run as a chunk of its own rows would be. Every output element of the
/// grouped MMA depends on its own slot's row and its K order alone (the chunk-width invariance
/// the tensor walk keeps), and the mix and the join are per row, so the halves give every bit the
/// whole chunk's band gives. Only where the runs wait differs: the activation runs are half-major
/// in `partials`, and the mix rows wait in `activation` (which the down pass over the runs leaves
/// unread), so half 0's mix rows cannot land on half 1's runs before half 1's down pass reads
/// them.
fn run_band_halves(
    shared: &Shared,
    scratch: &Scratch,
    band: &HalvesBand,
    overlap: &BandOverlap,
    rows: usize,
) -> Result<(), BackendError> {
    let config = &shared.config;
    let cuda = &shared.cuda;
    let top_k = config.num_experts_per_tok;
    let intermediate = shared.plan.expert_intermediate();
    let hidden = config.hidden_size;
    let (_, owned) = shared.plan.expert_range();
    let down = shared.plan.expert_down();
    let (rank, world) = (shared.exchange.rank(), shared.exchange.world());
    let int = std::mem::size_of::<i32>();
    let word = std::mem::size_of::<f32>();
    // A slot's activation run: `planes` bf16 values a column.
    let planes = shared.planes.count();
    let element = planes * std::mem::size_of::<u16>();
    let elements = rows * hidden;
    let mut places = Vec::with_capacity(2);
    let (mut first_item, mut runs, mut mixes) = (0, scratch.partials, scratch.activation);
    for (half, (first_row, half_rows)) in [(0, rows / 2), (rows / 2, rows - rows / 2)]
        .into_iter()
        .enumerate()
    {
        let slots = half_rows * top_k;
        let items = grouped::items(slots, owned);
        places.push(HalfPlace {
            half,
            first_row,
            rows: half_rows,
            first_slot: first_row * top_k,
            slots,
            items,
            first_item,
            runs,
            mixes,
        });
        first_item += items;
        runs = runs.offset(world * slots * intermediate * element);
        mixes = mixes.offset(world * half_rows * down.rows * word);
    }
    let plan_of = |place: &HalfPlace| {
        (
            scratch.plan.offset(place.first_slot * 2 * int),
            scratch.items.offset(place.first_item * 4 * int),
            scratch.item_count.offset(place.half * int),
        )
    };
    let gate_up = |place: &HalfPlace| -> Result<(), BackendError> {
        let (plan, items, count) = plan_of(place);
        let call = MoePlanCall::items(
            scratch.local.offset(place.first_slot * int).0,
            plan.0,
            (items.0, count.0),
            (place.slots, owned, top_k),
            grouped::ITEM_ROWS,
            place.items,
        );
        let (grid, block) = MoePlanCall::launch();
        shared
            .launch(shared.kernels.moe_plan, grid, block, &call.args())
            .map_err(device_error)?;
        let run = place.slots * intermediate * element;
        shared
            .launch(
                band.gate_up,
                (
                    (intermediate / grouped::GATE_UP_COLUMNS) as u32,
                    u32::try_from(place.items).map_err(device_error)?,
                    1,
                ),
                (grouped::THREADS, 1, 1),
                &[
                    band.meta.table.0,
                    band.meta.scales.0,
                    plan.0,
                    items.0,
                    count.0,
                    // The half's rows of the input's head plane; its remainder plane is the whole
                    // chunk's `elements` on, where split_bf16 wrote it.
                    scratch
                        .band
                        .offset(place.first_row * hidden * std::mem::size_of::<u16>())
                        .0,
                    elements as u64,
                    place.runs.offset(rank * run).0,
                    (place.slots * intermediate) as u64,
                    top_k as u64,
                    hidden as u64,
                    intermediate as u64,
                    bits(config.mlp_gate_softcap as f32),
                    bits(config.mlp_up_softcap as f32),
                ],
            )
            .map_err(device_error)
    };
    let down_and_mix = |place: &HalfPlace| -> Result<(), BackendError> {
        let (plan, items, count) = plan_of(place);
        // The band's input is read by both halves' gate_up passes, which the run's stream has
        // finished by now, so the projection lands in the band as the whole chunk's does.
        let projected = scratch.band.offset(place.first_slot * down.rows * word);
        shared
            .launch(
                band.down,
                (
                    (down.rows / grouped::DOWN_COLUMNS) as u32,
                    u32::try_from(place.items).map_err(device_error)?,
                    1,
                ),
                (grouped::THREADS, 1, 1),
                &[
                    band.meta.table.0,
                    band.meta.scales.0,
                    plan.0,
                    items.0,
                    count.0,
                    place.runs.0,
                    (place.slots * intermediate) as u64,
                    shared.route_ones.0,
                    projected.0,
                    top_k as u64,
                    down.rows as u64,
                    down.columns as u64,
                    0,
                    intermediate as u64,
                    (planes * place.slots * intermediate) as u64,
                ],
            )
            .map_err(device_error)?;
        shared
            .launch(
                shared.kernels.expert_mix,
                (flat(down.rows), place.rows as u32, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    place.mixes.offset(rank * place.rows * down.rows * word).0,
                    projected.0,
                    scratch.scales.offset(place.first_slot * word).0,
                    place.rows as u64,
                    top_k as u64,
                    down.rows as u64,
                    scratch.indices.offset(place.first_slot * int).0,
                    shared.plan.parallel_experts() as u64,
                    world as u64,
                ],
            )
            .map_err(device_error)
    };
    let side = &overlap.side;
    let fail = |error: String| device_error(error);
    // A half's `run` bytes a rank at `base`, gathered on the side stream once the run's stream
    // has written this rank's, and `done` marked behind them.
    let gather = |base: Deviceptr, run: usize, done: &StreamEvent| -> Result<(), BackendError> {
        cuda.side_follow_run(side).map_err(fail)?;
        shared
            .exchange
            .exchange_on_device(side.raw(), base.0 + (rank * run) as u64, base.0, run)
            .map_err(device_error)?;
        cuda.side_mark(side, done).map_err(fail)
    };
    for place in &places {
        gate_up(place)?;
        gather(
            place.runs,
            place.slots * intermediate * element,
            &overlap.gathered[place.half],
        )?;
    }
    for place in &places {
        cuda.run_wait(&overlap.gathered[place.half]).map_err(fail)?;
        down_and_mix(place)?;
        gather(
            place.mixes,
            place.rows * down.rows * word,
            &overlap.mixed[place.half],
        )?;
    }
    // The open refuses a split whose pieces are not whole 16-byte words.
    let words = down.rows * word / 16;
    for place in &places {
        cuda.run_wait(&overlap.mixed[place.half]).map_err(fail)?;
        shared
            .launch(
                shared.kernels.join_pieces,
                (flat(place.rows * world * words), 1, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    scratch.mixed.offset(place.first_row * hidden * word).0,
                    place.mixes.0,
                    place.rows as u64,
                    words as u64,
                    world as u64,
                ],
            )
            .map_err(device_error)?;
    }
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
    let intermediate = shared.plan.expert_intermediate();
    let hidden = config.hidden_size;
    let weights = &shared.weights;
    let weight = |index: usize| weights.get(index).pointer;
    let _ = activation;
    // A few rows (a verify, small fused lanes) pick few enough experts that grouping them buys
    // nothing, and the grouping's readback made each such layer drain the device and wait for the
    // host: those take the decode step's own steered band.
    if rows <= STEERED_BAND_ROWS && layer.experts != ExpertBody::Bf16 {
        return run_steered_band(
            shared,
            scratch,
            layer.index,
            gate_up,
            down,
            scratch.q_in,
            rows,
        );
    }

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
        || !shared.plan.expert_down().columns.is_multiple_of(PAIR_RUN)
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
    let shape = shared.plan.expert_down();
    let out = BandOut::of(shared, scratch, rows);
    let expert_bytes = codes_bytes(2 * intermediate, hidden, layer.experts);
    let expert_scale_bytes = scales_bytes(2 * intermediate, hidden, layer.experts);
    let down_bytes = codes_bytes(shape.rows, shape.columns, layer.experts);
    let down_scale_bytes = scales_bytes(shape.rows, shape.columns, layer.experts);

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
    // Under the row split each slot's activation is this rank's run of its columns, which the
    // gather joins into whole rows before the down projection.
    let slots = rows * top_k;
    let byte = std::mem::size_of::<f32>();
    let activated = if out.gathers {
        scratch
            .partials
            .offset(shared.exchange.rank() * slots * intermediate * byte)
    } else {
        scratch.activation
    };
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
                    activated.offset(first_slot * intermediate * byte).0,
                    intermediate as u64,
                    bits(config.mlp_gate_softcap as f32),
                    bits(config.mlp_up_softcap as f32),
                ],
            )
            .map_err(device_error)?;
        first_slot += take;
    }
    if out.gathers {
        gather_runs(
            shared,
            scratch.partials,
            Some(scratch.activation),
            slots,
            intermediate * byte,
        )?;
    }

    if layer.index == 0 {
        probe_band(
            shared,
            "chunk swiglu",
            scratch.activation,
            &entries,
            shape.columns,
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
                                .offset(slot * shape.columns * std::mem::size_of::<f32>()),
                            out.down
                                .offset(slot * shape.rows * std::mem::size_of::<f32>()),
                            shape.rows,
                            shape.columns,
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
                        (pair_grid(shape.rows), 1, 1),
                        (ELEMENT_BLOCK, 1, 1),
                        &[
                            down_codes.offset(local * down_bytes).0,
                            scale_pointer.0,
                            scratch.activation.0,
                            out.down.0,
                            shape.rows as u64,
                            shape.columns as u64,
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
        probe_band(shared, "chunk down", out.down, &entries, shape.rows)?;
    }

    // The mix is local; the exchange that follows it completes the sum over the ranks.
    band_mix(shared, scratch, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The routing band: one to eight rows take the few-row expert body, and the ninth, a
    /// BF16 layer, a head-only activation, a gathered band, a missing table or a shape the body
    /// does not cover keeps the row walk.
    #[test]
    fn the_decode_band_takes_the_expert_rows_body_from_one_to_eight_rows() {
        let fits = |rows, experts, planes, gathers, table, hidden, intermediate, down| {
            expert_rows_body_fits(
                rows,
                experts,
                planes,
                gathers,
                table,
                hidden,
                intermediate,
                down,
            )
        };
        for rows in 1..=super::super::BATCH_ROWS {
            assert!(
                fits(
                    rows,
                    ExpertBody::NvFp4,
                    ActivationPlanes::Both,
                    false,
                    true,
                    4096,
                    2048,
                    (4096, 2048)
                ),
                "{rows} rows take the NVFP4 body"
            );
            assert!(
                fits(
                    rows,
                    ExpertBody::MxFp8,
                    ActivationPlanes::Both,
                    false,
                    true,
                    4096,
                    2048,
                    (4096, 2048)
                ),
                "{rows} rows take the MXFP8 body"
            );
        }
        assert!(!fits(
            9,
            ExpertBody::NvFp4,
            ActivationPlanes::Both,
            false,
            true,
            4096,
            2048,
            (4096, 2048)
        ));
        assert!(!fits(
            1,
            ExpertBody::Bf16,
            ActivationPlanes::Both,
            false,
            true,
            4096,
            2048,
            (4096, 2048)
        ));
        assert!(!fits(
            1,
            ExpertBody::NvFp4,
            ActivationPlanes::Head,
            false,
            true,
            4096,
            2048,
            (4096, 2048)
        ));
        assert!(!fits(
            1,
            ExpertBody::NvFp4,
            ActivationPlanes::Both,
            true,
            true,
            4096,
            2048,
            (4096, 2048)
        ));
        assert!(!fits(
            1,
            ExpertBody::NvFp4,
            ActivationPlanes::Both,
            false,
            false,
            4096,
            2048,
            (4096, 2048)
        ));
        // The NVFP4 bodies walk whole 128-column K runs; the MXFP8 body whole 32-column ones.
        assert!(!fits(
            1,
            ExpertBody::NvFp4,
            ActivationPlanes::Both,
            false,
            true,
            4128,
            2048,
            (4096, 2048)
        ));
        // The MXFP8 body walks the tile's 64-column K stages.
        assert!(fits(
            1,
            ExpertBody::MxFp8,
            ActivationPlanes::Both,
            false,
            true,
            4160,
            2048,
            (4096, 2048)
        ));
        assert!(!fits(
            1,
            ExpertBody::MxFp8,
            ActivationPlanes::Both,
            false,
            true,
            4096,
            2048,
            (4096, 2000)
        ));
    }

    #[test]
    fn the_attention_halves_split_on_a_block_edge_where_both_sums_take_two_shots() {
        // Inkling-2-Small's residual over four ranks.
        let (hidden, world) = (4608, 4);
        assert_eq!(split_rows(4096, hidden, world), Some(2048));
        assert_eq!(split_rows(1000, hidden, world), Some(496));
        assert_eq!(split_rows(OVERLAP_ROWS - 1, hidden, world), None);
        for rows in OVERLAP_ROWS..=4096 {
            let split = split_rows(rows, hidden, world).expect("every chunk this wide splits");
            assert!(
                split.is_multiple_of(MMA_Q) && split > 0 && split < rows,
                "{rows}"
            );
        }
        // A half that is not four ranks' whole groups of four would take one shot: whole form.
        assert_eq!(split_rows(1000, 4605, world), None);
    }
}
