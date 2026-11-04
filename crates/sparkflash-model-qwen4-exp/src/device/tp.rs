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

//! The device arm over a tensor-parallel world.
//!
//! A rank holds `1 / world` of every attention block's query heads, of every Gated DeltaNet
//! block's key and value heads, and of every expert's intermediate width (the gate and up rows,
//! the down columns), and runs the rest whole: the embeddings, the hyper-connections, the router,
//! the indexer, the PLE, the LM head and the drafter. Each branch therefore leaves a partial of
//! its hidden row, and the ranks sum the partials in rank order before the row is injected
//! ([DeviceSequence::inject_branch]). The residual streams then hold the same bits on every rank,
//! so every replicated decision - the router's top-k, the indexer's blocks, the token, the
//! drafter's proposals - is the same on every rank without a second collective.
//!
//! The kernels run with the rank's own configuration ([rank_config]): the global one with its
//! sharded head counts and widths divided, so a launch sized from it covers this rank's share.

use std::sync::Arc;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, RunStream, StreamEvent};
use sparkflash_model::BackendError;
use sparkflash_weights::tp::{TpLayout, TpRuntime, Wire};

use super::{
    DENSE_ROWS, DeviceSequence, GEMM_PARTIAL_REGION, GpuHyper, GpuLayer, MAX_INTERMEDIATE,
    zero_split_tickets,
};
use crate::config::Qwen4ExpConfig;

/// The prefill MMA's expert tile (Q4P_BK and half of Q4P_BN): a rank's expert width is padded to
/// whole tiles with zero rows and columns, which add nothing to any sum.
const EXPERT_TILE: usize = 64;

/// Which share of each sharded tensor a rank holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shard {
    pub(crate) rank: usize,
    pub(crate) world: usize,
}

impl Shard {
    /// The whole model: one rank of one.
    pub(crate) const WHOLE: Self = Self { rank: 0, world: 1 };

    pub(crate) fn of(layout: TpLayout) -> Self {
        Self {
            rank: layout.rank,
            world: layout.world_size,
        }
    }

    /// This rank's `(first, count)` of `total` items split evenly; [rank_config] has checked the
    /// world divides them.
    pub(crate) fn even(self, total: usize) -> (usize, usize) {
        let count = total / self.world;
        (self.rank * count, count)
    }

    /// The key-value heads this rank attends with: its even share when the world divides them,
    /// else the one head its query heads read, which `world / heads` ranks then hold alike.
    pub(crate) fn kv_heads(self, heads: usize) -> (usize, usize) {
        if heads.is_multiple_of(self.world) {
            self.even(heads)
        } else {
            (self.rank * heads / self.world, 1)
        }
    }

    /// An expert's intermediate width as this rank holds it: its share, padded to whole
    /// [EXPERT_TILE]s past one rank.
    pub(crate) fn expert_width(self, intermediate: usize) -> usize {
        if self.world == 1 {
            intermediate
        } else {
            (intermediate / self.world).next_multiple_of(EXPERT_TILE)
        }
    }
}

/// The configuration a rank's kernels run with: `config` with every sharded count and width
/// divided by the world, the expert width padded ([Shard::expert_width]) and no MTP layer, which
/// no rank holds a share of; a sharded rank drafts with the DFlash 2 head or not at all.
///
/// # Errors
/// A width the world does not divide, or a key-value head count that neither divides it nor is
/// divided by it (a rank's query heads would read two key-value heads).
pub(crate) fn rank_config(config: &Qwen4ExpConfig, shard: Shard) -> Result<Qwen4ExpConfig, String> {
    let world = shard.world;
    if world == 0 || shard.rank >= world {
        return Err(format!("rank {} is not a rank of {world}", shard.rank));
    }
    if world == 1 {
        return Ok(config.clone());
    }
    for (what, count) in [
        ("query heads", config.num_attention_heads),
        ("GDN key heads", config.linear_num_key_heads),
        ("GDN value heads", config.linear_num_value_heads),
        (
            "shared-expert columns",
            config.shared_expert_intermediate_size,
        ),
    ] {
        if !count.is_multiple_of(world) {
            return Err(format!("{count} {what} do not split over {world} ranks"));
        }
    }
    let kv = config.num_key_value_heads;
    if !(kv.is_multiple_of(world) || world.is_multiple_of(kv)) {
        return Err(format!(
            "{kv} key-value heads neither split over {world} ranks nor are shared by them"
        ));
    }
    // A share of an NVFP4 row is whole 16-column scale groups and whole 16-byte loads.
    let expert = config.moe_intermediate_size;
    if !expert.is_multiple_of(world) || !(expert / world).is_multiple_of(32) {
        return Err(format!(
            "an expert {expert} wide does not split over {world} ranks into 32-column shares"
        ));
    }
    let width = shard.expert_width(expert);
    if width > MAX_INTERMEDIATE {
        return Err(format!(
            "a rank's expert share pads to {width}, over the kernels' {MAX_INTERMEDIATE}"
        ));
    }
    Ok(Qwen4ExpConfig {
        num_attention_heads: config.num_attention_heads / world,
        num_key_value_heads: shard.kv_heads(kv).1,
        linear_num_key_heads: config.linear_num_key_heads / world,
        linear_num_value_heads: config.linear_num_value_heads / world,
        moe_intermediate_size: width,
        shared_expert_intermediate_size: config.shared_expert_intermediate_size / world,
        mtp_num_hidden_layers: 0,
        ..config.clone()
    })
}

/// Whether `count` floats split over `world` ranks into whole float4 chunks, as the two-shot
/// all-reduce ([reduce_two_shot]) and q4e_sum_slots need.
#[must_use]
pub fn splits_two_shot(count: usize, world: usize) -> bool {
    world > 1 && count.is_multiple_of(4 * world)
}

/// The two-shot device all-reduce of `count` floats at `partial`, in place.
///
/// Every rank's chunk `rank` of each peer's partial lands in `slots` ([TpRuntime::scatter_on_device]),
/// `sum` adds them into this rank's own chunk (q4e_sum_slots: ascending rank order from 0.0, the
/// one-shot fold), and the summed chunks are gathered back in place, half the one-shot's bytes.
/// `sum` is handed this rank's chunk and its length in floats; [splits_two_shot] says when
/// `count` splits.
///
/// # Errors
/// A collective failure, or whatever `sum` reports.
pub fn reduce_two_shot(
    tp: &TpRuntime,
    stream: u64,
    partial: Deviceptr,
    slots: Deviceptr,
    count: usize,
    sum: impl FnOnce(Deviceptr, usize) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let chunk = count / tp.world();
    let mine = reduce_scatter(tp, stream, partial, slots, count, sum)?;
    gather_in_place(tp, stream, partial, mine, chunk * 4)
}

/// The two-shot all-reduce's first half: this rank's chunk of `count` floats at `partial`
/// (chunk `rank` of `world` equal ones) summed over the world in place, as [reduce_two_shot]
/// sums it. Returns the chunk; the rest of `partial` is left as this rank's own partial.
///
/// # Errors
/// A collective failure, or whatever `sum` reports.
pub fn reduce_scatter(
    tp: &TpRuntime,
    stream: u64,
    partial: Deviceptr,
    slots: Deviceptr,
    count: usize,
    sum: impl FnOnce(Deviceptr, usize) -> Result<(), BackendError>,
) -> Result<Deviceptr, BackendError> {
    let chunk = count / tp.world();
    let mine = partial.offset(tp.rank() * chunk * 4);
    tp.scatter_on_device(stream, partial.0, slots.0, count * 4)
        .map_err(collective_error)?;
    sum(mine, chunk)?;
    Ok(mine)
}

/// Every rank's `bytes` at `mine`, which is its own slot of `base` (`base + rank * bytes`),
/// exchanged into the peers' slots of `base` in rank order; the own slot is left as it is.
fn gather_in_place(
    tp: &TpRuntime,
    stream: u64,
    base: Deviceptr,
    mine: Deviceptr,
    bytes: usize,
) -> Result<(), BackendError> {
    debug_assert_eq!(mine, base.offset(tp.rank() * bytes));
    tp.exchange_on_device(stream, mine.0, base.0, bytes)
        .map_err(collective_error)
}

/// A switch read from the environment: 0 or 1, on when unset; any other value refuses.
///
/// `SPARKFLASH_SEQUENCE_PARALLEL` says whether a prefill chunk's hyper-connections run
/// sequence-parallel, `SPARKFLASH_SEQUENCE_OVERLAP` whether such a chunk overlaps its collectives
/// with the other half's compute ([DeviceSequence::overlapped_layers]).
fn switch(name: &str) -> Result<bool, BackendError> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(true),
        Ok(value) if value == "1" => Ok(true),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) => Err(BackendError::new(
            "qwen4_tp",
            format!("{name}={value:?} is neither 0 nor 1"),
        )),
        Err(error) => Err(BackendError::new("qwen4_tp", format!("{name}: {error}"))),
    }
}

fn collective_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::new("qwen4_tp", error.to_string())
}

/// The collective a sharded arm sums its branch partials with.
pub(super) struct TensorParallel {
    pub(super) runtime: TpRuntime,
    /// Whether the partials are exchanged on the devices (NCCL) rather than over the host seam.
    device_leg: bool,
    /// The fewest rows a step sums with [reduce_two_shot] ([sparkflash_weights::tp::two_shot_rows]).
    two_shot_rows: usize,
    /// Every peer's partial rows of one forward, `[world][rows][hidden]` floats, for the device
    /// leg to gather into.
    pub(super) slots: Deviceptr,
    /// The most rows one forward sums: the slots' `rows`.
    rows: usize,
    /// Whether a wide prefill chunk splits its hyper-connections over the ranks
    /// ([DeviceSequence::sequence_parallel]).
    sequence_parallel: bool,
    /// Whether such a chunk runs as two halves whose collectives overlap the other half's
    /// compute ([DeviceSequence::overlapped_layers]).
    overlap: bool,
}

impl TensorParallel {
    /// The collective over `runtime`, with gather room for `rows` rows of `hidden`.
    ///
    /// # Errors
    /// A device leg that does not sum on the f32 wire, or the slots allocation.
    pub(super) fn new(
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        runtime: TpRuntime,
        rows: usize,
        hidden: usize,
    ) -> Result<Self, BackendError> {
        // The device leg moves raw bits and the inject sums them in Wire::F32's order; a leg that
        // rounds another way is refused rather than summed as if it were.
        let device_leg = match runtime.device_wire() {
            None => false,
            Some(Wire::F32) => true,
            Some(wire) => {
                return Err(BackendError::new(
                    "qwen4_tp",
                    format!(
                        "the collective's device leg sums on the {wire:?} wire; this arm sums only \
                         Wire::F32"
                    ),
                ));
            }
        };
        let two_shot_rows = sparkflash_weights::tp::two_shot_rows().map_err(collective_error)?;
        let slots = cuda
            .alloc(runtime.world() * rows * hidden * 4, &mut 0)
            .map_err(|error| BackendError::new("qwen4_tp", error))?;
        Ok(Self {
            runtime,
            device_leg,
            two_shot_rows,
            slots,
            rows,
            sequence_parallel: switch("SPARKFLASH_SEQUENCE_PARALLEL")?,
            overlap: switch("SPARKFLASH_SEQUENCE_OVERLAP")?,
        })
    }
}

/// The second stream of a sequence-parallel chunk that runs as two halves
/// ([DeviceSequence::overlapped_layers]), made with a session's first such chunk.
pub(super) struct Overlap {
    /// Where each half's reduce-scatter, inject, fold and gather run, at the greatest stream
    /// priority: it is the chunk's critical path, and a collective that waits for SMs behind the
    /// other half's grids holds up its peers too.
    stream: RunStream,
    /// The split-GEMM sums of the folds on `stream`, apart from the session stream's
    /// `gemm_partials`, which the other half's projections are using at the same time.
    partials: Deviceptr,
    /// Recorded on the session's stream once a half's branch partials are written.
    computed: [StreamEvent; 2],
    /// Recorded on `stream` once a half's next block input is gathered.
    gathered: [StreamEvent; 2],
}

// SAFETY: a stream, events and a buffer of the backend's context, owned by one session and used
// by one thread at a time, as the session's own stream is.
unsafe impl Send for Overlap {}

impl Overlap {
    fn new(cuda: &Cuda, zero: Kernel) -> Result<Self, BackendError> {
        let error = |error: String| BackendError::new("qwen4_tp", error);
        let event = || cuda.stream_event().map_err(error);
        let (computed, gathered) = ([event()?, event()?], [event()?, event()?]);
        let stream = cuda.run_stream_urgent().map_err(error)?;
        let partials = cuda.alloc(GEMM_PARTIAL_REGION * 4, &mut 0).map_err(error)?;
        zero_split_tickets(cuda, &stream, zero, partials, 1).map_err(error)?;
        Ok(Self {
            stream,
            partials,
            computed,
            gathered,
        })
    }

    /// Drain the stream and give everything back to the context.
    pub(super) fn release(&self, cuda: &Cuda) {
        let _ = cuda.synchronize_stream(&self.stream);
        cuda.free_buffers(&[&self.partials]);
        for event in self.computed.iter().chain(&self.gathered) {
            cuda.destroy_event(event);
        }
        cuda.destroy_run_stream(&self.stream);
    }
}

impl DeviceSequence {
    /// Whether a sequence-parallel chunk of `batch` rows runs as two halves
    /// ([Self::overlapped_layers]): each half must itself split into whole [DENSE_ROWS] pieces.
    pub(super) fn overlaps(&self, batch: usize) -> bool {
        self.backend.tp.as_ref().is_some_and(|tp| {
            tp.overlap && batch.is_multiple_of(2 * tp.runtime.world() * DENSE_ROWS)
        })
    }

    /// The layers from `first` on of a sequence-parallel chunk of `batch` rows ([Self::overlaps]),
    /// run as two halves so that each half's collectives overlap the other half's compute.
    ///
    /// A half is computed as a chunk of its own would be: its attention at its own positions, the
    /// second after the first, and its rows in the same whole [DENSE_ROWS] pieces, so every row
    /// keeps its bits (a 2,048-row chunk gives the 4,096-row chunk's greedy texts). Once the
    /// session's stream has written a half's branch partials, the overlap stream reduce-scatters
    /// them, injects this rank's rows, folds them for the next block and gathers the folded rows,
    /// while the session's stream computes the other half's branch. Every collective runs on the
    /// one overlap stream, in the same order on every rank.
    pub(super) fn overlapped_layers(
        &mut self,
        layers: &[GpuLayer],
        first: usize,
        chunk: bool,
        batch: usize,
    ) -> Result<(), BackendError> {
        let backend = Arc::clone(&self.backend);
        let tp = backend.tp.as_ref().ok_or_else(|| {
            BackendError::new(
                "qwen4_tp",
                "an overlapped chunk on an arm that is not sharded",
            )
        })?;
        if self.overlap.is_none() {
            self.overlap = Some(Overlap::new(&backend.cuda, backend.kernels.zero)?);
        }
        let config = &backend.config;
        let (rank, world) = (tp.runtime.rank(), tp.runtime.world());
        let half = batch / 2;
        let own = half / world;
        let share = |session: &mut Self, h: usize| {
            session.window = h * half;
            session.own_rows = Some((h * half + rank * own, own));
        };
        let position = self.position;
        // The overlap stream starts after everything the session's stream has queued so far.
        self.hand_over(0)?;
        for h in 0..2 {
            self.beside(|session| {
                share(session, h);
                session.branch_input(&layers[0].attn_hyper, half)?;
                session.mark_gathered(h)
            })?;
        }
        for (offset, layer) in layers.iter().enumerate() {
            let index = first + offset;
            for h in 0..2 {
                self.await_gathered(h)?;
                let mixed = self.mixed.offset(h * half * config.hidden_size * 4);
                self.branch_at(h * half, |session| {
                    session.attend(
                        index,
                        layer,
                        mixed,
                        position + h * half,
                        half,
                        chunk,
                        0,
                        0,
                        0,
                    )
                })?;
                self.hand_over(h)?;
                self.beside(|session| {
                    share(session, h);
                    session.inject_branch(session.hyper, session.injection, half)?;
                    session.prof_mark("attn_inject")?;
                    session.branch_input(&layer.mlp_hyper, half)?;
                    session.mark_gathered(h)
                })?;
            }
            for h in 0..2 {
                self.await_gathered(h)?;
                let mixed = self.mixed.offset(h * half * config.hidden_size * 4);
                let injection = self.injection.offset(h * half * config.hc_count * 4);
                self.branch_at(h * half, |session| {
                    session.mlp_branch(
                        &layer.moe,
                        mixed,
                        session.hyper,
                        injection,
                        index,
                        chunk,
                        half,
                    )
                })?;
                self.hand_over(h)?;
                let next = layers.get(offset + 1);
                self.beside(|session| {
                    share(session, h);
                    session.inject_branch(session.hyper, session.injection, half)?;
                    session.capture_features(index, half)?;
                    session.prof_mark("inject_mlp")?;
                    if let Some(next) = next {
                        session.branch_input(&next.attn_hyper, half)?;
                    }
                    session.mark_gathered(h)
                })?;
            }
        }
        // The overlap stream is in order, so its last mark covers both halves.
        self.await_gathered(1)?;
        for h in 0..2 {
            share(self, h);
            self.gather_features()?;
        }
        Ok(())
    }

    /// Run `work` with its launches and collectives on the overlap stream and its split GEMMs
    /// summed in the overlap's own partials.
    fn beside<T>(
        &mut self,
        work: impl FnOnce(&mut Self) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        let overlap = self
            .overlap
            .as_ref()
            .ok_or_else(|| BackendError::new("qwen4_tp", "no overlap stream"))?;
        let stream = overlap.stream.alias();
        let partials = std::mem::replace(&mut self.gemm_partials, overlap.partials);
        let previous = self.batch_stream.replace(stream);
        let result = work(self);
        self.batch_stream = previous;
        self.gemm_partials = partials;
        result
    }

    /// Run `work` with `self.branch` at row `row`, where a half's partials go.
    fn branch_at<T>(
        &mut self,
        row: usize,
        work: impl FnOnce(&Self) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        let branch = self.branch;
        self.branch = branch.offset(row * self.backend.config.hidden_size * 4);
        let result = work(self);
        self.branch = branch;
        result
    }

    fn overlap_events(&self) -> Result<&Overlap, BackendError> {
        self.overlap
            .as_ref()
            .ok_or_else(|| BackendError::new("qwen4_tp", "no overlap stream"))
    }

    /// The session's stream marks half `h`'s branch as computed and the overlap stream waits for
    /// it before anything queued on it from here on.
    fn hand_over(&self, h: usize) -> Result<(), BackendError> {
        let overlap = self.overlap_events()?;
        let cuda = &self.backend.cuda;
        cuda.record_event_on(&self.stream, &overlap.computed[h])
            .and_then(|()| cuda.stream_wait_event(&overlap.stream, &overlap.computed[h]))
            .map_err(|error| BackendError::new("qwen4_tp", error))
    }

    /// The overlap stream marks half `h`'s next input as gathered.
    fn mark_gathered(&self, h: usize) -> Result<(), BackendError> {
        let overlap = self.overlap_events()?;
        self.backend
            .cuda
            .record_event_on(&overlap.stream, &overlap.gathered[h])
            .map_err(|error| BackendError::new("qwen4_tp", error))
    }

    /// The session's stream waits for half `h`'s last [Self::mark_gathered].
    fn await_gathered(&self, h: usize) -> Result<(), BackendError> {
        let overlap = self.overlap_events()?;
        self.backend
            .cuda
            .stream_wait_event(&self.stream, &overlap.gathered[h])
            .map_err(|error| BackendError::new("qwen4_tp", error))
    }

    /// This rank's `(first, count)` rows of a `batch`-row forward when its hyper-connections run
    /// sequence-parallel, else None.
    ///
    /// The hyper-connections are replicated work on a sharded arm: every rank widens and folds
    /// every row. In a wide prefill chunk each rank instead runs them over its own rows only and
    /// the ranks gather the folded rows each branch reads ([Self::branch_input]); the branch's
    /// partials are then reduce-scattered rather than all-reduced, so each rank sums and injects
    /// only its own rows ([Self::inject_branch]). The bytes moved are the two-shot all-reduce's.
    ///
    /// A share must be whole [DENSE_ROWS] pieces, so every row is computed by the same tiles as
    /// in the replicated forward and keeps its bits; the chunk must be one with no snapshot (a
    /// verify), and the session must not need the widened rows afterwards (the MTP reads them).
    pub(super) fn sequence_parallel(&self, batch: usize, chunk: bool) -> Option<(usize, usize)> {
        let tp = self.backend.tp.as_ref()?;
        let world = tp.runtime.world();
        let mtp = self.spec_enabled && !self.drafting();
        if !tp.sequence_parallel
            || !tp.device_leg
            || !chunk
            || mtp
            || world < 2
            || batch > tp.rows
            || !batch.is_multiple_of(world * DENSE_ROWS)
        {
            return None;
        }
        let count = batch / world;
        Some((tp.runtime.rank() * count, count))
    }

    /// The folded rows a branch reads and the injection weights of `block`, over every row of a
    /// `batch`-row forward. Under sequence parallelism ([Self::own_rows]) this rank folds its own
    /// rows and gathers its peers' of the `batch` rows from [Self::window] on; the injection
    /// weights stay its own rows' alone, which is all its inject reads.
    pub(super) fn branch_input(
        &self,
        block: &GpuHyper,
        batch: usize,
    ) -> Result<(Deviceptr, Deviceptr), BackendError> {
        let Some((first, count)) = self.own_rows else {
            let (mixed, injection) = self.hyper(block, self.hyper, true, batch)?;
            let injection = injection.expect("an injecting hyper-connection has weights");
            return Ok((mixed, injection));
        };
        let (mixed, _) = self.hyper_rows(block, self.hyper, true, first, count)?;
        let row = self.backend.config.hidden_size * 4;
        self.gather_rows(self.mixed.offset(self.window * row), mixed, count * row)?;
        self.prof_mark("sp_gather")?;
        Ok((self.mixed, self.injection))
    }

    /// Every rank's `bytes` at `mine`, its own slot of `base`, gathered into the rest of `base`.
    pub(super) fn gather_rows(
        &self,
        base: Deviceptr,
        mine: Deviceptr,
        bytes: usize,
    ) -> Result<(), BackendError> {
        let tp = self.backend.tp.as_ref().ok_or_else(|| {
            BackendError::new("qwen4_tp", "a row gather on an arm that is not sharded")
        })?;
        gather_in_place(&tp.runtime, self.clock().raw(), base, mine, bytes)
    }

    /// The widened rows the language head folds: the last `rows` rows of the forward, which
    /// under sequence parallelism the last rank alone holds and shares with the others.
    pub(super) fn head_rows(&self, batch: usize, rows: usize) -> Result<Deviceptr, BackendError> {
        let width = self.backend.config.hyper_width();
        let last = self.hyper.offset((batch - rows) * width * 4);
        let (Some(tp), Some(_)) = (self.backend.tp.as_ref(), self.own_rows) else {
            return Ok(last);
        };
        let world = tp.runtime.world();
        let bytes = rows * width * 4;
        if bytes > tp.rows * self.backend.config.hidden_size * 4 {
            return Err(BackendError::new(
                "qwen4_tp",
                format!("{rows} head rows do not fit the collective's slots"),
            ));
        }
        // Each rank sends the rows at the end of the chunk, which only the last rank's share
        // holds; the exchange leaves a rank's own slot unwritten, so the last rank reads its own.
        tp.runtime
            .exchange_on_device(self.clock().raw(), last.0, tp.slots.0, bytes)
            .map_err(collective_error)?;
        if tp.runtime.rank() + 1 == world {
            Ok(last)
        } else {
            Ok(tp.slots.offset((world - 1) * bytes))
        }
    }

    /// Add `rows` rows of `self.branch` into the widened residual `hyper` through `injection`.
    ///
    /// On a sharded arm the rows are this rank's partials and are summed over the world first:
    /// gathered into the slots for q4e_inject to fold in rank order (one shot), summed in place by
    /// [reduce_two_shot] from [TensorParallel::two_shot_rows] rows on, or summed on the host seam
    /// when the collective has no device leg. All three add the same bits. Under sequence
    /// parallelism ([Self::own_rows]) only the reduce-scatter runs and only this rank's rows are
    /// injected; the partials reduce-scattered are the `rows` from [Self::window] on.
    pub(super) fn inject_branch(
        &self,
        hyper: Deviceptr,
        injection: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        let backend = &self.backend;
        let config = &backend.config;
        let hidden = config.hidden_size;
        let count = rows * hidden;
        let mut target = (hyper, self.branch, injection, rows);
        let (slots, rank, world) = match &backend.tp {
            None => (0, 0, 1),
            Some(tp) if !tp.device_leg => {
                let mut values = vec![0.0f32; count];
                let cuda = &backend.cuda;
                cuda.copy_dtoh_on(
                    self.clock(),
                    bytemuck::cast_slice_mut(&mut values),
                    &self.branch,
                )
                .map_err(|error| BackendError::new("qwen4_tp", error))?;
                tp.runtime
                    .all_reduce(&mut values)
                    .map_err(collective_error)?;
                cuda.copy_htod_on(self.clock(), &self.branch, bytemuck::cast_slice(&values))
                    .map_err(|error| BackendError::new("qwen4_tp", error))?;
                (0, 0, 1)
            }
            Some(tp) if rows > tp.rows => {
                return Err(BackendError::new(
                    "qwen4_tp",
                    format!(
                        "a forward of {rows} rows is wider than the {} the collective's slots hold",
                        tp.rows
                    ),
                ));
            }
            Some(tp) => {
                let world = tp.runtime.world();
                let stream = self.clock().raw();
                if let Some((first, own)) = self.own_rows {
                    let mine = reduce_scatter(
                        &tp.runtime,
                        stream,
                        self.branch.offset(self.window * hidden * 4),
                        tp.slots,
                        count,
                        |mine, chunk| self.sum_slots(tp, mine, chunk),
                    )?;
                    debug_assert_eq!(mine, self.branch.offset(first * hidden * 4));
                    target = (
                        hyper.offset(first * config.hyper_width() * 4),
                        mine,
                        injection.offset(first * config.hc_count * 4),
                        own,
                    );
                    (0, 0, 1)
                } else if tp.two_shot_rows > 0
                    && rows >= tp.two_shot_rows
                    && splits_two_shot(count, world)
                {
                    reduce_two_shot(
                        &tp.runtime,
                        stream,
                        self.branch,
                        tp.slots,
                        count,
                        |mine, chunk| self.sum_slots(tp, mine, chunk),
                    )?;
                    (0, 0, 1)
                } else {
                    tp.runtime
                        .exchange_on_device(stream, self.branch.0, tp.slots.0, count * 4)
                        .map_err(collective_error)?;
                    (tp.slots.0, tp.runtime.rank() as u64, world as u64)
                }
            }
        };
        let (hyper, branch, injection, rows) = target;
        self.launch(
            backend.kernels.inject,
            (hidden.div_ceil(256).max(1) as u32, rows as u32, 1),
            (256, 1, 1),
            &[
                hyper.0,
                branch.0,
                injection.0,
                config.hc_count as u64,
                hidden as u64,
                slots,
                rank,
                world,
            ],
        )
    }

    /// q4e_sum_slots: this rank's `chunk` floats at `mine` summed with its peers' chunks in the
    /// slots, in ascending rank order from 0.0.
    fn sum_slots(
        &self,
        tp: &TensorParallel,
        mine: Deviceptr,
        chunk: usize,
    ) -> Result<(), BackendError> {
        self.launch(
            self.backend.kernels.sum_slots,
            ((chunk / 4).div_ceil(256).max(1) as u32, 1, 1),
            (256, 1, 1),
            &[
                mine.0,
                tp.slots.0,
                chunk as u64,
                tp.runtime.rank() as u64,
                tp.runtime.world() as u64,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rank_config_divides_the_sharded_widths() {
        let config = Qwen4ExpConfig::tiny();
        let whole = rank_config(&config, Shard::WHOLE).expect("one rank is the whole model");
        assert_eq!(whole, config);
        let world = 2;
        if let Ok(local) = rank_config(&config, Shard { rank: 1, world }) {
            assert_eq!(
                local.num_attention_heads * world,
                config.num_attention_heads
            );
            assert_eq!(local.mtp_num_hidden_layers, 0);
            assert!(local.moe_intermediate_size >= config.moe_intermediate_size / world);
        }
    }

    #[test]
    fn key_value_heads_are_split_or_shared() {
        let four = |rank| Shard { rank, world: 4 };
        assert_eq!(four(0).kv_heads(2), (0, 1));
        assert_eq!(four(1).kv_heads(2), (0, 1));
        assert_eq!(four(2).kv_heads(2), (1, 1));
        assert_eq!(four(3).kv_heads(2), (1, 1));
        assert_eq!(Shard { rank: 1, world: 2 }.kv_heads(2), (1, 1));
        assert_eq!(four(3).even(48), (36, 12));
    }

    #[test]
    fn the_expert_share_pads_to_whole_tiles() {
        assert_eq!(Shard::WHOLE.expert_width(640), 640);
        assert_eq!(Shard { rank: 0, world: 2 }.expert_width(640), 320);
        assert_eq!(Shard { rank: 3, world: 4 }.expert_width(640), 192);
    }
}
