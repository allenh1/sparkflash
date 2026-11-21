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

//! The single-process Dory arm as the runtime's planned walk ([PlanWalk] over [DoryHost]).
//!
//! The runtime on the shared tick engine drives it: the prompt parts, the ids, the sampler,
//! prefix reuse, the rewind point, the lanes, the ledger, the layer loop and the head's
//! record-then-choose are the runtime's, and this module supplies what only the arm knows:
//!
//! * The layers are the stage's schedule ([super::schedule]): each block in each loop it runs,
//!   the input prefix's blocks once, every recurrent block once a loop, then the output
//!   suffix's. A layer's two branches are the block's attention and its MLP, each its own
//!   nGPT lerp, so a branch's exit lands nothing: the work between blocks (the prefix's end, a
//!   loop's start and end, the suffix's start) is the frames' ([Frame]), each edge on the
//!   entry of the branch after it and the trailing ones on the last branch's exit.
//! * A sequence's state is one positional region a block ([PlanHost::regions]): a block that
//!   runs every loop keeps one cache a loop, a sliding block a ring of its window and one
//!   prompt chunk, so a rewind truncates and the arm has nothing to checkpoint.
//! * A prompt chunk is one pass of up to [PREFILL_BATCH] rows, and a step is one row a lane for
//!   at most [LANE_BATCH] lanes; each block runs the bodies the arm's own loops ran
//!   ([StageCore::attention_batch] for a chunk, [StageCore::attention_lanes] for a row a lane,
//!   and [super::mlp] for both), so a lane's bits are the bits those loops gave it.
//! * The head is the arm's: every row's logits, chosen on the host ([PlanHost::KEEPS_LOGITS]),
//!   and its end-of-sequence ids end the reply ([PlanHost::ends]).
//!
//! Two properties of the arm's prompt kernels shape a window ([PlanHost::begin_window]), because
//! a reused prompt forwards only the ids it added and those rows must have the bits the whole
//! prompt's chunks gave them. A prompt projection of eight rows or fewer takes the token GEMV and
//! a wider chunk the tensor-core tile, whose per-row bits differ, so a per-loop pass of
//! [TOKEN_ROWS] rows or fewer runs [PREFILL_Q] rows, the extra ones repeats of its own tokens. And
//! a prompt attention block's key window is its first row's, so a row's bits depend on which
//! query tile it sits in, which is why a pass that does not start on a [PREFILL_Q] boundary
//! re-runs the rows of its first tile from the ids the sequence holds ([PlanSeq::history]) and
//! its head reads its own rows past them. A cache row the extra rows wrote sits past the pass's
//! last row, so the rows after the pass overwrite it before any attention reads it.

use std::cell::RefCell;
use std::sync::Arc;

use sparkflash_model::backend::BackendError;
use sparkflash_model::ledger::State;
use sparkflash_model::{TpPlan, argmax};
use sparkflash_runtime::{
    Absent, ComputeDeclaration, Limits, NoVision, PartExec, PartHost, PassKind, PassMask,
    PlanBranch, PlanHost, PlanLane, PlanPart, PlanPass, PlanRows, PlanSeq, PlanWalk, PlanWindow,
    RankSplit, Region, Segments, StateKind, WalkCapabilities,
};

use crate::config::KvCacheMode;

use super::{
    DEVICE, DeviceShared, Edge, LANE_BATCH, OwnedScratch, PREFILL_BATCH, PREFILL_Q, Scratch,
    StageCore, Step, TOKEN_ROWS, edge, embed_rows, floats, head_rows, logits_rows, mlp, schedule,
};

/// The single-process Dory arm, exposed to the shared runtime as a Walk.
pub(crate) type DoryWalk = PlanWalk<DoryHost>;

/// The walk over `shared`'s weights, with the row buffers every lane it opens steps through.
///
/// # Errors
/// A stage other than the whole model, or a scratch allocation the device refuses.
pub(crate) fn walk(shared: Arc<DeviceShared>) -> Result<DoryWalk, BackendError> {
    PlanWalk::new(Arc::new(DoryHost::new(shared)?), LANE_BATCH)
}

impl PlanLane for StageCore {
    const DEVICE: &'static str = DEVICE;

    /// Every position is addressed by its own row of the KV caches, so a rewind is a
    /// truncation and the arm has no recurrent region to checkpoint.
    fn save(&mut self, _at: usize) -> Result<(), BackendError> {
        Ok(())
    }

    fn restore(&mut self, to: usize) -> Result<(), BackendError> {
        self.rewind(to)
    }

    fn clear(&mut self) -> Result<(), BackendError> {
        self.rewind(0)
    }

    fn room(&self, positions: usize, _reset: bool) -> bool {
        positions <= self.shared().context
    }
}

/// A branch's frame: the edges of the stage's walk the branch's entry runs before its part, and
/// the ones its exit runs after it (the walk's last branch only).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Frame {
    before: Vec<Edge>,
    after: Vec<Edge>,
}

/// The parts' host: a block's attention and MLP run the arm's own bodies.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DoryParts;

impl PartHost for DoryParts {
    type Pass<'a> = DoryPass<'a>;
    type Open<'a> = ();
    type Handle = PlanPart<DoryParts>;
}

/// One part's pass: the host, the rows the window's layers run and its sequences.
pub(crate) struct DoryPass<'a> {
    host: &'a DoryHost,
    rows: usize,
    segments: &'a PlanRows<'a, 'a, StageCore>,
}

impl DoryPass<'_> {
    /// The sequences' lanes, one a row, without a heap allocation in the serving loop.
    fn lanes<R>(
        &self,
        run: impl FnOnce(&[&StageCore], &[bool]) -> Result<R, BackendError>,
    ) -> Result<R, BackendError> {
        let count = self.segments.len();
        let Some((first, _)) = self.segments.first() else {
            return Err(BackendError::new(DEVICE, "a batched pass of no lanes"));
        };
        if count > LANE_BATCH {
            return Err(lane_refusal(count));
        }
        let mut lanes = [&first.lane; LANE_BATCH];
        for (lane, (seq, _)) in lanes.iter_mut().zip(self.segments.iter()) {
            *lane = &seq.lane;
        }
        run(&lanes[..count], &[true; LANE_BATCH][..count])
    }
}

/// A block's attention in one loop: from `hidden` into `state`.
struct Attention {
    block: usize,
    loop_index: usize,
}

impl PartExec<DoryParts> for Attention {
    /// The host's regions state the lane ([PlanHost::regions]).
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Replicated
    }

    /// A chunk of one row takes the token kernels, as the arm's own prompt loop's did.
    fn rows(&self, pass: &mut DoryPass<'_>) -> Result<(), BackendError> {
        if pass.rows == 1 {
            return self.token(pass);
        }
        let [(seq, _)] = pass.segments else {
            return Err(chunk_refusal(pass.segments.len()));
        };
        let block = &pass.host.shared.config.blocks[self.block];
        let lane = &seq.lane;
        lane.attention_batch(
            self.block,
            block,
            self.loop_index,
            lane.position(),
            pass.rows,
        )
    }

    fn token(&self, pass: &mut DoryPass<'_>) -> Result<(), BackendError> {
        let block = &pass.host.shared.config.blocks[self.block];
        pass.lanes(|lanes, active| {
            StageCore::attention_lanes(lanes, active, self.block, block, self.loop_index)
        })
    }
}

/// A block's MLP in one loop: from `state` into `hidden`.
struct Mlp {
    block: usize,
    loop_index: usize,
}

impl PartExec<DoryParts> for Mlp {
    /// The MLP keeps no state.
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Replicated
    }

    fn rows(&self, pass: &mut DoryPass<'_>) -> Result<(), BackendError> {
        let host = pass.host;
        mlp(
            &host.shared,
            &host.scratch,
            self.block,
            self.loop_index,
            pass.rows,
        )
    }

    // A decode row a lane and a prompt chunk run the one body.
    fn token(&self, pass: &mut DoryPass<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}

/// The window the walk is in, kept for its layers and its head.
#[derive(Debug, Default)]
struct Window {
    /// The ids the window's layers run: a prompt chunk's re-run tile rows, its own rows and its
    /// padding, or a step's one row a lane.
    batch: Vec<u32>,
    /// The batch row of the window's first own row.
    first: usize,
    /// The batch rows that are the sequence's, the re-run ones and the window's own.
    real: usize,
    /// The position a prompt chunk's batch starts at.
    from: usize,
}

/// One walk's state: the open window.
#[derive(Debug, Default)]
pub(crate) struct DoryScratch {
    window: RefCell<Window>,
}

/// The single-process stage and its schedule: the planned walk's host.
pub(crate) struct DoryHost {
    shared: Arc<DeviceShared>,
    /// The row buffers every lane of this walk steps through: one set, so a pass over several
    /// lanes is one batched launch, and the runtime steps the lanes one pass at a time.
    scratch: Arc<OwnedScratch>,
    /// Each block a loop runs, its attention and its MLP, in walk order.
    layers: Vec<[PlanBranch<DoryParts, Frame>; 2]>,
    limits: Limits,
    /// Whether a prompt runs loop by loop, a whole chunk at once, or one token at a time.
    per_loop: bool,
}

impl DoryHost {
    /// The host over `shared`'s weights, with the row buffers every lane it opens steps through.
    ///
    /// # Errors
    /// A stage other than the whole model, or a scratch allocation the device refuses.
    fn new(shared: Arc<DeviceShared>) -> Result<Self, BackendError> {
        let stage = &shared.stage;
        if !stage.prefix || !stage.suffix || stage.draft {
            return Err(BackendError::new(
                DEVICE,
                "the planned Dory walk runs the whole model in one process; a recurrence \
                 stage runs in its group",
            ));
        }
        let mut layers: Vec<[PlanBranch<DoryParts, Frame>; 2]> = Vec::new();
        let mut edges = Vec::new();
        for step in schedule(&shared.config, true, stage.loops.clone(), true) {
            let (block, loop_index) = match step {
                Step::Edge(at) => {
                    edges.push(at);
                    continue;
                }
                Step::Block { index, loop_index } => (index, loop_index),
            };
            let slot = 2 * layers.len();
            layers.push([
                PlanBranch {
                    frame: Frame {
                        before: std::mem::take(&mut edges),
                        after: Vec::new(),
                    },
                    part: Box::new(Attention { block, loop_index }) as PlanPart<DoryParts>,
                    summed: false,
                    slot,
                },
                PlanBranch {
                    frame: Frame::default(),
                    part: Box::new(Mlp { block, loop_index }) as PlanPart<DoryParts>,
                    summed: false,
                    slot: slot + 1,
                },
            ]);
        }
        let Some([_, last]) = layers.last_mut() else {
            return Err(BackendError::new(DEVICE, "the Dory stage runs no block"));
        };
        last.frame.after = edges;
        let scratch = Arc::new(OwnedScratch {
            buffers: Scratch::allocate(&shared)
                .map_err(|error| BackendError::new(DEVICE, error))?,
            shared: Arc::clone(&shared),
        });
        let per_loop = shared.config.recurrent_kv_cache_mode == KvCacheMode::PerLoop;
        Ok(Self {
            limits: walk_limits(per_loop, shared.context),
            shared,
            scratch,
            layers,
            per_loop,
        })
    }

    /// Runs `edges` over the window's rows.
    fn edges(&self, scratch: &DoryScratch, edges: &[Edge]) -> Result<(), BackendError> {
        let rows = scratch.window.borrow().batch.len();
        for at in edges {
            edge(&self.shared, &self.scratch, *at, rows)?;
        }
        Ok(())
    }
}

/// The limits the walk states at `context`.
///
/// A per-loop prompt takes a whole chunk a pass; last-loop's overwritten cache makes a row
/// depend on the tokens after it, so that mode keeps the token-at-a-time order.
fn walk_limits(per_loop: bool, context: usize) -> Limits {
    let rows = if per_loop { PREFILL_BATCH } else { 1 };
    Limits::new(rows, context).with_members(LANE_BATCH)
}

/// The arm's state, as the runtime needs to see it: one positional region a block's KV caches.
///
/// A block that runs every loop keeps one cache a loop, and a sliding block's cache is a ring
/// of its window and one prompt chunk; both are addressed by position, so a rewind truncates.
fn regions_of(shared: &DeviceShared) -> Vec<Region> {
    let config = &shared.config;
    let width = config.num_key_value_heads * config.head_dim;
    let mut regions = Vec::with_capacity(config.blocks.len());
    for block in &config.blocks {
        let slots = (0..block.cache_slots)
            .filter(|slot| shared.stage.keeps_slot(block, *slot))
            .count();
        if slots == 0 {
            continue;
        }
        regions.push(Region::Positional {
            name: "kv",
            // The keys and the values, BF16 a value, of every slot the block keeps.
            bytes_per_row: slots * width * 2 * 2,
        });
    }
    regions
}

/// The tokens a narrow per-loop prompt pass runs: its own rows, then repeats of them up to
/// [`PREFILL_Q`] rows, so the pass takes the tensor-core tile a wider chunk takes.
///
/// The extra rows are a lane's own tokens, so nothing is out of range, and their cache rows sit
/// past the pass's last row: the rows after this pass overwrite them before any attention reads
/// them.
fn padded(tokens: &[u32], rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|index| tokens[index % tokens.len()])
        .collect()
}

/// A prompt chunk of more than one sequence.
fn chunk_refusal(segments: usize) -> BackendError {
    BackendError::new(
        DEVICE,
        format!("a prompt chunk is one sequence's rows, not {segments} sequences'"),
    )
}

/// A step of more lanes than one batched pass carries.
fn lane_refusal(lanes: usize) -> BackendError {
    BackendError::new(
        DEVICE,
        format!("a batched pass takes one row for each of at most {LANE_BATCH} lanes, not {lanes}"),
    )
}

impl PlanHost for DoryHost {
    type Parts = DoryParts;
    type Frame = Frame;
    type Lane = StageCore;
    type Drafter = Absent<&'static str>;
    type Vision = NoVision;
    type Scratch = DoryScratch;
    type Mode = ();

    const DEVICE: &'static str = DEVICE;
    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        passes: PassMask::TOKEN_ROWS.union(PassMask::BULK),
        segments: Segments::Many,
        state: StateKind::Positional,
        // The arm's own four rows, word for word the declaration in
        // `sparkflash_model::ARM_LEDGERS`; a test below holds them equal. The token row is the
        // same one the profile check that refuses `decode.tensor_cores = true` reads.
        compute: ComputeDeclaration {
            prompt: State::Native(
                "prompt projections over eight rows on dory_mma_wide; attention on the MMA \
                 instances",
            ),
            token: State::Off("warp-dot row kernels; decode.tensor_cores=true is refused"),
            stored: State::Native("BF16 weights and a BF16 KV cache throughout"),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };
    // The penalties, temperature, top-k and top-p run on the host over the row every step
    // copies back, so a non-greedy request is served rather than approximated, and a greedy
    // one answers its logits as the arm's sessions always did.
    const KEEPS_LOGITS: bool = true;

    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn branches(&self, layer: usize) -> &[PlanBranch<DoryParts, Frame>] {
        &self.layers[layer]
    }

    fn vocab(&self) -> usize {
        self.shared.config.vocab_size
    }

    // A prompt chunk's rows; a step's lanes are fewer.
    fn pass_rows(&self) -> usize {
        PREFILL_BATCH
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn regions(&self) -> Vec<Region> {
        regions_of(&self.shared)
    }

    fn scratch(_this: &Arc<Self>, _rows: usize) -> Result<DoryScratch, BackendError> {
        Ok(DoryScratch::default())
    }

    fn lane(this: &Arc<Self>) -> Result<StageCore, BackendError> {
        StageCore::with_scratch(Arc::clone(&this.shared), Arc::clone(&this.scratch))
            .map_err(|error| BackendError::new(DEVICE, error))
    }

    fn drafter(_this: &Arc<Self>) -> Result<Absent<&'static str>, BackendError> {
        // The loop-1 exit lives on a recurrence group's drafting rank, where a rank holds the
        // output suffix beside its loops; a single process has no second exit to draft from,
        // so the arm publishes none rather than a chain it cannot verify.
        Ok(Absent("the single-process Dory arm has no loop-1 exit"))
    }

    fn mode(&self, _pass: PlanPass, _rows: usize) {}

    fn begin_window(
        &self,
        scratch: &DoryScratch,
        segments: &mut PlanRows<'_, '_, StageCore>,
        window: &PlanWindow,
    ) -> Result<(), BackendError> {
        let kind = window.pass.kind;
        let mut open = scratch.window.borrow_mut();
        match kind {
            PassKind::Bulk { .. } => {
                let [(seq, tokens)] = segments else {
                    return Err(chunk_refusal(segments.len()));
                };
                let position = seq.lane.position();
                // A prompt row's bits are the tile the prompt's own chunks gave it, so a pass
                // that does not start on a query tile re-runs the few rows of the tile before
                // it, and a pass too narrow for the wide tile is padded to one.
                let start = if self.per_loop {
                    position % PREFILL_Q
                } else {
                    0
                };
                let kept = seq.history().len();
                if kept < start || position != kept {
                    return Err(BackendError::new(
                        DEVICE,
                        format!("the pass starts at {position} and the sequence holds {kept} ids"),
                    ));
                }
                let mut run: Vec<u32> = Vec::with_capacity(start + tokens.len() + PREFILL_Q);
                run.extend_from_slice(&seq.history()[kept - start..]);
                run.extend_from_slice(tokens);
                let real = run.len();
                let from = position - start;
                // A narrow pass takes the wide tile, as the arm's own prompt loop's chunks did;
                // the room left in the context caps the padding, and with less than a wide batch
                // left the pass runs its rows alone.
                let pad = if self.per_loop && real <= TOKEN_ROWS {
                    let room = self.shared.context.saturating_sub(from);
                    Some(PREFILL_Q.min(room)).filter(|pad| *pad > real)
                } else {
                    None
                };
                let batch = match pad {
                    Some(pad) => padded(&run, pad),
                    None => run,
                };
                seq.lane.rewind(from)?;
                seq.lane.check_rows(batch.len())?;
                *open = Window {
                    batch,
                    first: start,
                    real,
                    from,
                };
            }
            PassKind::TokenRows => {
                if segments.len() > LANE_BATCH {
                    return Err(lane_refusal(segments.len()));
                }
                open.batch.clear();
                for (seq, rows) in segments.iter() {
                    let [token] = rows else {
                        return Err(BackendError::new(
                            DEVICE,
                            format!(
                                "a token pass is one row a sequence, and this segment holds {}",
                                rows.len()
                            ),
                        ));
                    };
                    seq.lane.check_rows(1)?;
                    open.batch.push(*token);
                }
                (open.first, open.real, open.from) = (0, segments.len(), 0);
            }
            PassKind::Paired { .. } | PassKind::Verify { .. } => {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "the Dory device arm runs a prompt chunk and a token row a sequence, \
                         not a {kind:?} pass"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The window's layers ran outside the lanes: each moves past the rows it holds.
    fn finish(
        &self,
        scratch: &DoryScratch,
        segments: &mut PlanRows<'_, '_, StageCore>,
        pass: PlanPass,
        _last: bool,
    ) -> Result<(), BackendError> {
        let open = scratch.window.borrow();
        if let PassKind::Bulk { .. } = pass.kind {
            let [(seq, _)] = segments else {
                return Err(chunk_refusal(segments.len()));
            };
            // The padding's cache rows sit past the sequence's last row: the next pass writes
            // over them.
            seq.lane.advance(open.batch.len())?;
            return seq.lane.rewind(open.from + open.real);
        }
        for (seq, _) in segments.iter_mut() {
            seq.lane.advance(1)?;
        }
        Ok(())
    }

    fn ends(&self, token: u32) -> bool {
        self.shared.config.next_token(token).is_none()
    }

    /// Dory's template opens a thinking reply with ` thinking\n` and writes a past reply back as
    /// ` thinking</think>...`, so the next turn keeps the prompt only up to its last id.
    fn rewind_point(&self, prompt: usize) -> usize {
        prompt.saturating_sub(1)
    }

    /// A sliding block's cache holds its window and one prompt chunk, so the rows a rewind to
    /// the prompt's end reads survive a reply of at most [`PREFILL_BATCH`] positions.
    fn rewind_slack(&self) -> usize {
        PREFILL_BATCH
    }

    /// A lane's caches are its own, sized for the context when it opens.
    fn back(&self, _lane: &mut StageCore, _reach: usize) -> Result<(), BackendError> {
        Ok(())
    }

    /// The window's batch, which holds `ids` past the rows it re-runs.
    fn embed(&self, scratch: &DoryScratch, ids: &[u32]) -> Result<(), BackendError> {
        let open = scratch.window.borrow();
        debug_assert_eq!(
            open.batch.get(open.first..open.first + ids.len()),
            Some(ids)
        );
        embed_rows(&self.shared, &self.scratch, &open.batch)
    }

    fn enter(
        &self,
        scratch: &DoryScratch,
        frame: &Frame,
        _rows: usize,
    ) -> Result<(), BackendError> {
        self.edges(scratch, &frame.before)
    }

    fn pass<'a>(
        &'a self,
        scratch: &'a DoryScratch,
        segments: &'a [(&'a mut PlanSeq<StageCore>, &'a [u32])],
        _layer: usize,
        _slot: usize,
        _rows: usize,
        _mode: (),
    ) -> DoryPass<'a> {
        DoryPass {
            host: self,
            rows: scratch.window.borrow().batch.len(),
            segments,
        }
    }

    // Each branch is its own nGPT lerp, so the exit lands nothing; the last one closes the walk.
    fn leave(
        &self,
        scratch: &DoryScratch,
        frame: &Frame,
        _next: Option<&Frame>,
        _summed: bool,
        _rows: usize,
    ) -> Result<(), BackendError> {
        self.edges(scratch, &frame.after)
    }

    fn residual(&self, scratch: &DoryScratch, rows: usize) -> Result<Vec<f32>, BackendError> {
        let hidden = self.shared.config.hidden_size;
        let first = scratch.window.borrow().first;
        let mut bytes = vec![0u8; rows * hidden * 4];
        self.shared
            .cuda
            .copy_dtoh(&mut bytes, &self.scratch.hidden.offset(first * hidden * 4))
            .map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(floats(&bytes))
    }

    /// Row by row, as the arm's own head read them: the head over one residual row, then its
    /// logits copied back.
    fn logits(
        &self,
        scratch: &DoryScratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let open = scratch.window.borrow();
        let rows = open.real - open.first;
        if first + count > rows {
            return Err(BackendError::new(
                DEVICE,
                format!("the head is asked for row {} of {rows}", first + count - 1),
            ));
        }
        let mut logits = Vec::with_capacity(count * self.vocab());
        for row in first..first + count {
            let index = open.first + row;
            head_rows(&self.shared, &self.scratch, open.real, index..index + 1)?;
            logits.extend(logits_rows(&self.shared, &self.scratch, 1)?.swap_remove(0));
        }
        Ok(logits)
    }

    fn choices(
        &self,
        scratch: &DoryScratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<u32>, BackendError> {
        (first..first + count)
            .map(|row| self.logits(scratch, row, 1).map(|logits| argmax(&logits)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_model::backend::StepOutput;
    use sparkflash_runtime::testing::ArmProfile;
    use sparkflash_runtime::{ArmFixture, Fixture, HeadRow, Hidden, StateLayout, Walk};

    use super::*;
    use crate::DoryDeviceBackend;

    /// The arm as the record and the generated checks name it: a marker with the walk's own
    /// name, since the walk is the runtime's type. Its fixture is the crate's tiny ring
    /// checkpoint, written into a temporary directory the walk keeps alive. A check that boots
    /// it needs a CUDA device and the arm's cubins.
    pub(crate) struct DoryWalk;

    impl ArmFixture for DoryWalk {
        type Walk = super::DoryWalk;

        fn fixture(world: usize) -> Result<Fixture<super::DoryWalk>, BackendError> {
            if world > 1 {
                return Err(BackendError::new(
                    "group",
                    "the Dory device arm is one process; TP 2 and TP 4 need a rank ring",
                ));
            }
            if !DoryDeviceBackend::available() {
                return Err(BackendError::new(
                    "fixture",
                    "the Dory conformance fixture needs a build with the arm's cubins",
                ));
            }
            let directory = tempfile::tempdir()
                .map_err(|error| BackendError::new("fixture", error.to_string()))?;
            crate::fixture::write(
                directory.path(),
                &crate::config::DoryConfig::tiny_ring_json(),
            )
            .map_err(|error| BackendError::new("fixture", error))?;
            let backend = DoryDeviceBackend::open(directory.path(), 5_000)
                .map_err(|error| BackendError::new("fixture", error))?;
            let walk = backend.walk_arm()?;
            let bytes = backend.weight_bytes() as u64;
            Ok(Fixture::new(
                walk,
                "the Dory tiny ring checkpoint",
                bytes,
                bytes,
                vec![sparkflash_runtime::FormatRule::stored("BF16")],
                Some(Box::new(directory)),
            ))
        }

        /// The single-process arm serves on the tick engine, so its generated checks drive it.
        fn tick_engine() -> bool {
            true
        }
    }

    /// The arm's state is position-addressed: a rewind truncates and needs no checkpoint.
    #[test]
    fn the_layout_is_positional() {
        let layout = StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 8,
        }]);
        assert_eq!(layout.kind(), StateKind::Positional);
        assert!(!layout.needs_checkpoint());
    }

    /// A narrow prompt pass runs the wide tile with its own rows repeated, in order and only
    /// its own ids, so nothing the pass reads is outside the prompt.
    #[test]
    fn a_narrow_pass_pads_with_its_own_rows() {
        assert_eq!(padded(&[4, 5, 6], 6), [4, 5, 6, 4, 5, 6]);
        let full = padded(&[4, 5, 6], PREFILL_Q);
        assert_eq!(full.len(), PREFILL_Q);
        assert_eq!(full[..9], [4, 5, 6, 4, 5, 6, 4, 5, 6]);
        assert_eq!(padded(&[9], PREFILL_Q), vec![9; PREFILL_Q]);
    }

    /// The walk's layers are the whole model's schedule: every block in each loop it runs, and
    /// the prefix's end, each loop's start and end and the suffix's start between them.
    #[test]
    fn the_schedule_runs_every_block_a_loop_with_its_edges() {
        let config = crate::config::DoryConfig::tiny_ring();
        let steps: Vec<Step> = schedule(&config, true, 0..config.n_recurrent_loops, true).collect();
        let blocks = steps
            .iter()
            .filter(|step| matches!(step, Step::Block { .. }))
            .count();
        assert_eq!(
            blocks,
            config.input_blocks().len()
                + config.n_recurrent_loops * config.recurrent_blocks().len()
                + config.output_blocks().len()
        );
        let edges = |at: Edge| steps.iter().filter(|step| **step == Step::Edge(at)).count();
        assert_eq!(edges(Edge::Prefix), 1);
        assert_eq!(edges(Edge::LoopStart), config.n_recurrent_loops);
        assert_eq!(edges(Edge::LoopEnd), config.n_recurrent_loops);
        assert_eq!(edges(Edge::Suffix), 1);
    }

    /// A row a narrow pass rewrites has the bits its wide chunk gave it, which is what lets a
    /// reused prompt forward only the ids it added.
    #[test]
    #[ignore = "boots the arm's fixture on a device"]
    fn a_rewritten_row_keeps_the_bits_of_its_wide_chunk() {
        use sparkflash_runtime::{Pass, Segment};

        let fixture = DoryWalk::fixture(1).expect("the Dory fixture");
        let walk = fixture.walk;
        let prompt: Vec<u32> = (1..=40).collect();
        let mut whole = walk.open().expect("a sequence");
        let mut hidden = walk
            .walk(
                &mut [Segment::new(&mut whole, &prompt)],
                Pass::bulk(prompt.len()).with_prompt_end(0),
            )
            .expect("the whole prompt");
        let last = hidden.rows - 1;
        let cold = walk
            .head(
                &mut hidden,
                &mut [HeadRow::prompt(&mut whole, last, &prompt)],
            )
            .expect("the whole prompt's last row");
        let mut rewound = walk.open().expect("a second sequence");
        let (head, tail) = prompt.split_at(prompt.len() - 1);
        walk.walk(
            &mut [Segment::new(&mut rewound, head)],
            Pass::bulk(head.len()).with_prompt_end(tail.len()),
        )
        .expect("the prompt but its last id");
        let mut hidden = walk
            .walk(
                &mut [Segment::new(&mut rewound, tail)],
                Pass::bulk(tail.len()).with_prompt_end(0),
            )
            .expect("the last id on its own");
        let last = hidden.rows - 1;
        let warm = walk
            .head(
                &mut hidden,
                &mut [HeadRow::prompt(&mut rewound, last, &prompt)],
            )
            .expect("the rewritten last row");
        let bits = |row: &StepOutput| -> Vec<u32> {
            row.logits.iter().map(|value| value.to_bits()).collect()
        };
        assert_eq!(bits(&cold[0]), bits(&warm[0]));
    }

    /// A row's bits do not depend on where the runtime cut the prompt's chunks, which is what
    /// lets a reused prompt forward only the ids it added.
    #[test]
    #[ignore = "boots the arm's fixture on a device"]
    fn a_chunk_boundary_does_not_move_a_rows_bits() {
        use sparkflash_runtime::{Pass, Segment};

        let fixture = DoryWalk::fixture(1).expect("the Dory fixture");
        let walk = fixture.walk;
        let prompt: Vec<u32> = (1..=40).collect();
        let logits = |pieces: &[&[u32]]| -> Vec<u32> {
            let mut seq = walk.open().expect("a sequence");
            let mut hidden = Hidden::new(0);
            let mut all: Vec<u32> = Vec::new();
            for (index, piece) in pieces.iter().enumerate() {
                let last = index + 1 == pieces.len();
                all.extend_from_slice(piece);
                hidden = walk
                    .walk(
                        &mut [Segment::new(&mut seq, piece)],
                        Pass::bulk(piece.len()).with_prompt_end(if last { 0 } else { 1 }),
                    )
                    .expect("a pass");
            }
            let last = hidden.rows - 1;
            let step = walk
                .head(&mut hidden, &mut [HeadRow::prompt(&mut seq, last, &all)])
                .expect("the head");
            step[0].logits.iter().map(|value| value.to_bits()).collect()
        };
        let whole40 = logits(&[&prompt]);
        let split_24_16 = logits(&[&prompt[..24], &prompt[24..]]);
        let split_39_1 = logits(&[&prompt[..39], &prompt[39..]]);
        let whole16 = logits(&[&prompt[..16]]);
        let split_8_8 = logits(&[&prompt[..8], &prompt[8..16]]);
        // A tile boundary at 32, a last row on its own (the reuse path), and a first chunk
        // narrower than the wide tile.
        assert_eq!(whole40, split_24_16, "40 rows in one pass, then 24 and 16");
        assert_eq!(whole40, split_39_1, "40 rows in one pass, then 39 and 1");
        assert_eq!(whole16, split_8_8, "16 rows in one pass, then 8 and 8");
    }

    /// What the arm states for the generated profile tests: its ledger row, one positional
    /// region, no drafter, and the chunks of each cache mode. A per-loop prompt of 5,000 tokens
    /// runs in a whole chunk and the rest; a last-loop prompt runs a token a pass.
    fn stated() -> ArmProfile {
        let layout = StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 8,
        }]);
        let per_loop = walk_limits(true, 8_192);
        let last_loop = walk_limits(false, 8_192);
        ArmProfile::new("Dory", layout, per_loop)
            .with_code(DEVICE)
            .prompt(per_loop, 5_000, &[PREFILL_BATCH, 5_000 - PREFILL_BATCH])
            .prompt(last_loop, 3, &[1, 1, 1])
    }

    sparkflash_runtime::conformance!(DoryWalk, profile: stated());
}
