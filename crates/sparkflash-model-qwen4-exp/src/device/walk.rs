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

//! The Qwen4-Exp arm as the runtime's planned walk: [PlanWalk] over [QwenHost].
//!
//! The shared runtime owns the prompt parts, history, resume, rewind, lane pool and ledger,
//! and [sparkflash_runtime::walk_rows] owns the layer loop. This file supplies the arm's
//! arithmetic as a host: a layer is its PLE block (where it has one), its attention (Gated
//! DeltaNet or sparse) and its experts, each a part inside the widened residual's frames, and
//! every part, frame and head runs the arm's existing kernels in the order the old loops ran
//! them.
//!
//! A prompt pass is split into the prefill plan's forwards ([PlanHost::windows]), so a prompt's
//! chunk boundaries and therefore its bits do not move. A token or verify pass is one window
//! over every member's rows in the batch's combined buffers ([BatchBuffers]): the stages that
//! read a layer's weights run once over every row, and the stages that touch a sequence's own
//! state run once a member. A verify settles each member inside its window, as the batch did;
//! a sequence's held answers reach the runtime only through [Draft::Held].
//!
//! Qwen serves through this walk: `Qwen4ExpDeviceBackend::enable_fusion` and
//! `enable_group_fusion` start the tick engine ([sparkflash_runtime::TickServe]) over it, alone
//! or as rank 0 of a tensor-parallel group, so the fragment is the serving path. The engine
//! chooses when a chunk or a step runs; the rows and the shape are the runtime's, which is what
//! keeps the arm's chunk boundaries and its verify width.

use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex, MutexGuard};

use sparkflash_model::TpPlan;
use sparkflash_model::backend::{BackendError, Speculation, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_runtime::{
    ChunkPolicy, ComputeDeclaration, Draft, DraftBudget, Drafter, ExpertUnionWall, Limits,
    NoVision, OutputConstraint, PartExec, PartHost, Pass, PassKind, PassMask, PlanBranch, PlanHost,
    PlanLane, PlanPart, PlanPass, PlanRows, PlanSeq, PlanWalk, PlanWindow, RankSplit, Region,
    Segment, Segments, StateKind, StepWall, Walk, WalkCapabilities, check_rewind,
};

/// The verify budget: every member's anchor and its whole depth. The batch buffers hold that
/// many rows, so a wider budget would claim memory the verify does not have and a narrower one
/// would refuse a chain the runtime asked for.
const DRAFT_BUDGET: DraftBudget = DraftBudget::AnchorsAndDrafts;

use sparkflash_kernels::cuda_driver::Deviceptr;

use super::batch::{BatchBuffers, MAX_BATCH};
use super::{DENSE_ROWS, DeviceSequence, GpuBranch, SCRATCH_SLOTS, SequenceShared};

/// A shared walk's cost at TP=1 on a GB10-class node, from the block drafter's own step cost
/// ([super::DFLASH2_STEP_COST]: 59 ms a step and 8.4 ms a verified draft): about 10 new experts
/// a row at 0.8 ms each, the rest of a step fixed, and a few milliseconds a member past the first
/// for its own attention, recurrence and draft. The expert counts are the checkpoint's.
pub(super) const JOINT_WALL: ExpertUnionWall = ExpertUnionWall {
    fixed_ms: 50.4,
    expert_ms: 0.81,
    row_ms: 0.5,
    seq_ms: 6.0,
    experts: 0,
    top_k: 0,
};

/// The error code of every refusal of the walk.
const DEVICE: &str = "qwen4_device";

/// The arm's walk: the runtime's planned walk over [QwenHost].
pub(super) type QwenWalk = PlanWalk<QwenHost>;

/// A sequence's handle, shared by its lane and the window that walks it.
type Handle = Arc<Mutex<DeviceSequence>>;

fn lock(device: &Mutex<DeviceSequence>) -> Result<MutexGuard<'_, DeviceSequence>, BackendError> {
    device
        .lock()
        .map_err(|_| BackendError::new(DEVICE, "a sequence's state is poisoned"))
}

/// A sequence's device state, as a planned walk's lane.
///
/// The sequence sits behind a lock so a window's frames, which see only the scratch, and its
/// parts, which see the segments, reach the sequence whose buffers the rows sit in. The walk
/// takes the locks one at a time, as the old loops called one sequence at a time.
pub(super) struct QwenLane {
    device: Handle,
    /// Whether the sequence's next drafts are proposed: a verify proposes them with the answer
    /// it settled; a prompt's or a plain token row's head chooses on the host, and the round
    /// proposes once it opens on the chosen token ([QwenDrafter::open_round]).
    drafted: bool,
    /// The step the last verify settled for this sequence, until [QwenDrafter::verify] hands it
    /// over.
    settled: Option<StepOutput>,
}

impl QwenLane {
    fn lock(&self) -> Result<MutexGuard<'_, DeviceSequence>, BackendError> {
        lock(&self.device)
    }
}

impl PlanLane for QwenLane {
    const DEVICE: &'static str = DEVICE;

    fn save(&mut self, at: usize) -> Result<(), BackendError> {
        let mut device = self.lock()?;
        if device.rewind_point() == Some(at) {
            return Ok(());
        }
        device.keep_point()
    }

    fn restore(&mut self, to: usize) -> Result<(), BackendError> {
        self.lock()?.rewind_state(to)
    }

    fn clear(&mut self) -> Result<(), BackendError> {
        self.lock()?.clear_state()
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        self.lock()
            .is_ok_and(|device| device.room(positions, reset))
    }

    /// The adapter sets a request's sampling once, and a kept sequence carries the last one's
    /// counts: a request reports its own speculation, not the session's lifetime.
    fn requested(&mut self) {
        if let Ok(mut device) = self.lock() {
            device.spec.reset();
        }
    }
}

/// Runs `body` on `seq`'s device with the walk's sampler and sampling in place of the
/// sequence's own: one sampler a sequence, whichever of the walk or the sequence's settle draws.
fn with_sampler<R>(
    seq: &mut PlanSeq<QwenLane>,
    body: impl FnOnce(&mut DeviceSequence) -> Result<R, BackendError>,
) -> Result<R, BackendError> {
    let mut device = lock(&seq.lane.device)?;
    std::mem::swap(&mut device.sampler, &mut seq.sampler);
    device.sampling = seq.sampling;
    let answer = body(&mut device);
    std::mem::swap(&mut device.sampler, &mut seq.sampler);
    answer
}

/// A branch's frame: where a part's input comes from and where its output lands.
pub(super) enum QwenFrame {
    /// The PLE block reads the residual and adds into it itself.
    Landed,
    /// Layer `0`'s attention, between its attention hyper-connection and the inject.
    Attention(usize),
    /// Layer `0`'s experts, between its MLP hyper-connection and the drafter's capture.
    Ffn(usize),
}

/// The parts' host: what a pass hands a Qwen part.
pub(super) struct QwenParts;

impl PartHost for QwenParts {
    type Pass<'a> = QwenPass<'a>;
    type Open<'a> = ();
    type Handle = PlanPart<QwenParts>;
}

/// One pass over one Qwen part.
pub(super) struct QwenPass<'a> {
    host: &'a QwenHost,
    scratch: &'a QwenScratch,
    layer: usize,
}

impl QwenPass<'_> {
    fn layer(&self) -> &super::GpuLayer {
        &self.host.shared.weights.layers[self.layer]
    }
}

/// The per-layer embedding: the layer's n-gram rows of each sequence's own history, convolved
/// and added back into the widened residual.
struct PlePart;

/// A Gated DeltaNet attention.
struct GdnPart;

/// A sparse (QSA) or full attention; the QSA indexer is folded in, as its projection sits in
/// the same grouped launch as q, k and v.
struct QsaPart;

/// The routed and shared experts, with the TP sum and the inject they end in.
struct MoePart;

impl PartExec<QwenParts> for PlePart {
    fn state(&self) -> &[Region] {
        // The arm's one recurrent region holds the PLE windows ([QwenHost::extra_state]).
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Replicated
    }

    fn rows(&self, pass: &mut QwenPass<'_>) -> Result<(), BackendError> {
        let window = pass.scratch.window()?;
        let mut lead = lock(&window.lead)?;
        lead.own_rows = window.shares.filter(|_| pass.layer >= window.parallel_from);
        let stride = lead.snapshot_stride as u64;
        lead.ple_step(
            pass.layer,
            pass.layer(),
            &window.tokens,
            window.rows,
            window.shares,
            0,
            0,
            stride,
        )
    }

    fn token(&self, pass: &mut QwenPass<'_>) -> Result<(), BackendError> {
        let window = pass.scratch.window()?;
        let index = pass.layer;
        for (member, handle) in window.members.iter().enumerate() {
            let width = window.widths[member];
            // A member's rows but the last save their windows: the state after its last row is
            // the state the next step would build anyway, so it is not worth a region.
            let verify = width - 1;
            let at = window.offsets[member];
            let mut session = lock(handle)?;
            let (_, ple_slot) = session.snapshot_slots(index);
            let ple_snapshot = match (verify > 0, ple_slot) {
                (true, Some(offset)) => session.snapshot.offset(offset * 4).0,
                _ => 0,
            };
            let stride = session.snapshot_stride as u64;
            session.ple_step(
                index,
                pass.layer(),
                &window.tokens[at..at + width],
                width,
                None,
                ple_snapshot,
                verify,
                stride,
            )?;
        }
        Ok(())
    }
}

impl PartExec<QwenParts> for GdnPart {
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut QwenPass<'_>) -> Result<(), BackendError> {
        attend_rows(pass)
    }

    fn token(&self, pass: &mut QwenPass<'_>) -> Result<(), BackendError> {
        let window = pass.scratch.window()?;
        let (index, layer) = (pass.layer, pass.layer());
        let (mixed, _) = pass.scratch.entered.get();
        let total = window.rows;
        {
            let lead = lock(&window.lead)?;
            lead.linear_project(&layer.branch, mixed, lead.linear_b, lead.linear_a, total)?;
        }
        for (member, handle) in window.members.iter().enumerate() {
            let width = window.widths[member];
            let session = lock(handle)?;
            let (conv_slot, _) = session.snapshot_slots(index);
            let conv_snapshot = match (width > 1, conv_slot) {
                (true, Some(offset)) => session.snapshot.offset(offset * 4).0,
                _ => 0,
            };
            // A verify keeps the rows a rejected draft would give back, and the replay that
            // commits the kept ones reads them from the session's own inputs.
            let (conv_out, b_out, a_out) = match (width > 1, session.gdn_replay[index]) {
                (true, Some(inputs)) => session.replay_slices(inputs),
                _ => (session.linear_conv, session.linear_b, session.linear_a),
            };
            if width > 1 {
                session.stash_gate_inputs(b_out, a_out, width)?;
            }
            session.linear_state(
                index,
                &layer.branch,
                conv_out,
                b_out,
                a_out,
                width,
                session.chunk_path(width),
                width - 1,
                conv_snapshot,
                session.snapshot_stride as u64,
            )?;
        }
        lock(&window.lead)?.linear_output(index, &layer.branch, total)
    }
}

impl PartExec<QwenParts> for QsaPart {
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut QwenPass<'_>) -> Result<(), BackendError> {
        attend_rows(pass)
    }

    fn token(&self, pass: &mut QwenPass<'_>) -> Result<(), BackendError> {
        let window = pass.scratch.window()?;
        let (index, layer) = (pass.layer, pass.layer());
        let (mixed, _) = pass.scratch.entered.get();
        let total = window.rows;
        lock(&window.lead)?.sparse_project(&layer.branch, mixed, total)?;
        for (member, handle) in window.members.iter().enumerate() {
            let session = lock(handle)?;
            session.sparse_attend(
                &layer.branch,
                index,
                session.position,
                window.widths[member],
            )?;
        }
        lock(&window.lead)?.sparse_output(&layer.branch, index, total)
    }
}

/// A prompt forward's attention: the arm's own block over the whole window, from the window's
/// first position.
fn attend_rows(pass: &QwenPass<'_>) -> Result<(), BackendError> {
    let window = pass.scratch.window()?;
    let (mixed, _) = pass.scratch.entered.get();
    let lead = lock(&window.lead)?;
    let stride = lead.snapshot_stride as u64;
    lead.attend(
        pass.layer,
        pass.layer(),
        mixed,
        window.position,
        window.rows,
        window.chunk,
        0,
        0,
        stride,
    )
}

impl PartExec<QwenParts> for MoePart {
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut QwenPass<'_>) -> Result<(), BackendError> {
        let window = pass.scratch.window()?;
        let (mixed, injection) = pass.scratch.entered.get();
        let lead = lock(&window.lead)?;
        lead.mlp(
            &pass.layer().moe,
            mixed,
            lead.hyper,
            injection,
            pass.layer,
            window.chunk,
            window.rows,
        )
    }

    fn token(&self, pass: &mut QwenPass<'_>) -> Result<(), BackendError> {
        let window = pass.scratch.window()?;
        let (mixed, injection) = pass.scratch.entered.get();
        let lead = lock(&window.lead)?;
        // The routed experts do not share between different tokens, but the router, the shared
        // expert and both hyper-connections read their weights once for every row. On the
        // chunk path the routed experts are the chunk's MMA, whose rows are computed alone.
        let chunk = lead.chunk_path(window.rows);
        lead.mlp(
            &pass.layer().moe,
            mixed,
            lead.hyper,
            injection,
            pass.layer,
            chunk,
            window.rows,
        )
    }
}

/// One window of a walk, as its begin set it up.
struct Window {
    /// The sequence whose buffers the window's rows sit in: the prompt's, or the batch's first
    /// member, whose scratch is the batch's combined rows.
    lead: Handle,
    /// Whether the window is a token or verify pass's batch.
    batch: bool,
    /// The window's rows and their ids.
    rows: usize,
    tokens: Vec<u32>,
    /// A prompt forward's first position, its kernel path and its sequence-parallel layout: the
    /// rank's share of the rows, the first layer it splits from (the one after the last PLE
    /// block) and the first it runs as overlapped halves (the layer count when it does not).
    position: usize,
    chunk: bool,
    shares: Option<(usize, usize)>,
    /// The prompt's rows this pass takes after a prompt forward ([PlanWindow::after]).
    after: usize,
    parallel_from: usize,
    overlap_from: usize,
    /// A batch's members in row order (the first is the lead), each one's rows and first row,
    /// and what each one's verify settles against.
    members: Vec<Handle>,
    widths: Vec<usize>,
    offsets: Vec<usize>,
    saved: Vec<[Deviceptr; SCRATCH_SLOTS]>,
    positions: Vec<usize>,
    hashers: Vec<Arc<Vec<Option<crate::ngram::PleHasher>>>>,
    drafts: Vec<Vec<u32>>,
    confidence: Vec<Vec<f32>>,
}

/// What the head of the last walk reads.
enum Head {
    /// No head row waits.
    None,
    /// A prompt's last forward deferred its head to this sequence ([DeviceSequence::deferred_logits]).
    Deferred(Handle),
    /// A token pass's logits, every row's, already read back.
    Rows(Vec<f32>),
}

/// One walk's buffers: the batch's combined rows, made on the first token pass and reused, and
/// what a window's frames and parts and the head read.
pub(super) struct QwenScratch {
    batch: RefCell<Option<BatchBuffers>>,
    window: RefCell<Option<Window>>,
    /// The input a frame's entry left and the injection weights its exit reads.
    entered: Cell<(Deviceptr, Deviceptr)>,
    head: RefCell<Head>,
}

impl QwenScratch {
    fn window(&self) -> Result<std::cell::Ref<'_, Window>, BackendError> {
        let window = self
            .window
            .try_borrow()
            .map_err(|_| BackendError::new(DEVICE, "the walk's window is being changed"))?;
        std::cell::Ref::filter_map(window, Option::as_ref)
            .map_err(|_| BackendError::new(DEVICE, "a Qwen part ran outside a window"))
    }

    /// Hands a batch window's members back to their own streams and scratch.
    fn exit(&self, window: &Window) -> Result<(), BackendError> {
        if !window.batch {
            return Ok(());
        }
        let batch = self.batch.borrow();
        let batch = batch
            .as_ref()
            .ok_or_else(|| BackendError::new(DEVICE, "a batch window has no batch buffers"))?;
        let mut guards = window
            .members
            .iter()
            .map(|handle| lock(handle))
            .collect::<Result<Vec<_>, _>>()?;
        let mut members: Vec<&mut DeviceSequence> =
            guards.iter_mut().map(|guard| &mut **guard).collect();
        batch.exit(&mut members, &window.saved)
    }
}

/// The Qwen arm as a planned walk's host: its layers' parts and frames over the arm's kernels.
pub(super) struct QwenHost {
    shared: Arc<SequenceShared>,
    context: usize,
    limits: Limits,
    /// Whether the arm drafts at all with the joint switch as set.
    joint: bool,
    branches: Vec<Vec<PlanBranch<QwenParts, QwenFrame>>>,
}

impl QwenHost {
    /// The host over `shared`'s weights, serving sequences of up to `context_limit` tokens, with
    /// room for `capacity` members in one token pass.
    ///
    /// # Errors
    /// A joint-rounds switch the runtime refuses.
    pub(super) fn new(
        shared: Arc<SequenceShared>,
        context_limit: usize,
        capacity: usize,
    ) -> Result<Self, BackendError> {
        let limits = walk_limits(shared.prefill.chunk, context_limit, capacity);
        let joint =
            sparkflash_runtime::joint_rounds().map_err(|error| BackendError::new(DEVICE, error))?;
        let mut slot = 0;
        let mut branch = |frame: QwenFrame, part: PlanPart<QwenParts>, summed: bool| {
            slot += 1;
            PlanBranch {
                frame,
                part,
                summed,
                slot: slot - 1,
            }
        };
        let branches = shared
            .weights
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let mut branches = Vec::with_capacity(3);
                if layer.ple.is_some() {
                    branches.push(branch(QwenFrame::Landed, Box::new(PlePart), false));
                }
                let attention: PlanPart<QwenParts> = match layer.branch {
                    GpuBranch::Linear { .. } => Box::new(GdnPart),
                    GpuBranch::Sparse { .. } => Box::new(QsaPart),
                };
                branches.push(branch(QwenFrame::Attention(index), attention, true));
                branches.push(branch(QwenFrame::Ffn(index), Box::new(MoePart), true));
                branches
            })
            .collect();
        Ok(Self {
            shared,
            context: context_limit,
            limits,
            joint,
            branches,
        })
    }

    /// The walk over this host, with room for `capacity` members in one token pass.
    ///
    /// # Errors
    /// As [Self::new].
    pub(super) fn walk(
        shared: Arc<SequenceShared>,
        context_limit: usize,
        capacity: usize,
    ) -> Result<QwenWalk, BackendError> {
        let host = Self::new(shared, context_limit, capacity)?;
        PlanWalk::new(Arc::new(host), capacity)
    }

    /// A prompt forward's window over `seq`: the prelude [DeviceSequence::forward_batch] runs
    /// before its layers.
    fn begin_prompt(
        &self,
        scratch: &QwenScratch,
        seq: &mut PlanSeq<QwenLane>,
        tokens: &[u32],
        after: usize,
    ) -> Result<Window, BackendError> {
        seq.lane.drafted = false;
        let mut device = seq.lane.lock()?;
        // A proposal from the previous round anchors on the old end, which this prompt may
        // replace; the fusion's begin_prompt cleared the same three lists. The prompt's head
        // drafts afresh once its round opens.
        device.spec_drafts.clear();
        device.spec_confidence.clear();
        device.spec_served.clear();
        device.consumed.extend_from_slice(tokens);
        let batch = tokens.len();
        device.probe_rows = batch;
        // This forward is one named coverage input, and the monitor probe when it is a decode
        // step, when either switch names a path; off, `begin` is an `Option` check
        // (docs/NUMERICS.md §5).
        device.begin_numerics(tokens, batch);
        // A wide chunk takes the tensor cores, and with the decode steps there every forward
        // does.
        let chunk = device.chunk_path(batch);
        // A sharded chunk splits its hyper-connections over the ranks from the layer after the
        // last PLE block on: a PLE block convolves each row with the rows before it, which a rank
        // holding only its own share would not have.
        device.own_rows = None;
        device.window = 0;
        let shares = device.sequence_parallel(batch, chunk);
        let layers = &self.shared.weights.layers;
        let parallel_from = layers
            .iter()
            .rposition(|layer| layer.ple.is_some())
            .map_or(0, |last| last + 1);
        // From there on, a chunk whose halves split into whole pieces runs them overlapped.
        let overlap_from = if shares.is_some() && device.overlaps(batch) {
            parallel_from
        } else {
            layers.len()
        };
        if super::prof::enabled() {
            super::prof::begin();
        }
        *scratch.head.borrow_mut() = Head::None;
        Ok(Window {
            lead: Arc::clone(&seq.lane.device),
            batch: false,
            rows: batch,
            tokens: tokens.to_vec(),
            position: device.position,
            chunk,
            shares,
            after,
            parallel_from,
            overlap_from,
            members: Vec::new(),
            widths: Vec::new(),
            offsets: Vec::new(),
            saved: Vec::new(),
            positions: Vec::new(),
            hashers: Vec::new(),
            drafts: Vec::new(),
            confidence: Vec::new(),
        })
    }

    /// A token or verify pass's window: every member's rows in the batch's combined buffers.
    fn begin_batch(
        &self,
        scratch: &QwenScratch,
        segments: &mut PlanRows<'_, '_, QwenLane>,
        verify: bool,
    ) -> Result<Window, BackendError> {
        let mut widths = Vec::with_capacity(segments.len());
        let mut tokens = Vec::new();
        let mut drafts = Vec::with_capacity(segments.len());
        let mut confidence = Vec::with_capacity(segments.len());
        for (seq, rows) in segments.iter_mut() {
            let mut device = lock(&seq.lane.device)?;
            let (anchor, rest) = rows.split_first().ok_or_else(|| {
                BackendError::new(DEVICE, "a token pass holds a segment with no rows")
            })?;
            if verify {
                // The drafter's own confidence in these leading drafts, as the old fusion's round
                // passed it to the batch: a policy calibrated on it must not be fed zeroes.
                confidence.push(
                    device
                        .spec_confidence
                        .iter()
                        .take(rest.len())
                        .copied()
                        .collect(),
                );
            } else {
                if !rest.is_empty() {
                    return Err(BackendError::new(
                        DEVICE,
                        "a token pass holds one row a segment",
                    ));
                }
                // A held answer is the drafter's to hand over (Draft::Held): walking its lane
                // would change the pass's row count, which picks the token kernels' split.
                if !device.spec_queue.is_empty() {
                    return Err(BackendError::new(
                        DEVICE,
                        "a token pass holds a sequence whose last verify holds its answer; \
                         held answers come through the drafter's round, not a token row",
                    ));
                }
                device.consumed.push(*anchor);
                seq.lane.drafted = false;
                // A proposal waiting for this row is not verified by a plain token row.
                device.spec_drafts.clear();
                device.spec_confidence.clear();
                confidence.push(Vec::new());
            }
            drafts.push(rest.to_vec());
            widths.push(rows.len());
            tokens.extend_from_slice(rows);
        }
        let mut batch = scratch.batch.borrow_mut();
        if batch.is_none() {
            let score_stride = lock(&segments[0].0.lane.device)?.score_stride;
            let rows =
                (self.limits.members.max(segments.len()) * self.shared.verify_rows.max(1)).max(1);
            *batch = Some(BatchBuffers::new(
                Arc::clone(&self.shared),
                score_stride,
                rows,
            )?);
        }
        let buffers = batch.as_ref().expect("the buffers were just made");
        let members: Vec<Handle> = segments
            .iter()
            .map(|(seq, _)| Arc::clone(&seq.lane.device))
            .collect();
        let mut offsets = Vec::with_capacity(widths.len());
        let mut at = 0usize;
        for width in &widths {
            offsets.push(at);
            at += width;
        }
        let (saved, positions, hashers) = {
            let mut guards = members
                .iter()
                .map(|handle| lock(handle))
                .collect::<Result<Vec<_>, _>>()?;
            let mut sessions: Vec<&mut DeviceSequence> =
                guards.iter_mut().map(|guard| &mut **guard).collect();
            let saved = buffers.enter(&mut sessions, &widths)?;
            // A verify gives back the rows the model did not agree with, so each member's n-gram
            // windows are its own to restore: the state from before this step, and the rows it
            // kept.
            let positions = sessions.iter().map(|session| session.position).collect();
            let hashers = sessions
                .iter()
                .map(|session| Arc::new(session.ple_hashers.clone()))
                .collect();
            let lead = &mut *sessions[0];
            lead.probe_rows = at;
            // A batch never runs sequence-parallel; a forward that failed part-way may have
            // left its share set.
            lead.own_rows = None;
            (saved, positions, hashers)
        };
        if super::prof::enabled() {
            super::prof::begin();
        }
        *scratch.head.borrow_mut() = Head::None;
        Ok(Window {
            lead: Arc::clone(&members[0]),
            batch: true,
            rows: at,
            tokens,
            position: 0,
            chunk: false,
            shares: None,
            after: 0,
            parallel_from: 0,
            overlap_from: self.shared.weights.layers.len(),
            members,
            widths,
            offsets,
            saved,
            positions,
            hashers,
            drafts,
            confidence,
        })
    }

    /// A prompt forward's tail ([DeviceSequence::forward_batch]'s, then the old prompt loop's):
    /// the overlapped layers, the deferred head of the prompt's last forward, and the drafter's
    /// commits or the MTP's pass over the forward.
    fn finish_prompt(
        &self,
        scratch: &QwenScratch,
        window: &Window,
        pass: PlanPass,
        last: bool,
    ) -> Result<(), BackendError> {
        let mut device = lock(&window.lead)?;
        let layers = &self.shared.weights.layers;
        let batch = window.rows;
        if window.overlap_from < layers.len() {
            device.overlapped_layers(
                &layers[window.overlap_from..],
                window.overlap_from,
                window.chunk,
                batch,
            )?;
        } else if device.own_rows.is_some() {
            device.gather_features()?;
        }
        // Only the prompt's last forward heads a row: it asks for the last row's widened
        // residual here and runs the mixer, the language head and the read-back when the
        // runtime heads the finished prompt. An earlier forward skips the head-row gather (a
        // collective in a TP group) altogether.
        let last = last && pass.prompt_end.is_none_or(|after| after == 0);
        let head = if last {
            Some(device.head_rows(batch, 1)?)
        } else {
            None
        };
        device.own_rows = None;
        device.window = 0;
        if let Some(head) = head {
            device.pending_head = Some(head);
        }
        let start = device.position;
        super::prof::report(start + batch - 1, batch);
        device.position += batch;
        device.finish_numerics();
        if device.drafting() {
            // The drafter's commit window reaches the prompt's true end, as the old prompt
            // loop's did, so a non-final pass does not stop the window at the chunk boundary.
            let end = start + batch + window.after + pass.prompt_end.unwrap_or(0);
            let commits: Vec<_> = device
                .backend
                .dflash
                .as_ref()
                .map_or_else(Vec::new, |draft| {
                    draft
                        .config
                        .window_commits((start, batch), end, DENSE_ROWS)
                        .collect()
                });
            for (row, rows, position) in commits {
                device.dflash_commit(row, rows, position)?;
            }
        } else if device.spec_enabled {
            device.assemble_mtp_rows(batch)?;
            device.mtp_pass(&window.tokens, start, false)?;
            device.stash_last_hyper(batch)?;
        }
        if last {
            // The point is the prompt's end, before any draft past it, exactly as the old
            // prompt head took it: the MTP stash the drafts rewrite is part of the point, so
            // saving after them would restore a different state.
            device.keep_point()?;
            *scratch.head.borrow_mut() = Head::Deferred(Arc::clone(&window.lead));
        }
        Ok(())
    }

    /// A token or verify pass's tail ([BatchBuffers]' step's): the mixer and the language head
    /// over every row, read back; then a verify settles each member on its rows, and a token
    /// row leaves its logits to the head.
    fn finish_batch(
        &self,
        scratch: &QwenScratch,
        window: &Window,
        segments: &mut PlanRows<'_, '_, QwenLane>,
        verify: bool,
    ) -> Result<(), BackendError> {
        let total = window.rows;
        let logits = {
            let lead = lock(&window.lead)?;
            let weights = &self.shared.weights;
            let (mixed, _) = lead.hyper(&weights.mixer, lead.hyper, false, total)?;
            lead.gemv(&weights.lm_head, mixed, lead.logits, total)?;
            lead.probe_finite(
                48,
                "logits",
                lead.logits,
                self.shared.config.vocab_size * total,
            )?;
            lead.prof_mark("lm_head")?;
            let logits = lead.read_logits(total)?;
            lead.prof_mark("logits_dtoh")?;
            super::prof::report(lead.position, total);
            logits
        };
        let vocab = self.shared.config.vocab_size;
        for (member, (seq, rows)) in segments.iter_mut().enumerate() {
            let at = window.offsets[member];
            let width = window.widths[member];
            let position = window.positions[member];
            if verify {
                let step = with_sampler(seq, |device| {
                    device.position += width;
                    device.resolve_verify(
                        rows,
                        &logits[at * vocab..(at + width) * vocab],
                        position,
                        &window.drafts[member],
                        &window.confidence[member],
                        &window.hashers[member],
                    )
                })?;
                seq.lane.settled = Some(step);
                seq.lane.drafted = true;
                continue;
            }
            // A plain token row is [DeviceSequence::resolve_verify] with no drafts, whose
            // choice is the head's: nothing is agreed with and nothing is given back.
            let mut device = seq.lane.lock()?;
            device.position = position + 1;
            device.spec_served.clear();
            device.spec.observe(&[], 0, 1);
            if device.drafting() {
                // The drafter's rings take the row, and the next block is proposed once the
                // round opens on the answer.
                device.dflash_commit(0, 1, position)?;
            } else if device.spec_enabled {
                // The MTP's next chain starts from this row.
                device.stash_last_hyper(1)?;
            }
        }
        if !verify {
            *scratch.head.borrow_mut() = Head::Rows(logits);
        }
        Ok(())
    }

    /// The head rows `first..first + count` of the last walk, as logits: a prompt's deferred
    /// row, or a token pass's rows.
    fn head_logits(
        &self,
        scratch: &QwenScratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let vocab = self.shared.config.vocab_size;
        match &*scratch.head.borrow() {
            Head::Deferred(handle) => {
                if count != 1 {
                    return Err(BackendError::new(
                        DEVICE,
                        format!("a prompt heads its last row alone, not {count} rows"),
                    ));
                }
                lock(handle)?.deferred_logits()
            }
            Head::Rows(logits) => logits
                .get(first * vocab..(first + count) * vocab)
                .map(<[f32]>::to_vec)
                .ok_or_else(|| {
                    BackendError::new(
                        DEVICE,
                        format!(
                            "the last token pass holds no head rows {first}..{}",
                            first + count
                        ),
                    )
                }),
            Head::None => Err(BackendError::new(DEVICE, "the last walk left no head row")),
        }
    }
}

impl PlanHost for QwenHost {
    type Parts = QwenParts;
    type Frame = QwenFrame;
    type Lane = QwenLane;
    type Drafter = QwenDrafter;
    /// The arm reads text alone.
    type Vision = NoVision;
    type Scratch = QwenScratch;
    type Mode = ();

    const DEVICE: &'static str = DEVICE;
    /// The head reads every row back for the host's pick, and the arm's sessions have always
    /// answered a greedy step's logits.
    const KEEPS_LOGITS: bool = true;
    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        passes: PassMask::TOKEN_ROWS
            .union(PassMask::BULK)
            .union(PassMask::VERIFY),
        segments: Segments::Many,
        state: StateKind::Both,
        // The four rows the arm states in `sparkflash_model::ARM_LEDGERS`; a test holds them
        // equal so the walk and the table cannot drift.
        compute: ComputeDeclaration {
            prompt: State::Native("prompt chunks on the wide tile and the grouped expert MMAs"),
            token: State::Native("few-row twins of the tile at every row count"),
            stored: State::Native("NVFP4 weights with BF16 activation pairs"),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };

    fn layers(&self) -> usize {
        self.branches.len()
    }

    fn branches(&self, layer: usize) -> &[PlanBranch<QwenParts, QwenFrame>] {
        &self.branches[layer]
    }

    fn vocab(&self) -> usize {
        self.shared.config.vocab_size
    }

    fn pass_rows(&self) -> usize {
        self.limits.rows
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn scratch(_this: &Arc<Self>, _rows: usize) -> Result<QwenScratch, BackendError> {
        Ok(QwenScratch {
            batch: RefCell::new(None),
            window: RefCell::new(None),
            entered: Cell::new((Deviceptr(0), Deviceptr(0))),
            head: RefCell::new(Head::None),
        })
    }

    fn lane(this: &Arc<Self>) -> Result<QwenLane, BackendError> {
        let mut device = DeviceSequence::allocate(Arc::clone(&this.shared), this.context)?;
        // Without joint rounds the arm drafts nothing, exactly as the fusion's own policy did: a
        // member would otherwise run the MTP's prompt pass and then throw every chain away.
        device.spec_enabled &= this.joint;
        Ok(QwenLane {
            device: Arc::new(Mutex::new(device)),
            drafted: false,
            settled: None,
        })
    }

    fn drafter(this: &Arc<Self>) -> Result<QwenDrafter, BackendError> {
        let config = &this.shared.config;
        Ok(QwenDrafter {
            depth: this.shared.verify_rows.saturating_sub(1),
            wall: JOINT_WALL.with_experts(config.num_experts, config.num_experts_per_tok),
            joint: this.joint,
        })
    }

    fn mode(&self, _pass: PlanPass, _rows: usize) {}

    /// A prompt chunk runs as the prefill plan's forwards, each a window; a token or verify
    /// pass is one batch.
    fn windows(
        &self,
        pass: PlanPass,
        segments: &PlanRows<'_, '_, QwenLane>,
    ) -> Result<Vec<usize>, BackendError> {
        let rows: usize = segments.iter().map(|(_, ids)| ids.len()).sum();
        match pass.kind {
            PassKind::Bulk { .. } => {
                if segments.len() != 1 {
                    return Err(BackendError::new(
                        DEVICE,
                        "a bulk prompt pass holds one segment",
                    ));
                }
                Ok(self
                    .shared
                    .prefill
                    .forwards(rows)
                    .into_iter()
                    .map(|range| range.len())
                    .collect())
            }
            PassKind::TokenRows | PassKind::Verify { .. } => Ok(vec![rows]),
            PassKind::Paired { .. } => Err(BackendError::new(
                DEVICE,
                "the Qwen arm does not pair several prompts in one pass",
            )),
        }
    }

    fn begin_window(
        &self,
        scratch: &QwenScratch,
        segments: &mut PlanRows<'_, '_, QwenLane>,
        window: &PlanWindow,
    ) -> Result<(), BackendError> {
        // A window a fault left is handed back first.
        self.abandon_window(scratch);
        let after = window.after.first().copied().unwrap_or(0);
        let window = match window.pass.kind {
            PassKind::Bulk { .. } => {
                let (seq, tokens) = segments
                    .first_mut()
                    .ok_or_else(|| BackendError::new(DEVICE, "a prompt window holds no rows"))?;
                self.begin_prompt(scratch, seq, tokens, after)?
            }
            PassKind::TokenRows => self.begin_batch(scratch, segments, false)?,
            PassKind::Verify { .. } => self.begin_batch(scratch, segments, true)?,
            PassKind::Paired { .. } => {
                return Err(BackendError::new(
                    DEVICE,
                    "the Qwen arm does not pair several prompts in one pass",
                ));
            }
        };
        *scratch.window.borrow_mut() = Some(window);
        Ok(())
    }

    fn walked_layers(&self, scratch: &QwenScratch) -> usize {
        scratch
            .window()
            .map_or(self.layers(), |window| window.overlap_from)
    }

    fn finish(
        &self,
        scratch: &QwenScratch,
        segments: &mut PlanRows<'_, '_, QwenLane>,
        pass: PlanPass,
        last: bool,
    ) -> Result<(), BackendError> {
        let window = scratch
            .window
            .borrow_mut()
            .take()
            .ok_or_else(|| BackendError::new(DEVICE, "a window finished that never began"))?;
        if !window.batch {
            return self.finish_prompt(scratch, &window, pass, last);
        }
        let verify = matches!(pass.kind, PassKind::Verify { .. });
        // The members are settled first: a draft the model agreed with is kept by giving the
        // rest of the chain's state back, and that needs each member's scratch still pointed at
        // its rows. The exit runs whatever the settle returned.
        let settled = self.finish_batch(scratch, &window, segments, verify);
        let exited = scratch.exit(&window);
        settled.and(exited)
    }

    fn abandon_window(&self, scratch: &QwenScratch) {
        let left = scratch.window.borrow_mut().take();
        if let Some(window) = left {
            if let Ok(mut device) = lock(&window.lead) {
                device.own_rows = None;
                device.window = 0;
            }
            let _ = scratch.exit(&window);
        }
    }

    /// The arm's caches as the lane memory lays them out: no branch part declares its own.
    fn regions(&self) -> Vec<Region> {
        regions_of(&self.shared.config)
    }

    fn ends(&self, token: u32) -> bool {
        self.shared.config.eos_token_ids.contains(&token)
    }

    fn back(&self, lane: &mut QwenLane, reach: usize) -> Result<(), BackendError> {
        lane.lock()?.back(reach)
    }

    fn embed(&self, scratch: &QwenScratch, ids: &[u32]) -> Result<(), BackendError> {
        let window = scratch.window()?;
        let device = lock(&window.lead)?;
        let backend = Arc::clone(&self.shared);
        let config = &backend.config;
        let kernels = &*backend.kernels;
        let weights = &backend.weights;
        let width = config.hyper_width();
        if window.batch {
            for (row, &token) in ids.iter().enumerate() {
                device.call(
                    kernels.embed,
                    sparkflash_ops::calls::embed(
                        weights.embed.data().0,
                        device.hyper.offset(row * width * 4).0,
                        token,
                        (config.hidden_size, width),
                    ),
                )?;
            }
        } else {
            // The whole chunk in one launch, grid row t widening tokens[t], on the session's
            // stream like every other launch of the forward: the default stream the loader uses
            // is not ordered against it, so an embed left there raced the layers that read its
            // rows. The ids are read in place from the slab, which is rewritten only once the
            // last embed ran.
            let cuda = &backend.cuda;
            cuda.host_wait(&device.tokens_staged.0)
                .map_err(|error| BackendError::new(DEVICE, error))?;
            device
                .token_staging
                .write_at(0, bytemuck::cast_slice::<u32, u8>(ids))
                .map_err(|error| BackendError::new(DEVICE, error))?;
            device.call(
                kernels.embed,
                sparkflash_ops::calls::embed_rows(
                    weights.embed.data().0,
                    device.hyper.0,
                    device.token_staging.device().0,
                    (config.hidden_size, width),
                    ids.len(),
                ),
            )?;
            cuda.record_event_on(device.clock(), &device.tokens_staged.0)
                .map_err(|error| BackendError::new(DEVICE, error))?;
        }
        device.probe_finite(0, "embed", device.hyper, ids.len() * width)?;
        device.prof_mark("embed")
    }

    fn enter(
        &self,
        scratch: &QwenScratch,
        frame: &QwenFrame,
        rows: usize,
    ) -> Result<(), BackendError> {
        let (index, ffn) = match frame {
            QwenFrame::Landed => return Ok(()),
            QwenFrame::Attention(index) => (*index, false),
            QwenFrame::Ffn(index) => (*index, true),
        };
        let window = scratch.window()?;
        let mut device = lock(&window.lead)?;
        let layer = &self.shared.weights.layers[index];
        let block = if ffn {
            &layer.mlp_hyper
        } else {
            &layer.attn_hyper
        };
        let entered = if window.batch {
            let (mixed, injection) = device.hyper(block, device.hyper, true, rows)?;
            // The attention's exit injects with the sequence's own weights buffer; the experts'
            // body reads the ones this hyper-connection answers.
            let injection = match (ffn, injection) {
                (_, Some(injection)) => injection,
                (false, None) => Deviceptr(0),
                (true, None) => {
                    return Err(BackendError::new(
                        DEVICE,
                        "the MLP hyper-connection injects",
                    ));
                }
            };
            (mixed, injection)
        } else {
            if !ffn {
                device.own_rows = window.shares.filter(|_| index >= window.parallel_from);
            }
            device.branch_input(block, rows)?
        };
        if ffn {
            device.probe_finite(
                index,
                "mixed",
                entered.0,
                rows * self.shared.config.hidden_size,
            )?;
            device.prof_mark("hyper_mlp")?;
        } else {
            device.prof_mark("hyper_attn")?;
        }
        scratch.entered.set(entered);
        Ok(())
    }

    fn pass<'a>(
        &'a self,
        scratch: &'a QwenScratch,
        _segments: &'a [(&'a mut PlanSeq<QwenLane>, &'a [u32])],
        layer: usize,
        _slot: usize,
        _rows: usize,
        _mode: (),
    ) -> QwenPass<'a> {
        QwenPass {
            host: self,
            scratch,
            layer,
        }
    }

    fn leave(
        &self,
        scratch: &QwenScratch,
        frame: &QwenFrame,
        _next: Option<&QwenFrame>,
        _summed: bool,
        rows: usize,
    ) -> Result<(), BackendError> {
        let width = self.shared.config.hyper_width();
        let window = scratch.window()?;
        match frame {
            QwenFrame::Landed => Ok(()),
            QwenFrame::Attention(index) => {
                let device = lock(&window.lead)?;
                device.inject_branch(device.hyper, device.injection, rows)?;
                device.probe_finite(*index, "after_attn", device.hyper, rows * width)?;
                device.prof_mark("attn_inject")
            }
            QwenFrame::Ffn(index) => {
                {
                    let device = lock(&window.lead)?;
                    device.probe_finite(*index, "after_mlp", device.hyper, rows * width)?;
                    if !window.batch {
                        device.capture_features(*index, rows)?;
                    }
                }
                // The drafter reads a few layers' widened rows; a member's own slice is the rows
                // its next block anchors on.
                if window.batch {
                    for (member, handle) in window.members.iter().enumerate() {
                        lock(handle)?.capture_features(*index, window.widths[member])?;
                    }
                }
                lock(&window.lead)?.prof_mark("inject_mlp")
            }
        }
    }

    fn residual(&self, _scratch: &QwenScratch, _rows: usize) -> Result<Vec<f32>, BackendError> {
        Err(BackendError::new(
            DEVICE,
            "the Qwen walk keeps no layer trace: its residual is widened per stream",
        ))
    }

    fn logits(
        &self,
        scratch: &QwenScratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>, BackendError> {
        self.head_logits(scratch, first, count)
    }

    /// The host sampler's argmax of every row: the logits are read back whatever the pick, so
    /// the greedy pick is the sampler's own on the host.
    fn choices(
        &self,
        scratch: &QwenScratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<u32>, BackendError> {
        let logits = self.head_logits(scratch, first, count)?;
        Ok(logits
            .chunks(self.shared.config.vocab_size)
            .map(sparkflash_model::argmax)
            .collect())
    }
}

/// The arm's state, as the runtime needs to see it: the QSA (and MTP) caches are positional, and
/// the Gated DeltaNet windows and recurrence with the PLE windows are one recurrent region.
fn regions_of(config: &crate::Qwen4ExpConfig) -> Vec<Region> {
    let sparse = config
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, crate::config::LayerKind::Sparse))
        .count()
        + usize::from(config.serves_mtp());
    let kv = config.kv_proj_width() * 2 * 2;
    let index = config.indexer_head_dim * 2;
    let mut regions = Vec::new();
    if sparse > 0 {
        regions.push(Region::Positional {
            name: "qsa-kv",
            bytes_per_row: sparse * (kv + index),
        });
    }
    let linear = config
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, crate::config::LayerKind::Linear))
        .count();
    let history = config.linear_conv_kernel_dim.saturating_sub(1).max(1);
    let conv = linear * history * config.linear_conv_width() * 4;
    let recurrent = linear
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * 4;
    let ple_layers = config.ple_layer_ids.len();
    let ple = ple_layers
        * config.ple_conv_kernel_size.saturating_sub(1).max(1)
        * config.ngram_size
        * config.hyper_width()
        * 4;
    if conv + recurrent + ple > 0 {
        regions.push(Region::Recurrent {
            name: "gated-deltanet",
            bytes: conv + recurrent + ple,
        });
    }
    regions
}

/// The pass width the runtime's [ChunkPolicy] can return for the arm's own prefill chunk.
///
/// The policy quantises to its documented sizes, so a limit that is not one of them would move
/// the arm's chunk boundary. This picks the smallest documented width at or above the plan's
/// chunk that it divides, so the walk's prefill split lands on the plan's boundaries; a chunk
/// with no such width keeps its own value (the policy then rounds it down, and the arm's prompt
/// boundaries move with it).
fn pass_rows(chunk: usize) -> usize {
    let chunk = chunk.max(1);
    ChunkPolicy::sizes()
        .iter()
        .copied()
        .find(|size| *size >= chunk && size % chunk == 0)
        .or_else(|| {
            ChunkPolicy::sizes()
                .iter()
                .copied()
                .find(|size| *size >= chunk)
        })
        .unwrap_or(chunk)
}

/// The limits the runtime sees: the arm's own prefill chunk for a pass, the context, and the
/// batch's members.
///
/// The member count is the verify width, not the prompt chunk: a verify of `members` members is
/// `members x (1 + depth)` rows, and the batch buffers hold exactly that many. Capping it at
/// [MAX_BATCH] is the arm's own batch width.
fn walk_limits(chunk: usize, context: usize, capacity: usize) -> Limits {
    Limits::new(pass_rows(chunk), context).with_members(capacity.clamp(1, MAX_BATCH))
}

/// The arm's draft source: the checkpoint's MTP stage, or a loaded DFlash 2 head.
///
/// With `SPARKFLASH_JOINT_ROUNDS=off` the arm drafts nothing (the fusion's own policy): the
/// walk opens each sequence with speculation off, and the empty chain states it as well.
pub(super) struct QwenDrafter {
    /// The most drafts the verify batch holds, past the anchor.
    depth: usize,
    wall: ExpertUnionWall,
    /// Whether the arm drafts at all with the joint switch as set.
    joint: bool,
}

impl Drafter<QwenWalk> for QwenDrafter {
    const PRESENT: bool = true;
    /// The batch already folded each member's rows into its step.
    type Rows = StepOutput;

    fn wall(&self) -> &dyn StepWall {
        &self.wall
    }

    /// The member's held answer, or its next chain. A sequence whose last step was a prompt's
    /// head or a plain token row proposes the chain here, on the token it was fed, from the
    /// same state the head left: a verify proposed its own already.
    fn open_round(&self, seq: &mut PlanSeq<QwenLane>, token: u32) -> Result<Draft, BackendError> {
        let drafted = seq.lane.drafted;
        let (held, chain, drafts) = with_sampler(seq, |device| {
            device
                .backend
                .cuda
                .make_current()
                .map_err(|error| BackendError::new(DEVICE, error))?;
            if let Some(step) = device.feed(token)? {
                return Ok((Some(step), Vec::new(), Vec::new()));
            }
            if !drafted {
                let position = device.position;
                if device.drafting() {
                    device.dflash_propose(token, position)?;
                } else if device.spec_enabled {
                    device.draft_chain(token, position)?;
                }
            }
            let mut chain = device
                .spec
                .chain(&device.spec_confidence, device.spec_drafts.len());
            chain.truncate(device.spec_drafts.len());
            // The confidence stays on the sequence: the verify hands it back to the batch,
            // which records the round's calibration. Taking it here would make every verify
            // look like a drafter that offers no probabilities.
            let drafts = std::mem::take(&mut device.spec_drafts);
            Ok((None, chain, drafts))
        })?;
        if let Some(step) = held {
            return Ok(Draft::Held(step));
        }
        seq.lane.drafted = true;
        if !self.joint {
            return Ok(Draft::Chain {
                confidence: Vec::new(),
                drafts: Vec::new(),
            });
        }
        Ok(Draft::Chain {
            confidence: chain,
            drafts,
        })
    }

    /// One verify pass of the walk over every member's anchor and drafts; each member's step is
    /// settled inside the pass, while its rows are still the batch's.
    fn verify(
        &self,
        walk: &QwenWalk,
        segments: &mut [Segment<'_, PlanSeq<QwenLane>>],
        _depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        walk.walk(
            segments,
            Pass::verify(rows, OutputConstraint::new(&[], rows)),
        )?;
        segments
            .iter_mut()
            .map(|segment| {
                segment.seq.lane.settled.take().ok_or_else(|| {
                    BackendError::new(DEVICE, "a verify left a member without a step")
                })
            })
            .collect()
    }

    /// [DeviceSequence::resolve_verify] already settled the round inside the batch; the history
    /// keeps the rows the sequence kept.
    fn settle(
        &self,
        seq: &mut PlanSeq<QwenLane>,
        _chunk: &[u32],
        rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        let position = seq.lane.lock()?.position;
        seq.keep(position)?;
        Ok(rows)
    }

    /// The most rows one verify of `seqs` members may hold: every member's anchor and its
    /// drafts, which is the batch buffers' width. `share_rows` spends it on drafts only, so a
    /// chain's own depth still bounds what a member takes.
    fn budget(&self, seqs: usize) -> usize {
        DRAFT_BUDGET.rows(seqs, self.depth)
    }

    fn ready(&self, seq: &PlanSeq<QwenLane>, token: u32) -> bool {
        seq.lane.lock().is_ok_and(|device| {
            device
                .spec_queue
                .first()
                .is_some_and(|pending| pending.token == token)
        })
    }

    fn give_back(&self, seq: &mut PlanSeq<QwenLane>, to: usize) -> Result<(), BackendError> {
        let position = with_sampler(seq, |device| {
            let count = device.consumed.len().saturating_sub(to);
            if count == 0 {
                return Ok(device.position);
            }
            let served = device.spec_served.len();
            if count > served {
                return Err(BackendError::new(
                    DEVICE,
                    format!("{count} answers to give back, and the last verify served {served}"),
                ));
            }
            let pending = device
                .spec_served
                .drain(served - count..)
                .next()
                .expect("the served answers were just counted");
            device.spec_queue.clear();
            device.unwind(&pending)?;
            let consumed = device.consumed.len() - count;
            device.consumed.truncate(consumed);
            let history = device.sampler.history().len();
            device.sampler.truncate(history - count);
            Ok(device.position)
        })?;
        seq.keep(position)
    }

    fn speculation(&self, seq: &PlanSeq<QwenLane>) -> Speculation {
        seq.lane
            .lock()
            .map_or_else(|_| Speculation::default(), |device| device.spec.stats())
    }

    fn describe(&self) -> String {
        "DFlash 2 or MTP".to_owned()
    }
}

impl DeviceSequence {
    /// The runtime's rewind: a prompt end restores the arm's point, and zero clears the state a
    /// fresh sequence would have.
    pub(super) fn rewind_state(&mut self, to: usize) -> Result<(), BackendError> {
        check_rewind("qwen4_device", to, self.position)?;
        if to == self.position {
            return Ok(());
        }
        if to == 0 {
            return self.clear_state();
        }
        self.rewind_point()
            .is_some_and(|point| point == to)
            .then_some(())
            .ok_or_else(|| {
                BackendError::new(
                    "qwen4_device",
                    format!(
                        "the sequence holds no rewind point at position {to} (it holds {:?})",
                        self.rewind_point()
                    ),
                )
            })?;
        DeviceSequence::rewind(self, to)
    }

    /// Puts the sequence back to the state a fresh allocation holds: no positions, no state.
    fn clear_state(&mut self) -> Result<(), BackendError> {
        let cuda = &self.backend.cuda;
        cuda.make_current()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let config = &self.backend.config;
        let history = config.linear_conv_kernel_dim.saturating_sub(1).max(1);
        let conv_elems = history * config.linear_conv_width();
        let rec_elems = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let ple_elems = (config.ple_conv_kernel_size.saturating_sub(1).max(1))
            * config.ngram_size
            * config.hyper_width();
        let zero_gpu = |pointer: Deviceptr, elements: usize| -> Result<(), BackendError> {
            let blocks = elements.div_ceil(256).max(1) as u32;
            cuda.launch_on(
                &self.stream,
                self.backend.kernels.zero.kernel,
                (blocks, 1, 1),
                (256, 1, 1),
                &[pointer.0, elements as u64],
            )
            .map_err(|error| BackendError::new("qwen4_device", error))
        };
        for &pointer in self.gdn_conv.iter().flatten() {
            zero_gpu(pointer, conv_elems)?;
        }
        for &pointer in self.gdn_recurrent.iter().flatten() {
            zero_gpu(pointer, rec_elems)?;
        }
        for &pointer in self.ple_conv.iter().flatten() {
            zero_gpu(pointer, ple_elems)?;
        }
        zero_gpu(self.mtp_stash, config.hyper_width())?;
        self.position = 0;
        self.consumed.clear();
        self.prefill_cached = 0;
        self.pending_head = None;
        self.spec_queue.clear();
        self.spec_served.clear();
        self.spec_drafts.clear();
        self.spec_confidence.clear();
        self.ple_hashers = fresh_hashers(config)?;
        self.point_hashers = fresh_hashers(config)?;
        if let Some(point) = self.point.as_mut() {
            point.forget();
        }
        Ok(())
    }
}

/// A fresh set of n-gram windows, one per PLE layer.
fn fresh_hashers(
    config: &crate::Qwen4ExpConfig,
) -> Result<Vec<Option<crate::ngram::PleHasher>>, BackendError> {
    let mut hashers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        if let Some(ple_layer_index) = config.ple_layer_ids.iter().position(|id| *id == index + 1) {
            hashers.push(Some(
                crate::ngram::PleHasher::new(config, ple_layer_index)
                    .map_err(|error| BackendError::new("qwen4_device", error))?,
            ));
        } else {
            hashers.push(None);
        }
    }
    Ok(hashers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_runtime::testing::ArmProfile;
    use sparkflash_runtime::{ArmFixture, Fixture, FormatRule, StateLayout};

    use crate::config::Qwen4ExpConfig;
    use crate::device::prefill::{DEFAULT_PREFILL_CHUNK, PrefillPlan};

    /// The Qwen fixture: the synthetic tiny checkpoint, opened through the arm's own TP=1
    /// constructor. It is a real checkpoint in the released storage formats, not a served image.
    /// A marker with the walk's old name, which the record and the generated checks state: the
    /// walk is the runtime's type.
    struct QwenWalk;

    impl ArmFixture for QwenWalk {
        type Walk = PlanWalk<QwenHost>;

        /// Qwen's single-process path serves on the tick engine, so its generated checks drive
        /// that engine exactly as dense's and GLM's do. Its TP leg skips because the fixture is
        /// one rank.
        fn tick_engine() -> bool {
            true
        }

        fn fixture(world: usize) -> Result<Fixture<PlanWalk<QwenHost>>, BackendError> {
            // Measured on one host through sparkflash_runtime::local_group (2026-11-21): rank 0's
            // upload refuses the split, rightly, so the check skips with that reason.
            one_rank(world)?;
            // The Qwen3.5/3.6 spelling, not `tiny`: its family is not the canonical Qwen3.8
            // target, so the hub cache's released DFlash 2 drafter (2,560 wide) is not loaded
            // into a 128-wide fixture and refused. The tiny MTP stage still exercises the round.
            open_fixture(
                Qwen4ExpConfig::tiny_qwen35_moe(crate::config::DenseQuant::Fp8Block),
                "the Qwen3.5-MoE tiny synthetic checkpoint",
            )
        }
    }

    /// Both tiny fixtures are one rank.
    fn one_rank(world: usize) -> Result<(), BackendError> {
        if world > 1 {
            return Err(BackendError::new(
                "group",
                "the tiny FP8 fixture's 128-wide projections are one 128x128 scale tile, \
                 which TP 2 and TP 4 would split; the arm refuses that split",
            ));
        }
        Ok(())
    }

    /// The tiny checkpoint `config` describes, opened through the arm's own TP=1 constructor.
    fn open_fixture(
        config: Qwen4ExpConfig,
        what: &'static str,
    ) -> Result<Fixture<PlanWalk<QwenHost>>, BackendError> {
        let directory =
            tempfile::tempdir().map_err(|error| BackendError::new("fixture", error.to_string()))?;
        crate::write_synthetic_checkpoint(directory.path(), &config, 7)
            .map_err(|error| BackendError::new("fixture", error))?;
        let backend = crate::device::Qwen4ExpDeviceBackend::open(
            directory.path(),
            config.max_position_embeddings,
        )?;
        let walk = backend.walk_arm(4)?;
        let tp = sparkflash_model::TpPlan::new(1, 0)
            .map_err(|error| BackendError::new("fixture", error.detail))?;
        let tensors = sparkflash_ops::tensor_plan::ArchitectureTensorPlan::served_and_mtp(
            &config.architecture,
            tp,
        )
        .map_err(|error| BackendError::new("fixture", error.to_string()))?;
        let (bytes, floor) = sparkflash_runtime::token_bytes(
            tensors.rank_bytes(),
            tensors.role_bytes(sparkflash_ops::tensor_plan::Role::Experts),
            config.num_experts,
            config.num_experts_per_tok,
        );
        Ok(Fixture::new(
            walk,
            what,
            bytes,
            floor,
            FormatRule::classes(&config.architecture.quantization),
            Some(Box::new(directory)),
        ))
    }

    /// The Qwen3.8 shape the Qwen3.5 fixture lacks: hyper-connections, the per-layer embedding,
    /// the QSA indexer and the MTP stage, at the FP8 experts' 128-wide tiles. Its bits are
    /// recorded so a change to those paths is checked against the code before it.
    ///
    /// Its family is the canonical Qwen3.8 one, so a released DFlash 2 drafter in the hub cache
    /// would be loaded and refused at this width: record it with `HF_HUB_CACHE` pointing at an
    /// empty directory.
    struct Qwen38Fixture;

    impl sparkflash_runtime::ArmFixture for Qwen38Fixture {
        type Walk = PlanWalk<QwenHost>;

        fn fixture(world: usize) -> Result<Fixture<PlanWalk<QwenHost>>, BackendError> {
            one_rank(world)?;
            open_fixture(
                Qwen4ExpConfig::tiny_fp8(),
                "the Qwen3.8 tiny synthetic checkpoint (hyper-connections, PLE, QSA, MTP)",
            )
        }

        fn tick_engine() -> bool {
            true
        }
    }

    #[test]
    #[ignore = "boots the Qwen3.8 fixture; set SPARKFLASH_BITS_RECORD=<dir>"]
    fn qwen38_bits_equal_the_record() {
        match sparkflash_runtime::record::check_record::<Qwen38Fixture>("Qwen38Fixture") {
            Ok(report) => println!("record: Qwen38Fixture: {report}"),
            Err(error) => panic!("record: Qwen38Fixture: {}", error.message()),
        }
    }

    /// The runtime's chunk policy is the arm's own prefill chunk: a prompt's boundaries do not
    /// move when the runtime drives it.
    #[test]
    fn the_runtime_chunk_is_the_prefill_plans_chunk() {
        for chunk in [DEFAULT_PREFILL_CHUNK, super::super::TOKEN_PREFILL_CHUNK] {
            let plan = PrefillPlan::new(chunk, false);
            let limits = Limits::new(pass_rows(plan.chunk), 4_096);
            let rows = ChunkPolicy::for_limits(limits).rows();
            assert!(rows >= plan.chunk, "{chunk} rows must be a whole pass");
            assert_eq!(
                rows % plan.chunk,
                0,
                "{chunk} rows: the runtime's pass splits on the plan's boundary"
            );
        }
        assert_eq!(pass_rows(DEFAULT_PREFILL_CHUNK), DEFAULT_PREFILL_CHUNK);
        assert_eq!(pass_rows(super::super::TOKEN_PREFILL_CHUNK), 256);
    }

    /// The arm's state is both kinds: the QSA caches are positional and the Gated DeltaNet
    /// recurrence is not, so the runtime must checkpoint and truncate.
    #[test]
    fn the_layout_is_positional_attention_and_recurrent_gated_deltanet() {
        let layout = StateLayout::new(regions_of(&Qwen4ExpConfig::tiny()));
        assert_eq!(layout.kind(), StateKind::Both);
        assert!(layout.bytes_per_row() > 0, "the KV cache is positional");
        assert!(layout.needs_checkpoint(), "the recurrence needs a point");
    }

    /// The engine's member cap is the verify width, not the prompt chunk: a four-row batch
    /// steps at most four members, though its prompt pass holds the whole prefill chunk.
    #[test]
    fn the_member_cap_is_the_verify_width_not_the_prompt_chunk() {
        let limits = walk_limits(DEFAULT_PREFILL_CHUNK, 4_096, 9);
        assert_eq!(limits.rows, 4_096, "a prompt pass keeps the arm's chunk");
        assert_eq!(limits.members, 4, "a batch is the arm's verify width");
        assert_eq!(walk_limits(DEFAULT_PREFILL_CHUNK, 4_096, 0).members, 1);
        assert_eq!(walk_limits(DEFAULT_PREFILL_CHUNK, 4_096, 2).members, 2);
    }

    /// The drafter's budget is the rule the profile states, at the drafter's own depth.
    #[test]
    fn the_drafter_budgets_by_the_stated_rule() {
        for depth in [0, 1, 3] {
            let drafter = QwenDrafter {
                depth,
                wall: JOINT_WALL,
                joint: true,
            };
            for seqs in [0, 1, 4] {
                assert_eq!(drafter.budget(seqs), DRAFT_BUDGET.rows(seqs, depth));
            }
        }
    }

    /// The walk takes the passes the generated checks read: a prompt chunk and many segments.
    #[test]
    fn the_capabilities_support_every_generated_check() {
        let caps = <PlanWalk<QwenHost> as Walk>::CAPABILITIES;
        assert!(caps.passes.contains(PassMask::BULK));
        assert!(caps.passes.contains(PassMask::TOKEN_ROWS));
        assert_eq!(caps.segments, Segments::Many);
        assert_eq!(caps.state, StateKind::Both);
        assert_eq!(pass_rows(DEFAULT_PREFILL_CHUNK), 4_096);
        let walk_batch = super::super::batch::MAX_BATCH;
        assert_eq!(walk_batch, 4);
        sparkflash_runtime::check_claims::<PlanWalk<QwenHost>>(
            sparkflash_runtime::CONFORMANCE_CHECKS,
        )
        .expect("every generated check is supported");
    }

    /// The prefill plan's forwards inside the passes the profile states: a 4,096-row pass and
    /// its tail are one forward each, and the token-kernel plan's 256-row pass is two of its
    /// 128-row forwards.
    #[test]
    fn the_plans_forwards_fill_the_stated_passes() {
        let plan = PrefillPlan::new(DEFAULT_PREFILL_CHUNK, false);
        assert_eq!(plan.forwards(4_096), vec![0..4_096], "one forward a chunk");
        assert_eq!(plan.forwards(904), vec![0..904], "the tail is one forward");
        let plan = PrefillPlan::new(super::super::TOKEN_PREFILL_CHUNK, false);
        assert_eq!(plan.chunk, super::super::TOKEN_PREFILL_CHUNK);
        assert_eq!(pass_rows(plan.chunk), 256);
        assert_eq!(
            plan.forwards(256),
            vec![0..128, 128..256],
            "a pass is two of the plan's forwards"
        );
    }

    /// What the arm states for the generated profile and round tests: the prefill plan's chunk
    /// as the pass width and the batch's verify width as the member cap, as [QwenHost::new]
    /// states its limits; the MTP verify's depth under the arm's budget and wall; and the
    /// chunks the default plan's and the token-kernel plan's prompts run in.
    fn stated() -> ArmProfile {
        let members = super::super::batch::MAX_BATCH;
        let limits = walk_limits(DEFAULT_PREFILL_CHUNK, 65_536, members);
        ArmProfile::new(
            "Qwen",
            StateLayout::new(regions_of(&Qwen4ExpConfig::tiny())),
            limits,
        )
        .with_code("qwen4_shape")
        .drafting(super::super::MTP_VERIFY_ROWS - 1, DRAFT_BUDGET)
        .with_wall(JOINT_WALL)
        .prompt(limits, 5_000, &[4_096, 904])
        .prompt(
            walk_limits(super::super::TOKEN_PREFILL_CHUNK, 65_536, members),
            600,
            &[256, 256, 88],
        )
    }

    sparkflash_runtime::conformance!(QwenWalk, profile: stated(), rounds);
}
