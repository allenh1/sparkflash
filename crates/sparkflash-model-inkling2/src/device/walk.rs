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

//! Inkling-2-Small as the runtime's planned walk ([PlanWalk] over [InklingHost]), with its
//! DSpark round as the [Drafter].
//!
//! The runtime owns the prompt parts, the ids, the sampler, resume, rewind, the lane pool,
//! give-back and the ledger, and the planned walk runs every pass as windows of the layers'
//! branches; this module supplies only what Inkling alone knows:
//!
//! * [PlanHost::regions] is what a sequence's state is: every layer's key and value rows grow
//!   with position, and the four convolution histories a layer plus the drafter's Markov feature
//!   are one live value that a rewind restores from the prompt-boundary checkpoint.
//! * [PlanHost::limits] is [BATCH_ROWS] members a batch and the rank's prompt pass rows: the
//!   runtime's [sparkflash_runtime::ChunkPolicy] over the arm's widest width
//!   ([Shared::prefill_chunk]). That width is 4,096, 8,192 or 65,535 - every one an entry of
//!   [sparkflash_runtime::CHUNK_SIZES] - and a context or a switch that lowers it is quantised
//!   to the documented list, which is what the runtime always did before it ran a chunk. So a
//!   pass is the number the arm's prompt boundaries were already cut at, and the host's windows
//!   ([PlanHost::windows]) cut it in the arm's own widths (4,096 rows until 16,384 positions
//!   remain, 8,192 above that, or the switch's fixed width), which is what keeps the boundaries -
//!   and the bits that depend on them - the ones the arm measured.
//! * A window runs one of Inkling's row sets ([PlanHost::begin_window]): a prompt chunk
//!   ([Rows::Chunk], or the one-row step when the walk forwards a position at a time), one token
//!   a lane ([Rows::Lanes]), or an anchor and its drafts verified together ([Rows::Segments]).
//! * Each layer's two branches are stages of the arm's own schedule ([Stages]): the attention's
//!   entry is its block residual and norms, its part the projections through the output
//!   projection, its exit the delta's sum, the convolution and the landing; the experts' entry is
//!   the MLP's block residual and norm, their part the router and the band, their exit the
//!   delta's sum, the convolution and the landing. The stages run the schedule's steps in its
//!   order, so a step that runs ahead (an attention half sums its delta, a band its mix) is not
//!   run again by the next stage.
//! * The head is the arm's sampler over every row's logits ([PlanHost::KEEPS_LOGITS]): its
//!   Gumbel draw ([PlanHost::choose]) over the unpadded vocabulary ([PlanHost::logits_width]),
//!   and its end-of-generation ids ([PlanHost::ends]).
//!
//! The drafter is the arm's DSpark (or prompt-lookup) round with [Drafter]'s policies stated
//! once: a lone member verifies the depth the arm's own depth policy picked, and a joint round
//! shares rows through [Drafter::budget], which caps a verify at [BATCH_ROWS] rows
//! (members x (1 + depth)). With SPARKFLASH_INKLING_DRAFT unset the round is the plain step and
//! the runtime takes its answer without a walk ([Draft::Held]).
//!
//! The arm serves through this walk: a single process's sessions are handles on
//! [sparkflash_runtime::TickServe] over it, and a text-only group's rank 0 and workers drive the
//! tick engine and its mirror. A group whose rank 0 holds the vision tower still serves through
//! [sparkflash_runtime::Fusion]'s group: rank 0's image encode is a collective that would run
//! before the tick group's request frame went out, and every worker's start waits for that frame
//! ([crate::group]). The image path is [InklingVision]: the runtime encodes a prompt's fresh
//! images once through rank 0's tower, the lane keeps their rows, and each chunk injects the
//! rows its span covers.

use std::cell::RefCell;
use std::sync::Arc;

use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_model::vision::VisionImage;
use sparkflash_model::{Sampler, TpPlan, argmax};
use sparkflash_runtime::{
    ChunkPolicy, ComputeDeclaration, Draft, DraftBudget, Drafter, ExpertUnionWall, Limits,
    OutputConstraint, PartExec, PartHost, Pass, PassKind, PassMask, PlanBranch, PlanHost, PlanLane,
    PlanPart, PlanPass, PlanRows, PlanSeq, PlanWalk, PlanWindow, RankSplit, Region,
    Segment as RunSegment, Segments, StateKind, StepWall, Vision, Walk, WalkCapabilities,
};

use super::chunk::{ChunkRun, Injection, Rows, Segment, exchange_sum};
use super::{
    BATCH_ROWS, Chunk, ChunkWidths, DEVICE, DraftState, Drafter as DraftKind, InklingDeviceSession,
    JOINT_WALL, Opening, PrefillWalk, RowRun, Shared, open_buffer,
};
use crate::device_plan::{Collective, LayerSchedule, Op, Projection, ResidualInput, Target};
use crate::host::pick;

/// The drafts one verify may hold: one verify is the members' anchors and their drafts, and the
/// rank's verify batch holds [BATCH_ROWS] rows, so members x (1 + depth) never exceeds it.
const DRAFT_BUDGET: DraftBudget = DraftBudget::OnePass(BATCH_ROWS);

/// The Inkling-2-Small arm, exposed to the shared runtime as a Walk.
pub type InklingWalk = PlanWalk<InklingHost>;

/// One sequence as the runtime drives it: the rank's caches in the [InklingLane], the ids and
/// the request's sampler in the runtime's [PlanSeq].
pub type InklingSequence = PlanSeq<InklingLane>;

/// The walk over `shared`'s rank, serving sequences of the rank's context.
///
/// # Errors
/// A schedule whose layers the stages cannot split ([Stages::of]).
pub(super) fn walk(shared: &Arc<Shared>) -> Result<InklingWalk, BackendError> {
    // The host keeps its own buffers, so the walk's scratch is not sized by a row count.
    PlanWalk::new(Arc::new(InklingHost::new(shared)?), BATCH_ROWS)
}

/// A branch's frame: the layer, and whether it frames the routed experts or the attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    layer: usize,
    ffn: bool,
}

/// Which of a branch's three stages runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Enter,
    Part,
    Leave,
}

/// Where one layer's schedule splits into the planned walk's stages: each bound is the step a
/// stage starts at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stages {
    /// The attention's first projection: the block residual and the norms before it are the
    /// attention's entry.
    attention: usize,
    /// The attention delta's sum: the attention's exit runs from it.
    delta: usize,
    /// The MLP's block residual: the experts' entry runs from it.
    mlp: usize,
    /// The router's projection: the experts' part runs from it.
    router: usize,
    /// The expert delta's sum: the experts' exit runs from it to the layer's end.
    experts: usize,
    /// The layer's steps.
    steps: usize,
}

impl Stages {
    /// The bounds of `layer`'s schedule.
    ///
    /// # Errors
    /// A layer without one of the bounds, or with them out of order: the stages would not run
    /// the schedule's steps in its own order.
    fn of(layer: &LayerSchedule) -> Result<Self, String> {
        let find = |what: &str, test: &dyn Fn(Op, Target) -> bool| {
            layer
                .steps
                .iter()
                .position(|step| test(step.op, step.target))
                .ok_or_else(|| format!("layer {} has no {what} to stage at", layer.index))
        };
        let stages = Self {
            attention: find("projection", &|op, _| op == Op::Gemv)?,
            delta: find("attention delta", &|op, _| {
                op == Op::Collective(Collective::AttentionDelta)
            })?,
            mlp: find("MLP block residual", &|op, target| {
                op == Op::BlockResidual && target == Target::Input(ResidualInput::Mlp)
            })?,
            router: find("router projection", &|op, target| {
                op == Op::Gemv && target == Target::Project(Projection::Router)
            })?,
            experts: find("expert delta", &|op, _| {
                op == Op::Collective(Collective::ExpertDelta)
            })?,
            steps: layer.steps.len(),
        };
        let ordered = 0 < stages.attention
            && stages.attention < stages.delta
            && stages.delta < stages.mlp
            && stages.mlp <= stages.router
            && stages.router < stages.experts
            && stages.experts < stages.steps;
        if !ordered {
            return Err(format!(
                "layer {}'s stages are out of order: {stages:?}",
                layer.index
            ));
        }
        Ok(stages)
    }

    /// The steps `stage` of the attention branch, or of the experts' when `ffn`, runs.
    fn range(self, ffn: bool, stage: Stage) -> (usize, usize) {
        let bounds = if ffn {
            [self.mlp, self.router, self.experts, self.steps]
        } else {
            [0, self.attention, self.delta, self.mlp]
        };
        match stage {
            Stage::Enter => (bounds[0], bounds[1]),
            Stage::Part => (bounds[1], bounds[2]),
            Stage::Leave => (bounds[2], bounds[3]),
        }
    }
}

/// The parts' host: Inkling's schedules run every part's steps.
#[derive(Debug, Clone, Copy)]
pub struct InklingParts;

impl PartHost for InklingParts {
    type Pass<'a> = InklingPass<'a>;
    type Open<'a> = ();
    type Handle = PlanPart<InklingParts>;
}

/// One part's pass: the host and the window's open walk.
pub struct InklingPass<'a> {
    host: &'a InklingHost,
    scratch: &'a InklingScratch,
}

/// A layer's attention or routed experts: the part stage of its schedule.
struct Branch(Frame);

impl PartExec<InklingParts> for Branch {
    /// The host's regions state the lane ([PlanHost::regions]).
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut InklingPass<'_>) -> Result<(), BackendError> {
        pass.host.stage(pass.scratch, &self.0, Stage::Part)
    }

    // The schedule picks each kernel from the window's rows, a token pass's too.
    fn token(&self, pass: &mut InklingPass<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}

/// A window's walk of one of the arm's schedules.
enum Run {
    /// The chunk schedule over a prompt chunk, fused lanes or verify segments.
    Chunk(Box<ChunkRun>),
    /// The row schedule over one prompt position.
    Row(Box<RowRun>),
}

/// The window the walk is in, between its begin and its tail.
struct Open {
    run: Run,
    /// The pass row of the window's first row.
    first: usize,
}

/// One walk's buffers: the open window, the logits each answered pass row left for the head,
/// and the row buffers every sequence borrows.
#[derive(Default)]
pub struct InklingScratch {
    open: RefCell<Option<Open>>,
    left: RefCell<Vec<(usize, Vec<f32>)>>,
    /// The rows one lane step and one prompt chunk walk in, shared by every sequence: the engine
    /// steps one pass at a time, so a buffer a lane - about 0.7 MiB a row - is lent rather than
    /// held per lane.
    lanes: RefCell<Option<Chunk>>,
    /// The shared walk's buffer when drafting members verify together: as wide as the widest
    /// round.
    joint: RefCell<Option<Chunk>>,
}

/// The arm's rank and its schedules, staged: the planned walk's host.
pub struct InklingHost {
    shared: Arc<Shared>,
    /// Each layer's attention and routed experts, in walk order.
    branches: Vec<[PlanBranch<InklingParts, Frame>; 2]>,
    /// The rank's caches as the runtime sees them ([regions_of]).
    regions: Vec<Region>,
    limits: Limits,
    /// Each layer's stages in the row schedule.
    rows: Vec<Stages>,
    /// Each layer's stages in the chunk schedule, when the rank has one.
    chunks: Option<Vec<Stages>>,
}

impl InklingHost {
    /// The host over `shared`'s rank.
    ///
    /// # Errors
    /// A schedule whose layers the stages cannot split, or a chunk schedule of another depth.
    fn new(shared: &Arc<Shared>) -> Result<Self, BackendError> {
        let refuse = |error: String| BackendError::new(DEVICE, error);
        let stages = |plan: &super::InklingDevicePlan| {
            plan.layers()
                .iter()
                .map(Stages::of)
                .collect::<Result<Vec<_>, _>>()
                .map_err(refuse)
        };
        let rows = stages(&shared.plan)?;
        let chunks = shared.prefill.as_ref().map(stages).transpose()?;
        if let Some(chunks) = &chunks
            && chunks.len() != rows.len()
        {
            return Err(refuse(format!(
                "the chunk schedule has {} layers and the row schedule {}",
                chunks.len(),
                rows.len()
            )));
        }
        let branches = (0..rows.len())
            .map(|layer| {
                [false, true].map(|ffn| PlanBranch {
                    frame: Frame { layer, ffn },
                    part: Box::new(Branch(Frame { layer, ffn })) as PlanPart<InklingParts>,
                    // The exit sums every rank's partial in the schedule's own collective.
                    summed: true,
                    slot: 2 * layer + usize::from(ffn),
                })
            })
            .collect();
        // The pass's rows, not the buffers' width: the widest chunk the arm can walk is what the
        // rings are sized for, and the rows one prompt pass takes are the runtime's
        // ChunkPolicy over it - 4,096, 8,192 or 65,535 at the three documented widths, and the
        // largest documented size at or below anything lower. A tick engine reads its pass width
        // from here, so stating the policy's number is what keeps a prompt's chunks the ones the
        // runtime ran before the engine moved.
        let pass =
            ChunkPolicy::for_limits(Limits::new(shared.prefill_chunk, shared.context)).rows();
        Ok(Self {
            shared: Arc::clone(shared),
            branches,
            regions: regions_of(shared),
            limits: Limits::new(pass, shared.context).with_members(BATCH_ROWS),
            rows,
            chunks,
        })
    }

    /// The most drafts a round verifies (SPARKFLASH_INKLING_DRAFT_DEPTH).
    fn depth(&self) -> usize {
        self.shared.draft.map_or(0, |(_, depth)| depth)
    }

    /// Runs `stage` of `frame`'s branch on the open window: its steps of the layer's schedule.
    fn stage(
        &self,
        scratch: &InklingScratch,
        frame: &Frame,
        stage: Stage,
    ) -> Result<(), BackendError> {
        let mut open = scratch.open.borrow_mut();
        let open = open
            .as_mut()
            .ok_or_else(|| BackendError::new(DEVICE, "a stage ran outside a window"))?;
        let stages = match open.run {
            Run::Chunk(_) => self.chunks.as_deref(),
            Run::Row(_) => Some(self.rows.as_slice()),
        };
        let (from, to) = stages
            .and_then(|stages| stages.get(frame.layer))
            .ok_or_else(|| {
                BackendError::new(
                    DEVICE,
                    format!("the window's schedule has no layer {}", frame.layer),
                )
            })?
            .range(frame.ffn, stage);
        match &mut open.run {
            Run::Chunk(run) => run.steps(frame.layer, from, to),
            Run::Row(run) => run.steps(frame.layer, from, to),
        }
    }

    /// A prompt window's walk: the one-row step where the walk forwards a position at a time
    /// and the row is text, else the chunk schedule in the rank's prompt walk.
    fn prompt_run(
        &self,
        scratch: &InklingScratch,
        segments: &PlanRows<'_, '_, InklingLane>,
    ) -> Result<Run, BackendError> {
        let shared = &self.shared;
        let [(seq, ids)] = segments else {
            return Err(BackendError::new(
                DEVICE,
                "a prompt chunk holds one segment",
            ));
        };
        let session = &seq.lane.session;
        let take = ids.len();
        let row_bytes = shared.hidden() * std::mem::size_of::<f32>();
        // The chunk's own image rows: an image row is always a chunk's, never the row step's.
        let pieces = injections(session, take, row_bytes);
        let taps = session.drafter.as_ref().map(DraftState::sink);
        if take == 1 && shared.prefill_walk != PrefillWalk::Tensor && pieces.is_empty() {
            return RowRun::begin(shared, session.state.view(), ids[0], taps)
                .map(|run| Run::Row(Box::new(run)));
        }
        let walk = if shared.prefill_walk.chunked() {
            shared.prefill_walk
        } else {
            PrefillWalk::Chunk
        };
        let mut lanes = scratch.lanes.borrow_mut();
        open_buffer(shared, &mut lanes, take, 1)?;
        let chunk = lanes.as_ref().expect("the row buffer was just opened");
        let view = Box::new(session.state.view());
        ChunkRun::begin(shared, Rows::Chunk(view), chunk, ids, walk, &pieces, taps)
            .map(|run| Run::Chunk(Box::new(run)))
    }

    /// One token a lane ([Rows::Lanes]), the arm's fused token walk.
    fn lanes_run(
        &self,
        scratch: &InklingScratch,
        segments: &PlanRows<'_, '_, InklingLane>,
    ) -> Result<Run, BackendError> {
        let shared = &self.shared;
        let mut tokens = Vec::with_capacity(segments.len());
        for (_, ids) in segments {
            let &[token] = *ids else {
                return Err(BackendError::new(
                    DEVICE,
                    "a token pass holds one row a member",
                ));
            };
            tokens.push(token);
        }
        let mut lanes = scratch.lanes.borrow_mut();
        open_buffer(
            shared,
            &mut lanes,
            tokens.len(),
            tokens.len().next_power_of_two(),
        )?;
        let chunk = lanes.as_ref().expect("the lanes' buffer was just opened");
        let views = segments
            .iter()
            .map(|(seq, _)| seq.lane.session.state.view())
            .collect();
        ChunkRun::begin(
            shared,
            Rows::Lanes(views),
            chunk,
            &tokens,
            PrefillWalk::Chunk,
            &[],
            None,
        )
        .map(|run| Run::Chunk(Box::new(run)))
    }

    /// One walk of every segment's anchor and drafts ([Rows::Segments]): the body the drafter's
    /// verify runs.
    fn verify_run(
        &self,
        scratch: &InklingScratch,
        segments: &mut PlanRows<'_, '_, InklingLane>,
    ) -> Result<Run, BackendError> {
        let shared = &self.shared;
        let mut tokens = Vec::new();
        let mut shapes = Vec::with_capacity(segments.len());
        for (seq, ids) in segments.iter_mut() {
            let Some((&anchor, drafts)) = ids.split_first() else {
                return Err(BackendError::new(DEVICE, "a verify segment holds no rows"));
            };
            let session = &mut seq.lane.session;
            // A caller that walks rows of its own without the arm's round: the segment's rows are
            // the anchor and its drafts.
            let opening = session.opening.get_or_insert_with(|| Opening {
                start: session.state.position,
                anchor,
                drafts: drafts.to_vec(),
                confidence: Vec::new(),
            });
            let depth = drafts.len();
            if opening.anchor != anchor {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "a verify segment of {} rows (anchor {anchor}) was opened on anchor \
                         {} at depth {depth}",
                        ids.len(),
                        opening.anchor
                    ),
                ));
            }
            let tails = session
                .verify
                .as_ref()
                .map(|verify| verify.tails)
                .ok_or_else(|| BackendError::new(DEVICE, "a lane walks with no verify buffer"))?;
            tokens.extend_from_slice(ids);
            shapes.push((depth, tails, session.drafter.as_ref().map(DraftState::sink)));
        }
        let mut joint = scratch.joint.borrow_mut();
        if joint
            .as_ref()
            .is_some_and(|chunk| chunk.rows < tokens.len())
        {
            let old = joint.take().expect("the joint buffer was just checked");
            let pointers = old.buffers();
            shared.cuda.free_buffers(&[&pointers[0], &pointers[1]]);
        }
        if joint.is_none() {
            // As wide as every lane's deepest round, so it opens once.
            let widest = BATCH_ROWS * (self.depth() + 1);
            let width = widest.max(tokens.len());
            *joint = Some(
                Chunk::allocate(shared, width, width, true)
                    .map_err(|error| BackendError::new(DEVICE, error))?,
            );
        }
        let chunk = joint.as_ref().expect("the joint buffer was just opened");
        let walked = segments
            .iter()
            .zip(&shapes)
            .map(|((seq, _), &(depth, tails, taps))| Segment {
                state: seq.lane.session.state.view(),
                tails,
                rows: depth + 1,
                taps,
            })
            .collect();
        ChunkRun::begin(
            shared,
            Rows::Segments(walked),
            chunk,
            &tokens,
            PrefillWalk::Chunk,
            &[],
            None,
        )
        .map(|run| Run::Chunk(Box::new(run)))
    }
}

/// The prompt's encoded image rows `session` holds over the `rows` positions from its next one,
/// as chunk-relative runs: one a fresh image whose span overlaps them.
fn injections(session: &InklingDeviceSession, rows: usize, row_bytes: usize) -> Vec<Injection> {
    let Some(buffer) = session.images else {
        return Vec::new();
    };
    let first = session.state.position;
    let mut runs = Vec::new();
    let mut at = 0usize;
    for &(start, count) in &session.image_spans {
        let from = start.max(first);
        let to = (start + count).min(first + rows);
        if from < to {
            runs.push(Injection {
                row: from - first,
                source: buffer.offset((at + from - start) * row_bytes),
                count: to - from,
            });
        }
        at += count;
    }
    runs
}

/// A prompt window's `rows` rows landed on `session`: its position moves past them, and the
/// drafter's context takes their taps.
fn prompt_rows(session: &mut InklingDeviceSession, rows: usize) -> Result<(), BackendError> {
    let first = session.state.position;
    session.state.position += rows;
    session.inject(rows, first)
}

impl PlanHost for InklingHost {
    type Parts = InklingParts;
    type Frame = Frame;
    type Lane = InklingLane;
    /// The arm's DSpark or prompt-lookup round.
    type Drafter = InklingDrafter;
    /// Rank 0's vision tower.
    type Vision = InklingVision;
    type Scratch = InklingScratch;
    /// The schedules pick their kernels from the window they are in.
    type Mode = ();

    const DEVICE: &'static str = DEVICE;
    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        // A prompt runs in the arm's chunks, a step runs one row a lane, and a round verifies an
        // anchor and its drafts in one walk.
        passes: PassMask::TOKEN_ROWS
            .union(PassMask::BULK)
            .union(PassMask::VERIFY),
        segments: Segments::Many,
        state: StateKind::Both,
        // The four rows the arm states in sparkflash_model::ARM_LEDGERS; a test below holds
        // them equal so the walk and the table cannot drift.
        compute: ComputeDeclaration {
            prompt: State::Native("dense projections and NVFP4/MXFP8 experts on the grouped MMAs"),
            token: State::Off("row token GEMV; decode.tensor_cores=true is refused"),
            stored: State::Native("NVFP4 and MXFP8 experts, BF16 dense, in their stored formats"),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };
    /// Every answered row's logits come back to the host, a greedy one's too: the arm's head has
    /// always answered them, and its sampler chooses over them ([Self::choose]).
    const KEEPS_LOGITS: bool = true;

    fn layers(&self) -> usize {
        self.branches.len()
    }

    fn branches(&self, layer: usize) -> &[PlanBranch<InklingParts, Frame>] {
        &self.branches[layer]
    }

    fn vocab(&self) -> usize {
        self.shared.config.vocab_size
    }

    /// A walk takes any rows: the host's windows are the arm's chunks and steps.
    fn pass_rows(&self) -> usize {
        usize::MAX
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn regions(&self) -> Vec<Region> {
        self.regions.clone()
    }

    fn scratch(_this: &Arc<Self>, _rows: usize) -> Result<InklingScratch, BackendError> {
        Ok(InklingScratch::default())
    }

    fn lane(this: &Arc<Self>) -> Result<InklingLane, BackendError> {
        InklingDeviceSession::open(&this.shared).map(|session| InklingLane { session })
    }

    fn drafter(this: &Arc<Self>) -> Result<InklingDrafter, BackendError> {
        let shared = &this.shared;
        let config = &shared.config;
        Ok(InklingDrafter {
            present: shared.draft.is_some(),
            kind: shared.draft.map(|(kind, _)| kind),
            depth: this.depth(),
            wall: JOINT_WALL.with_experts(config.n_routed_experts, config.num_experts_per_tok),
        })
    }

    fn mode(&self, _pass: PlanPass, _rows: usize) {}

    /// A prompt pass goes in the arm's chunk widths ([chunk_splits]), or a position at a time
    /// when the rank's prompt walk is the row step; a token pass and a verify are one window.
    fn windows(
        &self,
        pass: PlanPass,
        segments: &PlanRows<'_, '_, InklingLane>,
    ) -> Result<Vec<usize>, BackendError> {
        match pass.kind {
            PassKind::Bulk { .. } => {
                let [(_, ids)] = segments else {
                    return Err(BackendError::new(
                        DEVICE,
                        "a prompt chunk holds one segment",
                    ));
                };
                // The row walk forwards a position at a time whatever the widths are, as the
                // arm's own prompt loop does; every chunked walk takes the widths after the
                // positions remaining.
                Ok(if self.shared.prefill_walk.chunked() {
                    chunk_splits(
                        self.shared.chunk_widths,
                        ids.len(),
                        pass.prompt_end.unwrap_or(0),
                    )
                } else {
                    vec![1; ids.len()]
                })
            }
            PassKind::TokenRows | PassKind::Verify { .. } => {
                Ok(vec![segments.iter().map(|(_, ids)| ids.len()).sum()])
            }
            PassKind::Paired { .. } => Err(BackendError::new(
                DEVICE,
                "the Inkling arm does not pair several prompts in one pass",
            )),
        }
    }

    fn begin_window(
        &self,
        scratch: &InklingScratch,
        segments: &mut PlanRows<'_, '_, InklingLane>,
        window: &PlanWindow,
    ) -> Result<(), BackendError> {
        self.shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        if window.number == 0 {
            scratch.left.borrow_mut().clear();
        }
        let run = match window.pass.kind {
            PassKind::Bulk { .. } => self.prompt_run(scratch, segments)?,
            PassKind::TokenRows => self.lanes_run(scratch, segments)?,
            PassKind::Verify { .. } => self.verify_run(scratch, segments)?,
            PassKind::Paired { .. } => {
                return Err(BackendError::new(
                    DEVICE,
                    "the Inkling arm does not pair several prompts in one pass",
                ));
            }
        };
        scratch.open.replace(Some(Open {
            run,
            first: window.first,
        }));
        Ok(())
    }

    fn abandon_window(&self, scratch: &InklingScratch) {
        scratch.open.take();
    }

    /// The epilogue and the answered rows' logits; each sequence's position moves past the rows
    /// it walked, and a prompt chunk's taps go to the drafter's context.
    fn finish(
        &self,
        scratch: &InklingScratch,
        segments: &mut PlanRows<'_, '_, InklingLane>,
        pass: PlanPass,
        _last: bool,
    ) -> Result<(), BackendError> {
        let open = scratch
            .open
            .take()
            .ok_or_else(|| BackendError::new(DEVICE, "a tail ran outside a window"))?;
        let mut left = scratch.left.borrow_mut();
        let first = open.first;
        let answered = match open.run {
            Run::Row(run) => vec![run.finish()?],
            Run::Chunk(run) => run.finish()?,
        };
        if let PassKind::Bulk { .. } = pass.kind {
            let [(seq, ids)] = segments else {
                return Err(BackendError::new(
                    DEVICE,
                    "a prompt chunk holds one segment",
                ));
            };
            let take = ids.len();
            prompt_rows(&mut seq.lane.session, take)?;
            let last = answered
                .into_iter()
                .next_back()
                .ok_or_else(|| BackendError::new(DEVICE, "a chunk answered no row"))?;
            left.push((first + take - 1, last));
            return Ok(());
        }
        // A token pass's row a lane, or a verify's anchor and drafts: every row answers.
        for (seq, ids) in segments.iter_mut() {
            seq.lane.session.state.position += ids.len();
        }
        left.extend(
            answered
                .into_iter()
                .enumerate()
                .map(|(row, logits)| (first + row, logits)),
        );
        Ok(())
    }

    fn ends(&self, token: u32) -> bool {
        self.shared.config.next_token(token).is_none()
    }

    /// The arm's own draw ([pick]): a sampled request's Gumbel noise is a function of the row's
    /// index in the sequence, so a verify's row and a plain step's agree.
    fn choose(&self, sampler: &mut Sampler, logits: &[f32], sampling: &Sampling) -> u32 {
        pick(sampler, logits, sampling)
    }

    /// The gathered head's released entries: the padded vocabulary trimmed to the unpadded one.
    fn logits_width(&self) -> usize {
        let padded = self.shared.plan.rank_vocab() * self.shared.exchange.world();
        self.shared.config.unpadded_vocab_size.min(padded).max(1)
    }

    fn rewind_slack(&self) -> usize {
        // The sliding cache keeps its window and a widest chunk of rows past the prompt's end;
        // the arm's old rewind point also reserved the drafts a verify may add.
        self.shared.prefill_chunk.saturating_sub(self.depth())
    }

    fn back(&self, lane: &mut InklingLane, reach: usize) -> Result<(), BackendError> {
        self.shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        lane.session
            .state
            .grow(&self.shared, reach)
            .map_err(|error| BackendError::new("context", error))
    }

    fn embed(&self, scratch: &InklingScratch, _ids: &[u32]) -> Result<(), BackendError> {
        let mut open = scratch.open.borrow_mut();
        match open.as_mut().map(|open| &mut open.run) {
            Some(Run::Chunk(run)) => run.embed(),
            Some(Run::Row(run)) => run.embed(),
            None => Err(BackendError::new(DEVICE, "an embed ran outside a window")),
        }
    }

    fn enter(
        &self,
        scratch: &InklingScratch,
        frame: &Frame,
        _rows: usize,
    ) -> Result<(), BackendError> {
        self.stage(scratch, frame, Stage::Enter)
    }

    fn pass<'a>(
        &'a self,
        scratch: &'a InklingScratch,
        _segments: &'a [(&'a mut InklingSequence, &'a [u32])],
        _layer: usize,
        _slot: usize,
        _rows: usize,
        _mode: (),
    ) -> InklingPass<'a> {
        InklingPass {
            host: self,
            scratch,
        }
    }

    /// The exit's own collective sums every rank's partial, so `summed` is the schedule's.
    fn leave(
        &self,
        scratch: &InklingScratch,
        frame: &Frame,
        _next: Option<&Frame>,
        _summed: bool,
        _rows: usize,
    ) -> Result<(), BackendError> {
        self.stage(scratch, frame, Stage::Leave)
    }

    /// The schedules keep no copy of a layer's residual for a trace.
    fn residual(&self, _scratch: &InklingScratch, _rows: usize) -> Result<Vec<f32>, BackendError> {
        Err(BackendError::new(
            DEVICE,
            "Inkling's walk does not trace its layers' residual streams",
        ))
    }

    fn logits(
        &self,
        scratch: &InklingScratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let left = scratch.left.borrow();
        let mut logits = Vec::with_capacity(count * self.logits_width());
        for row in first..first + count {
            let (_, values) = left.iter().find(|(at, _)| *at == row).ok_or_else(|| {
                BackendError::new(
                    DEVICE,
                    format!("row {row} of the last walk left no logits for a head"),
                )
            })?;
            logits.extend_from_slice(values);
        }
        Ok(logits)
    }

    /// Each row's argmax over the logits the walk read back, as the sampler's greedy choice
    /// takes it.
    fn choices(
        &self,
        scratch: &InklingScratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<u32>, BackendError> {
        (first..first + count)
            .map(|row| self.logits(scratch, row, 1).map(|logits| argmax(&logits)))
            .collect()
    }
}

/// One sequence's device state, as the runtime drives it: the rank's caches, histories, drafter
/// context and round. The runtime's [PlanSeq] holds the ids and the request's sampler.
pub struct InklingLane {
    session: InklingDeviceSession,
}

impl PlanLane for InklingLane {
    const DEVICE: &'static str = DEVICE;

    fn save(&mut self, at: usize) -> Result<(), BackendError> {
        self.session
            .save_boundary(at)
            .map_err(|error| BackendError::new(DEVICE, error))
    }

    fn restore(&mut self, to: usize) -> Result<(), BackendError> {
        if to == self.session.state.position {
            return Ok(());
        }
        self.session.round = None;
        self.session.opening = None;
        self.session.rewind_to(to)
    }

    fn clear(&mut self) -> Result<(), BackendError> {
        if self.session.state.position == 0 {
            return Ok(());
        }
        self.session.round = None;
        self.session.opening = None;
        self.session.forget()
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        self.session.state.room(positions, reset)
    }

    /// A request reports its own drafts, not the session's lifetime.
    fn requested(&mut self) {
        self.session.speculator.reset();
    }
}

/// The rows one prompt pass of `rows` positions walks, with `after` positions still to come.
///
/// This is [ChunkWidths::next]'s rule applied until the pass is consumed. It is a separate
/// function so a test can hold a runtime-chunked prompt to the boundaries the arm's own prompt
/// loop walks: the widths depend only on the positions remaining, so a pass the runtime cuts at
/// its own chunk size falls on the same boundaries.
fn chunk_splits(widths: ChunkWidths, rows: usize, after: usize) -> Vec<usize> {
    let mut splits = Vec::new();
    let mut walked = 0usize;
    while walked < rows {
        let left = after + rows - walked;
        let take = widths.next(left).min(rows - walked);
        splits.push(take);
        walked += take;
    }
    splits
}

/// The arm's state, as the runtime needs to see it.
///
/// Every layer's key and value rows grow with position; the four convolution histories a layer
/// and the drafter's Markov feature are one live value at a time, which is what the
/// prompt-boundary checkpoint copies.
fn regions_of(shared: &Shared) -> Vec<Region> {
    let layers = shared.plan.layers().len();
    let layer_row = shared.kv_width() * 2 * std::mem::size_of::<f32>();
    let histories = shared.conv_slot_bytes().iter().sum::<usize>() * layers;
    let feature = shared
        .dspark
        .as_ref()
        .map_or(0, |model| model.feature_bytes());
    regions_from(layer_row, layers, histories, feature)
}

/// [regions_of] from its numbers, so a test can hold it without a device.
fn regions_from(layer_row: usize, layers: usize, histories: usize, feature: usize) -> Vec<Region> {
    let mut regions = Vec::new();
    if layer_row > 0 && layers > 0 {
        regions.push(Region::Positional {
            name: "kv-cache",
            bytes_per_row: layer_row * layers,
        });
    }
    if histories + feature > 0 {
        regions.push(Region::Recurrent {
            name: "conv-histories-and-drafter",
            bytes: histories + feature,
        });
    }
    regions
}

/// The arm's draft source: prompt lookup or the DSpark block drafter, over the arm's own round.
pub struct InklingDrafter {
    /// Whether the rank has a drafter at all (SPARKFLASH_INKLING_DRAFT).
    present: bool,
    /// Which one it is, for the ledger.
    kind: Option<DraftKind>,
    /// The most drafts a round verifies (SPARKFLASH_INKLING_DRAFT_DEPTH).
    depth: usize,
    wall: ExpertUnionWall,
}

impl Drafter<InklingWalk> for InklingDrafter {
    const PRESENT: bool = true;

    /// The rank drafts only when SPARKFLASH_INKLING_DRAFT named one. Off, the runtime runs one
    /// plain token pass over every lane, the arm's old fused `Rows::Lanes` step, rather than a
    /// held decode a lane.
    fn drafts(&self) -> bool {
        self.present
    }

    /// The arm's round already folded each member's rows into its step: a verify answers
    /// [InklingDeviceSession::close]'s step.
    type Rows = StepOutput;

    fn wall(&self) -> &dyn StepWall {
        &self.wall
    }

    /// The step the last verify holds for `token`, or the last round settled and the next chain
    /// drafted: the ids keep the rows the settle kept, and a lone member verifies the depth the
    /// arm's own policy picked.
    fn open_round(&self, seq: &mut InklingSequence, token: u32) -> Result<Draft, BackendError> {
        if !self.present {
            // No drafter on this rank: the runtime steps the token through a token pass.
            return Err(BackendError::new(
                DEVICE,
                "this rank has no drafter: a round opens only where the arm drafts",
            ));
        }
        let session = &mut seq.lane.session;
        if session.held(token) {
            return session
                .decode(token, &mut seq.sampler, &seq.sampling)
                .map(Draft::Held);
        }
        session.settle()?;
        let position = session.state.position;
        seq.keep(position)?;
        // Prompt lookup drafts from the ids the sequence holds and the anchor.
        let lookup = if matches!(self.kind, Some(DraftKind::Lookup)) {
            let mut ids = Vec::with_capacity(position + 1);
            ids.extend_from_slice(seq.history());
            ids.push(token);
            ids
        } else {
            Vec::new()
        };
        let session = &mut seq.lane.session;
        let chain = session.open_round(token, &lookup, &seq.sampling)?;
        let opening = session
            .opening
            .as_ref()
            .ok_or_else(|| BackendError::new(DEVICE, "a round opened with no opening"))?;
        let room = session.shared.context.saturating_sub(opening.start + 1);
        // A lone member verifies the depth the arm's own policy picked, so the chain the walk
        // sees is the one the arm drafted for; a joint round shares it.
        let chosen = session
            .speculator
            .depth(&opening.confidence, room)
            .min(chain.len())
            .min(opening.drafts.len());
        let mut confidence = chain;
        confidence.truncate(chosen);
        Ok(Draft::Chain {
            confidence,
            drafts: opening.drafts[..chosen].to_vec(),
        })
    }

    /// One verify pass of the walk over every segment's anchor and drafts, each member's rows
    /// read back and closed into its round ([InklingDeviceSession::close]).
    fn verify(
        &self,
        walk: &InklingWalk,
        segments: &mut [RunSegment<'_, InklingSequence>],
        depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        for (index, segment) in segments.iter().enumerate() {
            let depth = segment.rows.len().saturating_sub(1);
            let given = depths.get(index).copied().unwrap_or(depth);
            if given != depth {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "a verify segment of {} rows was given depth {given}",
                        segment.rows.len()
                    ),
                ));
            }
        }
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        walk.walk(
            segments,
            Pass::verify(rows, OutputConstraint::new(&[], rows)),
        )?;
        let width = walk.host().logits_width();
        let mut first = 0;
        let mut steps = Vec::with_capacity(segments.len());
        for segment in segments.iter_mut() {
            let count = segment.rows.len();
            let logits = walk.logits(first, count)?;
            first += count;
            let rows = logits.chunks(width).map(<[f32]>::to_vec).collect();
            let seq = &mut *segment.seq;
            let step = seq
                .lane
                .session
                .close(count - 1, rows, &mut seq.sampler, &seq.sampling)?;
            steps.push(step);
        }
        Ok(steps)
    }

    fn settle(
        &self,
        _seq: &mut InklingSequence,
        _chunk: &[u32],
        rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        // [InklingDeviceSession::close] settled the round inside the verify.
        Ok(rows)
    }

    fn budget(&self, seqs: usize) -> usize {
        DRAFT_BUDGET.rows(seqs, self.depth)
    }

    fn ready(&self, seq: &InklingSequence, token: u32) -> bool {
        seq.lane.session.held(token)
    }

    /// The round's fed rows past `to` go back: the ids the caller was answered are the round's
    /// fed rows while it is open, else the position.
    fn give_back(&self, seq: &mut InklingSequence, to: usize) -> Result<(), BackendError> {
        let session = &mut seq.lane.session;
        let answered = session
            .round
            .as_ref()
            .map_or(session.state.position, |round| round.start + round.fed);
        session.give_back(answered.saturating_sub(to), &mut seq.sampler)
    }

    fn speculation(&self, seq: &InklingSequence) -> Speculation {
        seq.lane.session.speculator.stats()
    }

    fn describe(&self) -> String {
        let name = match self.kind {
            Some(DraftKind::Lookup) => "prompt lookup",
            Some(DraftKind::Dspark) => "DSpark",
            None => {
                return "SPARKFLASH_INKLING_DRAFT is off: the anchor alone, its round the \
                        plain step"
                    .to_owned();
            }
        };
        format!("{name}, up to {} drafts", self.depth)
    }
}

/// Inkling's image path: rank 0's vision tower, encoded once a prompt.
///
/// The runtime calls [Vision::encode] when a prompt starts over or carries fresh images after
/// the reused prefix, and [Vision::clear] when its chunks are done. Rank 0 alone runs the
/// tower; every rank's rows arrive through the group's sum, so a worker's buffer carries rank
/// 0's rows exactly as its prompt path did. The lane keeps the rows and their spans, which the
/// host's prompt windows inject.
pub struct InklingVision;

impl Vision<InklingWalk> for InklingVision {
    const PRESENT: bool = true;

    fn encode(
        walk: &InklingWalk,
        seq: &mut InklingSequence,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<(), BackendError> {
        let shared = &walk.host().shared;
        shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        shared
            .span_check()
            .check(tokens, images)
            .map_err(|error| BackendError::new("request", error))?;
        let session = &mut seq.lane.session;
        session.free_images();
        if images.is_empty() {
            return Ok(());
        }
        if shared.encodes() && shared.vision.is_none() {
            return Err(BackendError::new(
                "vision",
                format!(
                    "the request carries {} images and this rank holds no vision tower",
                    images.len()
                ),
            ));
        }
        // The images the reused prefix does not already hold: the arm encoded the others in
        // the prompt that wrote those positions.
        let start = session.state.position;
        let fresh: Vec<&VisionImage> = images
            .iter()
            .filter(|image| image.span.start >= start)
            .collect();
        if fresh.is_empty() {
            return Ok(());
        }
        let hidden = shared.hidden();
        let row_bytes = hidden * std::mem::size_of::<f32>();
        let rows: usize = fresh.iter().map(|image| image.span.grid.image_rows()).sum();
        let mut total = 0usize;
        let buffer = shared
            .cuda
            .alloc(rows * row_bytes, &mut total)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let fill = || -> Result<(), BackendError> {
            shared
                .cuda
                .memset_d8_async(buffer, 0, rows * row_bytes)
                .map_err(|error| BackendError::new(DEVICE, error))?;
            if shared.encodes() {
                let tower = shared
                    .vision
                    .as_ref()
                    .ok_or_else(|| BackendError::new(DEVICE, "rank 0 holds no vision tower"))?;
                let mut tower = tower.lock().map_err(|_| {
                    BackendError::new(DEVICE, "the vision tower's lock is poisoned")
                })?;
                let mut at = 0usize;
                for image in &fresh {
                    let grid = &image.span.grid;
                    let made = tower
                        .encode(
                            &shared.cuda,
                            &image.pixels,
                            grid.best_height,
                            grid.best_width,
                            buffer.offset(at * row_bytes),
                        )
                        .map_err(|error| BackendError::new(DEVICE, error))?;
                    if made != grid.image_rows() {
                        return Err(BackendError::new(
                            DEVICE,
                            format!(
                                "the tower made {made} rows of a span that holds {}",
                                grid.image_rows()
                            ),
                        ));
                    }
                    at += made;
                }
            }
            let piece = shared.prefill_chunk.max(1);
            let mut done = 0usize;
            while done < rows {
                let count = piece.min(rows - done);
                exchange_sum(shared, buffer.offset(done * row_bytes), count * hidden)?;
                done += count;
            }
            Ok(())
        };
        match fill() {
            Ok(()) => {
                session.images = Some(buffer);
                session.image_spans = fresh
                    .iter()
                    .map(|image| (image.span.start, image.span.grid.image_rows()))
                    .collect();
                Ok(())
            }
            Err(error) => {
                let _ = shared.cuda.synchronize();
                shared.cuda.free_buffers(&[&buffer]);
                Err(error)
            }
        }
    }

    fn clear(walk: &InklingWalk, seq: &mut InklingSequence) {
        if let Err(error) = walk.host().shared.cuda.make_current() {
            debug_assert!(false, "the vision path's clear lost its context: {error}");
            return;
        }
        seq.lane.session.free_images();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sparkflash_runtime::testing::ArmProfile;
    use sparkflash_runtime::{ArmFixture, ChunkPolicy, Fixture, FormatRule, StateLayout, Walk};

    use super::super::{
        DEFAULT_DRAFT_DEPTH, DEFAULT_PREFILL_CHUNK, LONG_PREFILL_CHUNK, LONG_PREFILL_FROM,
        MAX_PREFILL_CHUNK,
    };
    use super::*;

    /// The Inkling fixture at `world` ranks: a [`sparkflash_runtime::local_group`] whose ranks
    /// open the tiny host fixture's converted rank files, as served ranks open theirs, and
    /// exchange over the host TCP leg on loopback (the leg `SPARK_COMM_REDUCE=tcp` serves). The
    /// group is text only, as a worker's wire is. The TP check reads its tokens only, so the
    /// fixture states no byte counts.
    fn group_fixture(world: usize) -> Result<Fixture<super::InklingWalk>, BackendError> {
        use sparkflash_comm::tcp::TcpConfig;

        const CONTEXT: usize = 16;
        // The four-expert fixture, as at TP 1: every axis a four-rank plan shards divides.
        let config = crate::config::Inkling2Config::tiny_world();
        // A served rank opens its converted file, so each rank here does too.
        let directory =
            tempfile::tempdir().map_err(|error| BackendError::new("fixture", error.to_string()))?;
        let source = crate::fixture::tiny_source(&config);
        let mut files = Vec::with_capacity(world);
        for rank in 0..world {
            let layout = sparkflash_weights::tp::TpLayout {
                rank,
                world_size: world,
            };
            let written =
                crate::convert::convert_rank(&source, &config, layout, directory.path(), "tiny")
                    .map_err(|error| {
                        BackendError::new("fixture", format!("rank {rank}: {error}"))
                    })?;
            files.push(written.path);
        }
        let check = crate::vision::SpanCheck::new(config.vision.clone(), false);
        // Each rank binds `base_port + rank` on loopback; a port the system just handed out is
        // free, and the ones above it nearly always are.
        let base_port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .map_err(|error| BackendError::new("fixture", error.to_string()))?
            .port();
        let open = move |rank: usize| {
            let failed =
                |error: String| BackendError::new("fixture", format!("rank {rank}: {error}"));
            let tp =
                sparkflash_model::TpPlan::new(world, rank).map_err(|error| failed(error.detail))?;
            let mut tcp = TcpConfig::loopback4(rank, base_port);
            tcp.world_size = world;
            let layout = sparkflash_weights::tp::TpLayout {
                rank,
                world_size: world,
            };
            let file = crate::convert::RankFileSource::open(&files[rank], &config, layout)
                .map_err(failed)?;
            let exchange = crate::collective::TcpExchange::connect(&tcp).map_err(failed)?;
            crate::device::InklingDeviceBackend::open_rank_file(
                crate::descriptor(),
                config.clone(),
                &file,
                CONTEXT,
                tp,
                Box::new(exchange),
            )
            .map_err(failed)?
            .walk_arm()
        };
        let wire = move || -> Box<dyn sparkflash_runtime::GroupWire> {
            Box::new(crate::group::InklingWire::new(
                CONTEXT,
                check.clone(),
                false,
            ))
        };
        let (walk, group, workers) = sparkflash_runtime::local_group(world, open, wire)?;
        let keep: Box<dyn std::any::Any> = Box::new((workers, directory));
        Ok(Fixture::new(
            walk,
            format!("the Inkling-2 four-expert tiny synthetic checkpoint at TP {world}"),
            0,
            0,
            Vec::new(),
            Some(keep),
        )
        .in_group(group))
    }

    /// The arm as the record and the generated checks name it: a marker with the walk's own
    /// name, since the walk is the runtime's type. Its fixture is the four-expert tiny host
    /// fixture the device tests share, opened through the arm's own TP=1 constructor.
    pub struct InklingWalk;

    impl ArmFixture for InklingWalk {
        type Walk = super::InklingWalk;

        /// Inkling's single-process path serves on the tick engine, so its generated checks drive
        /// that engine exactly as dense's, GLM's and Qwen's do. Its TP leg skips because the
        /// fixture is one rank.
        fn tick_engine() -> bool {
            true
        }

        fn fixture(world: usize) -> Result<Fixture<super::InklingWalk>, BackendError> {
            if world > 1 {
                return group_fixture(world);
            }
            // The four-expert variant of the tiny release: every axis TP 2 and TP 4 shard
            // divides, so the TP check reads the same weights at every world.
            let config = crate::config::Inkling2Config::tiny_world();
            let source = Arc::new(crate::fixture::tiny_source(&config));
            let backend = crate::device::InklingDeviceBackend::open(
                crate::descriptor(),
                config.clone(),
                source,
                16,
            )
            .map_err(|error| BackendError::new("fixture", error))?;
            let walk = backend.walk_arm()?;
            // The shared tensor plan does not spell Inkling's block-attention residual, so the
            // fixture's floor is the weights it uploaded: every dense weight is read a token.
            let bytes = backend.weight_bytes() as u64;
            Ok(Fixture::new(
                walk,
                "the Inkling-2 four-expert tiny synthetic checkpoint",
                bytes,
                bytes,
                FormatRule::classes(&config.architecture.quantization),
                None,
            ))
        }
    }

    fn widths() -> ChunkWidths {
        ChunkWidths {
            base: DEFAULT_PREFILL_CHUNK,
            long: LONG_PREFILL_CHUNK,
            long_from: LONG_PREFILL_FROM,
        }
    }

    /// Every chunk width the arm can walk is a documented policy size, and the limits state the
    /// pass [InklingHost::new] gives the runtime with the rank's verify width for members.
    #[test]
    fn the_arms_three_chunk_widths_are_policy_sizes() {
        for size in [DEFAULT_PREFILL_CHUNK, LONG_PREFILL_CHUNK, MAX_PREFILL_CHUNK] {
            assert_eq!(
                ChunkPolicy::for_limits(Limits::new(size, 1_048_576)).rows(),
                size,
                "{size} rows must be a documented ChunkPolicy size"
            );
            // The walk's limits are this policy's rows, so a width stated as the widest chunk is
            // also the width one prompt pass runs.
            let limits = Limits::new(size, 1_048_576).with_members(BATCH_ROWS);
            assert_eq!(
                ChunkPolicy::for_limits(Limits::new(limits.rows, limits.context)).rows(),
                size
            );
        }
        let limits = Limits::new(LONG_PREFILL_CHUNK, 1_048_576).with_members(BATCH_ROWS);
        assert_eq!(limits.rows, LONG_PREFILL_CHUNK);
        assert_eq!(limits.members, BATCH_ROWS);
        assert_eq!(ChunkPolicy::for_limits(limits).rows(), LONG_PREFILL_CHUNK);
        // A context or a switch below a documented size is quantised to the one below it, which
        // is the number the runtime ran before the engine read the same limits: no chunk moves.
        assert_eq!(
            ChunkPolicy::for_limits(Limits::new(5_000, 8_192)).rows(),
            DEFAULT_PREFILL_CHUNK
        );
        assert_eq!(
            ChunkPolicy::for_limits(Limits::new(1_000, 1_000)).rows(),
            256
        );
    }

    /// A prompt the runtime cuts at its policy width falls on the same boundaries as the arm's
    /// own prompt loop: the widths depend on the positions remaining alone.
    #[test]
    fn the_runtime_chunks_do_not_move_the_arms_prompt_boundaries() {
        let widths = widths();
        for total in [
            1, 4_095, 4_096, 4_097, 8_192, 10_000, 16_383, 16_384, 40_000, 65_535, 100_000,
        ] {
            let want = chunk_splits(widths, total, 0);
            // The pass a chunked prompt runs is the policy over the widest width: 8,192 by
            // default, and the same number the tick engine's exact tick runs.
            let pass = ChunkPolicy::for_limits(Limits::new(LONG_PREFILL_CHUNK, 1_048_576)).rows();
            assert_eq!(pass, LONG_PREFILL_CHUNK);
            let mut got = Vec::new();
            let mut start = 0usize;
            while start < total {
                let rows = pass.min(total - start);
                let after = total - start - rows;
                got.extend(chunk_splits(widths, rows, after));
                start += rows;
            }
            assert_eq!(got, want, "{total} positions split differently");
        }
        // The released walk's own examples: a long prompt widens until its tail.
        assert_eq!(chunk_splits(widths, 16_384, 0), [8_192, 4_096, 4_096]);
        assert_eq!(chunk_splits(widths, 10_000, 0), [4_096, 4_096, 1_808]);
    }

    /// The drafter's budget is the rule the profile states, at the drafter's own depth, and an
    /// off drafter says it drafts nothing.
    #[test]
    fn the_drafter_budgets_by_the_stated_rule() {
        let drafter = InklingDrafter {
            present: true,
            kind: Some(DraftKind::Dspark),
            depth: 7,
            wall: JOINT_WALL,
        };
        for seqs in 1..=BATCH_ROWS {
            assert_eq!(drafter.budget(seqs), DRAFT_BUDGET.rows(seqs, 7));
        }
        assert_eq!(drafter.budget(4), BATCH_ROWS - 4);
        assert_eq!(drafter.budget(8), 0);
        assert_eq!(drafter.describe(), "DSpark, up to 7 drafts");
        assert!(drafter.drafts());
        let off = InklingDrafter {
            present: false,
            kind: None,
            depth: 0,
            wall: JOINT_WALL,
        };
        assert!(
            !off.drafts(),
            "an off drafter must take the plain lanes step"
        );
    }

    /// The layout is both kinds: the key and value rows are positional, the convolution
    /// histories and the drafter's feature need the prompt-boundary checkpoint.
    #[test]
    fn the_layout_is_positional_caches_and_recurrent_histories() {
        let layout = StateLayout::new(regions_from(64, 48, 1_024, 256));
        assert_eq!(layout.kind(), StateKind::Both);
        assert!(layout.needs_checkpoint());
        assert_eq!(layout.bytes_per_row(), 64 * 48);
        assert_eq!(layout.bytes_at(2), 64 * 48 * 2 + 1_280);
        assert_eq!(
            StateLayout::new(regions_from(0, 0, 0, 0)).kind(),
            StateKind::None,
            "a rank with no state keeps nothing"
        );
    }

    /// The walk and its sequences are what a fused engine thread takes.
    #[test]
    #[allow(clippy::assertions_on_constants)] // the claim is the point: it is a type constant
    fn the_walk_and_its_sequences_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::InklingWalk>();
        assert_send_sync::<InklingSequence>();
        assert_eq!(
            <super::InklingWalk as Walk>::CAPABILITIES.segments,
            Segments::Many
        );
        assert!(
            <<super::InklingWalk as Walk>::Vision as Vision<super::InklingWalk>>::PRESENT,
            "the arm serves images through its tower"
        );
    }

    /// What the arm states for the generated profile and round tests: the pass rows a
    /// [ChunkPolicy] over a widest prompt chunk gives, as [InklingHost::new] states them, with
    /// [BATCH_ROWS] as the member cap; the default draft depth under the arm's budget and wall;
    /// the key rows and the histories as its state; and the chunks a prompt past one width runs
    /// in, at the default schedule's widest chunk and at the 65,535-row switch.
    fn stated() -> ArmProfile {
        let context = 1_048_576;
        let limits = |widest: usize| {
            let pass = ChunkPolicy::for_limits(Limits::new(widest, context)).rows();
            Limits::new(pass, context).with_members(BATCH_ROWS)
        };
        ArmProfile::new(
            "Inkling",
            StateLayout::new(regions_from(64, 48, 1_024, 256)),
            limits(LONG_PREFILL_CHUNK),
        )
        .with_code(DEVICE)
        .drafting(DEFAULT_DRAFT_DEPTH, DRAFT_BUDGET)
        .with_wall(JOINT_WALL)
        .prompt(limits(LONG_PREFILL_CHUNK), 20_000, &[8_192, 8_192, 3_616])
        .prompt(limits(MAX_PREFILL_CHUNK), 100_000, &[65_535, 34_465])
    }

    sparkflash_runtime::conformance!(super::InklingWalk, profile: stated(), rounds);
}
