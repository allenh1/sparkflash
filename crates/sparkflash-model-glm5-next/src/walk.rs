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

//! GLM-5.3-Flash as a planned walk: [GlmWalk] is the runtime's [PlanWalk] over [Shared].
//!
//! The runtime owns the layer loop, the sequence's ids and sampler, the head's
//! record-then-choose, reuse, rewind and the lane pool; this host supplies only what GLM's rank
//! alone knows, a stage of [DeviceRank]'s step at a time:
//!
//! * A layer is two branches, its token mixer (a KDA recurrence, or a DSA attention with its
//!   sparse indexer folded into the attention's part, as the rank runs it) and its
//!   feed-forward. The manifold hyper-connection is the frames around them: [PlanHost::enter] is
//!   the streams' mix in and the norm ([DeviceRank::step_enter]), [PlanHost::leave] the prefetch
//!   of the next reads, the partials' sum and the mix out ([DeviceRank::step_leave]), which may
//!   work out the next entry's projection.
//! * A pass's windows are the rank's: token and verify rows [MAX_ROWS] a window; a prompt's rows
//!   the token path's windows (a short prompt, or a rank or sequence the bulk chunks do not
//!   take: [DeviceRank::takes_chunks]) or bulk chunks on the rank's chunk scratch, whose block
//!   drafter commits its window rows as each chunk ends ([DeviceRank::chunk_plan]).
//! * The tail ([PlanHost::finish]) is the step's ([DeviceRank::step_tail]): the fold, the norm,
//!   the LM head and the MTP layer's caches; then the sequence's position, its pending verify and
//!   its block drafter's commits ([DeviceRank::after_run], [DeviceRank::after_chunk]). The head
//!   reads back the logits the tail left, on the host or on the device.
//! * The layout is what the rank's caches are, from the rank's own plan ([DeviceRank::kv_plan]),
//!   and the limits the rank's bulk chunk width, the served context and its verify width as the
//!   member cap.
//! * The drafter is [GlmDrafter], the arm's one round: the MTP layer's or the DFlash 2 block
//!   drafter's chain ([DeviceRank::draft_chain]), a verify pass of this walk, and the settlement
//!   ([DeviceRank::commit]).

use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use sparkflash_model::backend::{BackendError, Speculation, StepOutput};
use sparkflash_model::compose::StepRole;
use sparkflash_model::ledger::State;
use sparkflash_model::parts::Part;
use sparkflash_model::{DepthPolicy, Queue, Speculator, TpPlan, argmax, kept_to_end, verdict};
use sparkflash_runtime::{
    ComputeDeclaration, Draft, DraftBudget, Drafter, ExpertUnionWall, Limits, NoVision,
    OutputConstraint, PartExec, PartHost, Pass, PassKind, PassMask, PlanBranch, PlanHost, PlanLane,
    PlanPart, PlanPass, PlanRows, PlanSeq, PlanWalk, PlanWindow, RankSplit, Region, Segment,
    Segments, StateKind, StepWall, Walk, WalkCapabilities, plan_layer_folding,
};
use sparkflash_weights::tp::{BitGatherHandle, TpRuntime};

use crate::device::{
    ChunkPlan, DeviceRank, DeviceSequence, HeadRows, LogitsAt, MAX_ROWS, Run, StepSet,
};

/// Every refusal's code.
const DEVICE: &str = "glm5_device";

/// The GLM-5.3-Flash arm, exposed to the shared runtime as a Walk.
pub type GlmWalk = PlanWalk<Shared>;

/// The walk over `shared`'s rank.
///
/// # Errors
/// A poisoned rank lock.
pub(crate) fn walk(shared: &Arc<Shared>) -> Result<GlmWalk, BackendError> {
    // The host keeps its own head rows, so the walk's scratch is not sized by a row count.
    PlanWalk::new(Arc::clone(shared), MAX_ROWS)
}

/// How a session drafts, once the backend has been told to.
#[derive(Debug, Clone)]
pub(crate) struct SpecConfig {
    /// The most drafts a step verifies.
    pub(crate) drafts: usize,
    /// The depth policy, when the drafter says how sure it is of each draft (the block drafter).
    pub(crate) policy: Option<DepthPolicy>,
}

/// A branch's frame: the layer, and whether it frames the feed-forward or the token mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    layer: usize,
    ffn: bool,
}

/// The backend's rank and the collectives every walk runs through: the planned walk's host.
pub struct Shared {
    pub(crate) rank: Mutex<DeviceRank>,
    pub(crate) tp: TpRuntime,
    pub(crate) gather: BitGatherHandle,
    pub(crate) eos: Vec<u32>,
    /// Set once, before the first sequence opens: a sequence's sequence is a drafting one or it
    /// is not, so the choice cannot change under a live one.
    pub(crate) spec: OnceLock<SpecConfig>,
    /// The positions a sequence holds.
    context: usize,
    vocab: usize,
    limits: Limits,
    /// The rank's caches as the runtime sees them ([regions_of]).
    regions: Vec<Region>,
    /// Each layer's token mixer and feed-forward, in walk order.
    branches: Vec<[PlanBranch<GlmParts, Frame>; 2]>,
}

impl Shared {
    /// The host over `rank`, its collectives, the reply's end tokens and `context` positions a
    /// sequence.
    ///
    /// # Errors
    /// A rank whose cache plan cannot be read, or a checkpoint whose layers the rank's do not
    /// match or that names a step this walk does not run.
    pub(crate) fn new(
        rank: DeviceRank,
        tp: TpRuntime,
        gather: BitGatherHandle,
        eos: Vec<u32>,
        context: usize,
    ) -> Result<Self, BackendError> {
        let config = rank.config();
        let plan = rank.kv_plan().map_err(|error| {
            BackendError::new(DEVICE, format!("the rank's cache plan: {error}"))
        })?;
        let steps = config.architecture.steps();
        let built = rank.layer_parts();
        if steps.len() != built.len() {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "the checkpoint has {} layers and the rank built {}",
                    steps.len(),
                    built.len()
                ),
            ));
        }
        let mut branches = Vec::with_capacity(steps.len());
        for (step, (mixer, ffn)) in steps.iter().zip(built) {
            // The sparse indexer is folded into the DSA attention's part: the rank runs the
            // selection inside the attention step, so it is no branch of its own.
            let layer =
                plan_layer_folding(step, <Self as PlanHost>::ROLES, <Self as PlanHost>::FOLDED)
                    .map_err(|error| BackendError::new(DEVICE, error))?;
            let planned: Vec<Part> = layer.branches.iter().map(|&(_, part)| part).collect();
            if planned != [mixer, ffn] || layer.residual.is_none() {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "layer {} is {planned:?} with residual {:?}; the rank built {mixer:?} \
                         and {ffn:?} inside hyper-connection frames",
                        step.index, layer.residual
                    ),
                ));
            }
            let index = branches.len();
            let branch = |ffn: bool| PlanBranch {
                frame: Frame { layer: index, ffn },
                part: if ffn {
                    Box::new(FeedForward(index)) as PlanPart<GlmParts>
                } else {
                    Box::new(Mixer(index))
                },
                // The exit sums every rank's partial inside its mix out.
                summed: true,
                slot: 2 * index + usize::from(ffn),
            };
            branches.push([branch(false), branch(true)]);
        }
        let vocab = config.vocab_size;
        // The runtime's ChunkPolicy quantises the chunk, so a profile's 4096 stays 4096 and the
        // rank's scratch can always hold the chunk it is handed. A verify is one rank step, so
        // the member cap is the rank's verify width, not the prompt chunk.
        let limits = walk_limits(rank.prefill_rows(), context);
        Ok(Self {
            rank: Mutex::new(rank),
            tp,
            gather,
            eos,
            spec: OnceLock::new(),
            context,
            vocab,
            limits,
            regions: regions_of(plan),
            branches,
        })
    }

    /// The rank, locked for one call.
    fn rank(&self) -> Result<MutexGuard<'_, DeviceRank>, BackendError> {
        self.rank
            .lock()
            .map_err(|_| BackendError::new(DEVICE, "the rank lock is poisoned"))
    }
}

/// A shared walk's cost at TP=4 on four GB10-class nodes, from the device harness: a plain step
/// 31.6 ms, about 0.75 ms a distinct expert, and the block drafter's 4.6-5 ms a stream for each
/// member past the first. The expert counts are the checkpoint's.
const JOINT_WALL: ExpertUnionWall = ExpertUnionWall {
    fixed_ms: 25.0,
    expert_ms: 0.75,
    row_ms: 0.3,
    seq_ms: 5.0,
    experts: 0,
    top_k: 0,
};

/// The arm's state, as the runtime needs to see it: the sparse layers' latent and indexer rows
/// grow a position at a time, and the linear layers' convolution history and recurrent memory
/// are one live value a layer.
fn regions_of(plan: sparkflash_model::KvPlan) -> Vec<Region> {
    let mut regions = Vec::new();
    if plan.per_token() > 0 {
        regions.push(Region::Positional {
            name: "dlm-latent-and-indexer",
            bytes_per_row: usize::try_from(plan.per_token()).unwrap_or(usize::MAX),
        });
    }
    if plan.fixed() > 0 {
        regions.push(Region::Recurrent {
            name: "kda-conv-and-recurrent",
            bytes: usize::try_from(plan.fixed()).unwrap_or(usize::MAX),
        });
    }
    regions
}

/// The drafts one verify may hold: the anchors take one row a member, and the rest of
/// [MAX_ROWS] is shared among the members' chains. This is the [Drafter::budget] the row share
/// reads, and it keeps a verify to one rank step whatever depth the drafter proposes.
const DRAFT_BUDGET: DraftBudget = DraftBudget::OnePass(MAX_ROWS);

/// The walk's [Limits]: the rank's prompt chunk, the served context, and the rank's verify
/// width as the member cap. A drafting walk's batch is a verify, not a prompt chunk.
pub(crate) fn walk_limits(chunk: usize, context: usize) -> Limits {
    Limits::new(chunk, context).with_members(MAX_ROWS)
}

/// One sequence's device state, as the runtime drives it: the rank's cache and, for a drafting
/// sequence, its depth and accounting and the answers a verify already produced, because a
/// verify produces more than the one token a step answers with. The runtime's [PlanSeq] holds
/// the ids and the sampler.
pub struct GlmLane {
    shared: Arc<Shared>,
    sequence: DeviceSequence,
    /// The draft depth and its accounting, when this sequence drafts.
    spec: Option<Speculator>,
    /// Tokens a verify already produced, oldest first, each with the token that follows it.
    pending: Queue,
    /// The drafter's confidence in each of the last round's drafts.
    confidence: Vec<f32>,
    /// Whether a prompt is running a bulk chunk at a time.
    prefill_open: bool,
}

impl GlmLane {
    /// Whether the lane already holds the answer for a step fed `token`.
    #[must_use]
    pub fn holds(&self, token: u32) -> bool {
        self.pending
            .front()
            .is_some_and(|front| front.token == token)
    }

    /// The queued answers, the confidence and the open prompt go: the state moved under them.
    fn forget(&mut self) {
        self.pending.clear();
        self.confidence.clear();
        self.prefill_open = false;
    }
}

impl PlanLane for GlmLane {
    const DEVICE: &'static str = DEVICE;

    fn save(&mut self, _at: usize) -> Result<(), BackendError> {
        let rank = self.shared.rank()?;
        rank.keep_point(&mut self.sequence)
    }

    fn restore(&mut self, to: usize) -> Result<(), BackendError> {
        let rank = self.shared.rank()?;
        rank.rewind(&mut self.sequence, to)?;
        drop(rank);
        self.forget();
        Ok(())
    }

    fn clear(&mut self) -> Result<(), BackendError> {
        let rank = self.shared.rank()?;
        rank.reset(&mut self.sequence)?;
        drop(rank);
        self.forget();
        Ok(())
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        // The runtime asks about the history's positions; the state is at least there, and a
        // drafting step also writes its chain's rows past it.
        let drafts = self.spec.as_ref().map_or(0, Speculator::max_depth);
        let wanted = positions.max(self.sequence.position() + 1 + drafts);
        self.sequence.room(wanted, reset)
    }

    /// A request reports its own speculation, not the session's lifetime, and a kept sequence's
    /// queued answers and confidence were the last request's.
    fn requested(&mut self) {
        if let Some(spec) = self.spec.as_mut() {
            spec.reset();
        }
        self.confidence.clear();
        self.pending.clear();
    }
}

/// The parts' host: GLM's rank runs every part's kernels.
#[derive(Debug, Clone, Copy)]
pub struct GlmParts;

impl PartHost for GlmParts {
    type Pass<'a> = GlmPass<'a>;
    type Open<'a> = ();
    type Handle = PlanPart<GlmParts>;
}

/// One part's pass: the rank, locked for the part, the window's runs and its rows.
pub struct GlmPass<'a> {
    rank: Option<MutexGuard<'a, DeviceRank>>,
    runs: Vec<Run<'a>>,
    rows: usize,
}

impl GlmPass<'_> {
    fn rank(&self) -> Result<&DeviceRank, BackendError> {
        self.rank
            .as_deref()
            .ok_or_else(|| BackendError::new(DEVICE, "the rank lock is poisoned"))
    }
}

/// A layer's token mixer: its KDA recurrence, or its DSA attention with the indexer's selection
/// ([DeviceRank::step_mixer]).
struct Mixer(usize);

impl PartExec<GlmParts> for Mixer {
    /// The rank's cache plan states the lane ([Shared::regions]).
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut GlmPass<'_>) -> Result<(), BackendError> {
        pass.rank()?.step_mixer(self.0, &pass.runs, pass.rows)
    }

    // The rank picks each kernel from the rows and the runs, a token pass's too.
    fn token(&self, pass: &mut GlmPass<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}

/// A layer's feed-forward: its dense MLP or its routed experts ([DeviceRank::step_ffn]).
struct FeedForward(usize);

impl PartExec<GlmParts> for FeedForward {
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut GlmPass<'_>) -> Result<(), BackendError> {
        pass.rank()?.step_ffn(self.0, pass.rows)
    }

    // The rank picks each kernel from the rows, a token pass's too.
    fn token(&self, pass: &mut GlmPass<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}

/// Which path a window runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// Token rows, [MAX_ROWS] a step: a verify's when `verify`, whose rows go to the host, else
    /// a greedy step's, whose logits stay on the device.
    Token { verify: bool },
    /// A prompt's rows on the token path, every row's logits to the host.
    Whole,
    /// A bulk chunk on the chunk scratch, its last row's logits to the host.
    Chunk,
}

/// The window the walk is in, between its begin and its tail.
struct Open {
    path: Path,
    set: StepSet,
    /// The block drafter's reads of a bulk chunk.
    chunk: Option<ChunkPlan>,
    /// The first segment's position.
    position: usize,
    /// The pass row of the window's first row.
    first: usize,
    /// Each segment's rows of the pass after the window.
    after: Vec<usize>,
    /// Whether the chunk scratch is swapped in.
    swapped: bool,
    /// Whether the last exit worked out the next entry's projection.
    fused: bool,
    started: Instant,
}

/// Where a head row's logits are.
enum Left {
    Host(Vec<f32>),
    Device(LogitsAt),
}

/// One walk's buffers: the rank keeps its own, so this is the open window and what the head
/// reads back, by pass row.
pub struct Scratch {
    open: RefCell<Option<Open>>,
    left: RefCell<Vec<(usize, Left)>>,
}

impl Shared {
    /// A window's runs over `segments`: a bulk chunk's one run with `chunk`'s captures, else
    /// one token run a segment.
    fn runs<'a>(
        path: Path,
        chunk: Option<&ChunkPlan>,
        segments: &'a [(&'a mut PlanSeq<GlmLane>, &'a [u32])],
    ) -> Vec<Run<'a>> {
        let mut first = 0;
        segments
            .iter()
            .map(|(seq, ids)| {
                let sequence = &seq.lane.sequence;
                let run = match (path, chunk) {
                    (Path::Chunk, Some(plan)) => Run::chunk(sequence, ids.len(), plan),
                    (path, _) => Run::token(
                        sequence,
                        first,
                        ids.len(),
                        path == Path::Token { verify: true },
                    ),
                };
                first += ids.len();
                run
            })
            .collect()
    }

    /// Runs `with` on the open window.
    fn open<T>(
        scratch: &Scratch,
        with: impl FnOnce(&mut Open) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        let mut open = scratch.open.borrow_mut();
        let open = open
            .as_mut()
            .ok_or_else(|| BackendError::new(DEVICE, "a stage ran outside a window"))?;
        with(open)
    }

    /// The path a window of the prompt rows `rows` on `lane` takes: the chunks once a prompt
    /// opened them or when the rank takes them for a prompt of that many rows.
    fn prompt_path(rank: &DeviceRank, lane: &GlmLane, rows: usize) -> Path {
        if lane.prefill_open || rank.takes_chunks(&lane.sequence, rows) {
            Path::Chunk
        } else {
            Path::Whole
        }
    }
}

impl PlanHost for Shared {
    type Parts = GlmParts;
    type Frame = Frame;
    type Lane = GlmLane;
    /// The checkpoint's MTP layer or the loaded DFlash 2 block drafter, as the runtime's round.
    type Drafter = GlmDrafter;
    /// The arm reads text alone.
    type Vision = NoVision;
    type Scratch = Scratch;
    /// The rank picks its kernels from the window it is in.
    type Mode = ();

    const DEVICE: &'static str = DEVICE;
    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        // A prompt runs in bulk chunks and a step runs token rows; a verify pass is the round
        // the drafter runs, for a runtime caller that has rows to check.
        passes: PassMask::TOKEN_ROWS
            .union(PassMask::BULK)
            .union(PassMask::VERIFY),
        segments: Segments::Many,
        state: StateKind::Both,
        // The arm's own four rows, word for word the declaration in
        // `sparkflash_model::ARM_LEDGERS`; a test below holds them equal. The token row is the
        // same constant the CLI's GLM profile check reads.
        compute: ComputeDeclaration {
            prompt: State::Native("bulk GEMMs on the FP8 E4M3 MMAs; BF16 as head and remainder"),
            token: crate::TOKEN_DECODE,
            stored: State::Native(
                "FP8 E4M3 projections and experts run E4M3 x E4M3; BF16 stays BF16",
            ),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };
    /// The manifold hyper-connection is the frames.
    const ROLES: &'static [StepRole] = &[StepRole::Residual];
    /// The sparse indexer runs inside the DSA attention's part.
    const FOLDED: &'static [StepRole] = &[StepRole::Indexer];

    fn layers(&self) -> usize {
        self.branches.len()
    }

    fn branches(&self, layer: usize) -> &[PlanBranch<GlmParts, Frame>] {
        &self.branches[layer]
    }

    fn vocab(&self) -> usize {
        self.vocab
    }

    /// A walk takes any rows: the host's windows are the rank's steps.
    fn pass_rows(&self) -> usize {
        usize::MAX
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn regions(&self) -> Vec<Region> {
        self.regions.clone()
    }

    fn scratch(_this: &Arc<Self>, _rows: usize) -> Result<Scratch, BackendError> {
        Ok(Scratch {
            open: RefCell::new(None),
            left: RefCell::new(Vec::new()),
        })
    }

    fn lane(this: &Arc<Self>) -> Result<GlmLane, BackendError> {
        let rank = this.rank()?;
        // A sequence is a drafting one or it is not for its whole life: the rank's drafter is
        // chosen here, before the first prompt.
        let sequence = match this.spec.get() {
            Some(_) if rank.has_block_drafter() => {
                rank.new_block_drafting_sequence(this.context)?
            }
            Some(_) => rank.new_drafting_sequence(this.context)?,
            None => rank.new_sequence(this.context)?,
        };
        drop(rank);
        let spec = this
            .spec
            .get()
            .map(|spec| Speculator::new(spec.drafts, spec.policy.clone()));
        Ok(GlmLane {
            shared: Arc::clone(this),
            sequence,
            spec,
            pending: Queue::new(),
            confidence: Vec::new(),
            prefill_open: false,
        })
    }

    fn drafter(this: &Arc<Self>) -> Result<GlmDrafter, BackendError> {
        let rank = this.rank()?;
        let config = rank.config();
        let wall = JOINT_WALL.with_experts(config.n_routed_experts, config.num_experts_per_tok);
        let block = rank.has_block_drafter();
        drop(rank);
        let spec = this.spec.get();
        Ok(GlmDrafter {
            shared: Arc::clone(this),
            depth: spec.map_or(0, |spec| spec.drafts),
            wall,
            kind: if block { "DFlash 2" } else { "MTP" },
            adaptive: spec.is_some_and(|spec| spec.policy.is_some()),
        })
    }

    fn mode(&self, _pass: PlanPass, _rows: usize) {}

    /// Token and verify rows go [MAX_ROWS] a step, one segment after another; a verify is one
    /// step. A prompt's rows go a segment at a time, in the token path's steps or the rank's
    /// bulk chunks ([Shared::prompt_path]).
    fn windows(
        &self,
        pass: PlanPass,
        segments: &PlanRows<'_, '_, GlmLane>,
    ) -> Result<Vec<usize>, BackendError> {
        let split = |rows: usize, width: usize, into: &mut Vec<usize>| {
            into.extend((0..rows.div_ceil(width)).map(|at| width.min(rows - at * width)));
        };
        let mut windows = Vec::new();
        match pass.kind {
            PassKind::TokenRows | PassKind::Verify { .. } => {
                let rows: usize = segments.iter().map(|(_, ids)| ids.len()).sum();
                if matches!(pass.kind, PassKind::Verify { .. }) && rows > MAX_ROWS {
                    return Err(BackendError::new(
                        DEVICE,
                        format!(
                            "a verify of {rows} rows; a verify is one step of at most {MAX_ROWS}"
                        ),
                    ));
                }
                split(rows, MAX_ROWS, &mut windows);
            }
            PassKind::Bulk { .. } | PassKind::Paired { .. } => {
                let rank = self.rank()?;
                for (seq, ids) in segments {
                    let width = match Self::prompt_path(&rank, &seq.lane, ids.len()) {
                        Path::Chunk => rank.prefill_rows(),
                        _ => MAX_ROWS,
                    };
                    split(ids.len(), width, &mut windows);
                }
            }
        }
        Ok(windows)
    }

    fn begin_window(
        &self,
        scratch: &Scratch,
        segments: &mut PlanRows<'_, '_, GlmLane>,
        window: &PlanWindow,
    ) -> Result<(), BackendError> {
        if window.number == 0 {
            scratch.left.borrow_mut().clear();
        }
        let mut rank = self.rank()?;
        rank.make_current()?;
        let device_leg = rank.collective(&self.tp, &self.gather)?;
        let path = match window.pass.kind {
            PassKind::TokenRows => Path::Token { verify: false },
            PassKind::Verify { .. } => Path::Token { verify: true },
            PassKind::Bulk { .. } | PassKind::Paired { .. } => {
                let [(seq, ids)] = segments else {
                    return Err(BackendError::new(
                        DEVICE,
                        "a prompt window holds one sequence's rows",
                    ));
                };
                let lane = &mut seq.lane;
                Self::prompt_path(&rank, lane, ids.len() + window.after[0])
            }
        };
        let mut chunk = None;
        match path {
            Path::Token { verify } => {
                for (seq, ids) in segments.iter_mut() {
                    rank.admit(&seq.lane.sequence, ids)?;
                    if !verify {
                        // A step ends any prompt that was running a chunk at a time.
                        seq.lane.prefill_open = false;
                    }
                }
            }
            Path::Whole | Path::Chunk => {
                let (seq, ids) = &mut segments[0];
                let lane = &mut seq.lane;
                let opened = lane.prefill_open;
                if !opened {
                    // A new prompt's tokens replace any chain the last one left queued.
                    lane.pending.clear();
                    lane.confidence.clear();
                }
                rank.admit(&lane.sequence, ids)?;
                if path == Path::Chunk {
                    // The block drafter's window commits read where the whole prompt ends.
                    let end = lane.sequence.position()
                        + ids.len()
                        + window.after[0]
                        + window.pass.prompt_end.unwrap_or(0);
                    if opened {
                        // A later chunk backs every row the prompt still writes.
                        rank.back(&mut lane.sequence, end)?;
                    }
                    lane.prefill_open = true;
                    chunk = Some(rank.chunk_plan(&lane.sequence, ids.len(), end)?);
                }
            }
        }
        rank.next_step();
        let swapped = path == Path::Chunk;
        if swapped {
            rank.swap_prefill()?;
        } else {
            rank.prof_begin();
        }
        let runs = Self::runs(path, chunk.as_ref(), segments);
        let set = rank.step_set(&self.tp, &runs, window.rows, device_leg);
        drop(runs);
        drop(rank);
        scratch.open.replace(Some(Open {
            path,
            set,
            chunk,
            position: segments[0].0.lane.sequence.position(),
            first: window.first,
            after: window.after.clone(),
            swapped,
            fused: false,
            started: Instant::now(),
        }));
        Ok(())
    }

    fn abandon_window(&self, scratch: &Scratch) {
        let Some(open) = scratch.open.take() else {
            return;
        };
        if open.swapped
            && let Ok(mut rank) = self.rank()
        {
            // The decode scratch goes back whatever the window's failure was; the failure is
            // the walk's answer.
            let _ = rank.swap_prefill();
        }
    }

    fn back(&self, lane: &mut GlmLane, reach: usize) -> Result<(), BackendError> {
        self.rank()?.back(&mut lane.sequence, reach)
    }

    fn embed(&self, scratch: &Scratch, ids: &[u32]) -> Result<(), BackendError> {
        let rank = self.rank()?;
        Self::open(scratch, |open| {
            rank.step_embed(&open.set, Some(open.position), ids)
        })
    }

    fn enter(&self, scratch: &Scratch, frame: &Frame, _rows: usize) -> Result<(), BackendError> {
        let rank = self.rank()?;
        Self::open(scratch, |open| {
            rank.step_enter(&self.tp, &open.set, frame.layer, frame.ffn, open.fused)
        })
    }

    fn pass<'a>(
        &'a self,
        scratch: &'a Scratch,
        segments: &'a [(&'a mut PlanSeq<GlmLane>, &'a [u32])],
        _layer: usize,
        _slot: usize,
        rows: usize,
        _mode: (),
    ) -> GlmPass<'a> {
        let runs = scratch
            .open
            .borrow()
            .as_ref()
            .map_or_else(Vec::new, |open| {
                Self::runs(open.path, open.chunk.as_ref(), segments)
            });
        GlmPass {
            rank: self.rank.lock().ok(),
            runs,
            rows,
        }
    }

    /// The exit sums every rank's partial inside the mix out, so `summed` is the rank's to act
    /// on, and the rank's layers name the next entry.
    fn leave(
        &self,
        scratch: &Scratch,
        frame: &Frame,
        _next: Option<&Frame>,
        _summed: bool,
        _rows: usize,
    ) -> Result<(), BackendError> {
        let rank = self.rank()?;
        Self::open(scratch, |open| {
            open.fused = rank.step_leave(&self.tp, &open.set, frame.layer, frame.ffn)?;
            Ok(())
        })
    }

    fn finish(
        &self,
        scratch: &Scratch,
        segments: &mut PlanRows<'_, '_, GlmLane>,
        _pass: PlanPass,
        _last: bool,
    ) -> Result<(), BackendError> {
        let mut rank = self.rank()?;
        let open = scratch
            .open
            .take()
            .ok_or_else(|| BackendError::new(DEVICE, "a tail ran outside a window"))?;
        let ids: Vec<u32> = segments
            .iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        let rows = ids.len();
        let collectives = (&self.tp, &self.gather);
        let mut left = scratch.left.borrow_mut();
        let host = |left: &mut Vec<(usize, Left)>, first: usize, logits: Vec<Vec<f32>>| {
            left.extend(
                logits
                    .into_iter()
                    .enumerate()
                    .map(|(row, logits)| (first + row, Left::Host(logits))),
            );
        };
        if open.path == Path::Chunk {
            let tail = {
                let runs = Self::runs(open.path, open.chunk.as_ref(), segments);
                rank.step_tail::<Vec<f32>>(collectives, &open.set, &runs, &ids, HeadRows::Last)
            };
            // The decode scratch goes back before anything else reads the rank.
            rank.swap_prefill()?;
            let logits = tail?;
            let plan = open.chunk.as_ref().ok_or_else(|| {
                BackendError::new(DEVICE, "a bulk chunk has no plan for its drafter")
            })?;
            rank.after_chunk(&mut segments[0].0.lane.sequence, rows, plan)?;
            host(&mut left, open.first + rows - 1, logits);
            return Ok(());
        }
        let verify = open.path == Path::Token { verify: true };
        let logits = {
            let runs = Self::runs(open.path, open.chunk.as_ref(), segments);
            if open.path == (Path::Token { verify: false }) {
                rank.step_tail::<u32>(collectives, &open.set, &runs, &ids, HeadRows::All)?;
                None
            } else {
                Some(rank.step_tail::<Vec<f32>>(
                    collectives,
                    &open.set,
                    &runs,
                    &ids,
                    HeadRows::All,
                )?)
            }
        };
        rank.prof_end(rows, open.started.elapsed().as_secs_f64() * 1000.0);
        // A greedy step's logits stay on the device, stamped with the step's LM head.
        let head = logits.is_none().then(|| rank.heads());
        let mut first = 0;
        for ((seq, ids), after) in segments.iter_mut().zip(&open.after) {
            let sequence = &mut seq.lane.sequence;
            // A block drafter's next block sits just past the pass's rows of the sequence.
            let end = sequence.position() + ids.len() + after;
            let at = rank.after_run(sequence, (first, ids.len(), rows), verify, end, head)?;
            if let Some(at) = at {
                left.push((open.first + first + ids.len() - 1, Left::Device(at)));
            }
            first += ids.len();
        }
        if let Some(logits) = logits {
            host(&mut left, open.first, logits);
        }
        Ok(())
    }

    /// A head over a verify the drafter did not settle is refused: its rows wait for the
    /// commit. A prompt's end closes its chunks, and a token a verify already consumed answers
    /// with the row it sampled, exactly as the one-token step it stands in for would have.
    fn answered(
        &self,
        lane: &mut GlmLane,
        fed: Option<u32>,
    ) -> Result<Option<StepOutput>, BackendError> {
        if lane.sequence.verifying() {
            return Err(BackendError::new(
                DEVICE,
                "a verify's rows are settled by its drafter's round, not by a head",
            ));
        }
        let Some(token) = fed else {
            // The prompt's chunks are done: the next prompt starts a new one.
            lane.prefill_open = false;
            return Ok(None);
        };
        if !lane.holds(token) {
            return Ok(None);
        }
        let entry = lane
            .pending
            .pop(token)
            .ok_or_else(|| BackendError::new(DEVICE, "the queue changed under its front"))?;
        Ok(Some(StepOutput {
            logits: entry.logits,
            next_token: entry.next_token,
            draft: Vec::new(),
        }))
    }

    fn ends(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }

    /// The rank keeps no copy of a layer's residual for a trace.
    fn residual(&self, _scratch: &Scratch, _rows: usize) -> Result<Vec<f32>, BackendError> {
        Err(BackendError::new(
            DEVICE,
            "GLM's walk does not trace its layers' residual streams",
        ))
    }

    fn logits(
        &self,
        scratch: &Scratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let left = scratch.left.borrow();
        let mut logits = Vec::with_capacity(count * self.vocab);
        for row in first..first + count {
            match left.iter().find(|(at, _)| *at == row) {
                Some((_, Left::Host(row))) => logits.extend_from_slice(row),
                Some((_, Left::Device(at))) => logits.extend(self.rank()?.logits_row(*at)?),
                None => {
                    return Err(BackendError::new(
                        DEVICE,
                        format!("row {row} of the last walk left no logits for a head"),
                    ));
                }
            }
        }
        Ok(logits)
    }

    /// Each row's argmax over the logits the host reads back, as the sampler's greedy choice
    /// takes it.
    fn choices(
        &self,
        scratch: &Scratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<u32>, BackendError> {
        (first..first + count)
            .map(|row| self.logits(scratch, row, 1).map(|logits| argmax(&logits)))
            .collect()
    }
}

/// The rows a verified chain has, its anchor included.
type Chain = Vec<u32>;

/// One member's verify answer: the drafter's confidence in each draft and the full-vocabulary
/// logits of every row the verify ran.
pub struct VerifyRows {
    confidence: Vec<f32>,
    rows: Vec<Vec<f32>>,
}

/// The arm's draft source: the checkpoint's MTP layer, or a loaded DFlash 2 block drafter.
///
/// This is the arm's round on the runtime's [Drafter] contract: [DeviceRank::draft_chain]
/// proposes the chain, one verify pass of the walk checks every member's anchor and drafts, and
/// the settle keeps the leading run ([DeviceRank::commit]). The runtime decides when rows are
/// shared ([Drafter::budget], [Drafter::wall]), which member walks alone, and how deep a chain
/// is; the arm states only what its rank can verify in one step.
pub struct GlmDrafter {
    shared: Arc<Shared>,
    /// The most drafts this rank's speculator proposes a step (zero when none is configured).
    depth: usize,
    wall: ExpertUnionWall,
    /// Whether the drafter is the block drafter or the checkpoint's MTP layer.
    kind: &'static str,
    /// Whether the depth policy chooses from the drafts' confidences.
    adaptive: bool,
}

impl GlmDrafter {
    /// The chain a round verifies after `anchor`: the anchor and the drafts the depth allows
    /// ([DeviceRank::draft_chain]), with the drafter's confidence in each draft it proposed,
    /// which the depth was chosen from and the round's verdict calibrates.
    fn draft_after(
        &self,
        rank: &DeviceRank,
        lane: &mut GlmLane,
        anchor: u32,
    ) -> Result<(Chain, Vec<f32>), BackendError> {
        // A draft writes the MTP caches past what the sequence keeps, so a chain longer than the
        // room left would draft past the window.
        let room = lane
            .sequence
            .capacity()
            .saturating_sub(lane.sequence.position() + 1);
        let (drafts, confidence) = match lane.spec.as_ref() {
            Some(spec) => rank.draft_chain(
                &self.shared.tp,
                &self.shared.gather,
                &mut lane.sequence,
                anchor,
                spec,
                room,
            )?,
            None => (Vec::new(), Vec::new()),
        };
        lane.confidence.clone_from(&confidence);
        let mut chunk = Vec::with_capacity(drafts.len() + 1);
        chunk.push(anchor);
        chunk.extend_from_slice(&drafts);
        Ok((chunk, confidence))
    }
}

impl Drafter<GlmWalk> for GlmDrafter {
    const PRESENT: bool = true;
    /// The verify's logits and the confidence the settle needs; the runtime never reads logits.
    type Rows = VerifyRows;

    fn wall(&self) -> &dyn StepWall {
        &self.wall
    }

    /// The member's held answer, or its next chain. The token joins the sampler history either
    /// way.
    fn open_round(&self, seq: &mut PlanSeq<GlmLane>, token: u32) -> Result<Draft, BackendError> {
        seq.sampler.record(token);
        if let Some(front) = seq.lane.pending.front() {
            let expected = front.token;
            let Some(next) = seq.lane.pending.pop(token) else {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "the engine fed token {token} where the verify produced {expected}; a \
                         check that did not accept the drafts must be re-drafted"
                    ),
                ));
            };
            return Ok(Draft::Held(StepOutput {
                logits: next.logits,
                next_token: next.next_token,
                draft: Vec::new(),
            }));
        }
        let rank = self.shared.rank()?;
        let (chunk, confidence) = self.draft_after(&rank, &mut seq.lane, token)?;
        drop(rank);
        let drafts = chunk[1..].to_vec();
        let chain = seq
            .lane
            .spec
            .as_ref()
            .map_or_else(Vec::new, |spec| spec.chain(&confidence, drafts.len()));
        Ok(Draft::Chain {
            confidence: chain,
            drafts,
        })
    }

    /// One verify pass of the walk over every segment's anchor and drafts, each row's logits
    /// read back with the member's confidence for [Drafter::settle].
    fn verify(
        &self,
        walk: &GlmWalk,
        segments: &mut [Segment<'_, PlanSeq<GlmLane>>],
        _depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        walk.walk(
            segments,
            Pass::verify(rows, OutputConstraint::new(&[], rows)),
        )?;
        let vocab = walk.host().vocab;
        let mut first = 0;
        let mut answers = Vec::with_capacity(segments.len());
        for segment in segments.iter() {
            let count = segment.rows.len();
            let logits = walk.logits(first, count)?;
            first += count;
            answers.push(VerifyRows {
                confidence: segment.seq.lane.confidence.clone(),
                rows: logits.chunks(vocab).map(<[f32]>::to_vec).collect(),
            });
        }
        Ok(answers)
    }

    /// The verify's settlement: the verdict, the commit and the queue of kept answers; the
    /// history keeps the rows the rank kept.
    fn settle(
        &self,
        seq: &mut PlanSeq<GlmLane>,
        chunk: &[u32],
        rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        let VerifyRows { confidence, rows } = rows;
        // Rows past the first disagreement draw nothing, so a seeded answer does not depend on
        // how many drafts the round verified.
        let picks =
            seq.sampler
                .choose_verify(rows.iter().map(Vec::as_slice), &chunk[1..], &seq.sampling);
        let kept = verdict(chunk, &picks);
        let eos = &self.shared.eos;
        // No row past the reply's end stays in the caches, so a reply that ends inside the round
        // leaves its sequence at the history its caller saw, which the next turn extends.
        let held = kept_to_end(kept, &picks, eos);
        let lane = &mut seq.lane;
        if chunk.len() > 1 {
            // Give back the rows the sequence does not keep, before this rank's next step.
            self.shared.rank()?.commit(&mut lane.sequence, held)?;
        }
        if let Some(spec) = lane.spec.as_mut() {
            spec.observe(&confidence, chunk.len() - 1, kept);
        }
        lane.pending.extend_round(chunk, &picks, &rows, held, eos);
        let position = lane.sequence.position();
        seq.keep(position)?;
        let next_token = picks.first().copied().filter(|token| !eos.contains(token));
        Ok(StepOutput {
            logits: rows.into_iter().next().unwrap_or_default(),
            next_token,
            draft: Vec::new(),
        })
    }

    /// One rank step holds at most [MAX_ROWS] rows: the anchors take one a member, and every
    /// member's drafts share the rest.
    fn budget(&self, seqs: usize) -> usize {
        DRAFT_BUDGET.rows(seqs, self.depth)
    }

    fn ready(&self, seq: &PlanSeq<GlmLane>, token: u32) -> bool {
        seq.lane.holds(token)
    }

    /// The caches have already consumed the answers; the history is cut by the runtime, and the
    /// next prompt goes back to the rewind point. Only the queued answers and the confidence go.
    fn give_back(&self, seq: &mut PlanSeq<GlmLane>, _to: usize) -> Result<(), BackendError> {
        seq.lane.pending.clear();
        seq.lane.confidence.clear();
        Ok(())
    }

    fn speculation(&self, seq: &PlanSeq<GlmLane>) -> Speculation {
        seq.lane
            .spec
            .as_ref()
            .map_or_else(Speculation::default, Speculator::stats)
    }

    fn describe(&self) -> String {
        if self.depth == 0 {
            return "the round path with no drafts configured".to_owned();
        }
        let depth = if self.adaptive {
            "adaptive depth"
        } else {
            "fixed depth"
        };
        format!("{}, up to {} drafts, {depth}", self.kind, self.depth)
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_model::ledger::Strategy;
    use sparkflash_runtime::testing::ArmProfile;
    use sparkflash_runtime::{
        ArmFixture, Fixture, FormatRule, Mode, StateLayout, Walk, ledger_for,
    };

    use super::*;

    /// The GLM fixture: the synthetic `small` checkpoint, opened through the arm's own TP=1
    /// constructor. It is a real checkpoint in the released storage formats, not a served image.
    /// The GLM fixture at `world` ranks: a [`sparkflash_runtime::local_group`] whose ranks open
    /// the synthetic checkpoint through the arm's group constructor, as a served group's ranks
    /// do, and sum over one in-process bus on the exact f32 wire. The TP check reads its tokens
    /// only, so the fixture states no byte counts.
    fn group_fixture(
        world: usize,
        config: crate::Glm5NextConfig,
        directory: tempfile::TempDir,
    ) -> Result<Fixture<super::GlmWalk>, BackendError> {
        use sparkflash_weights::tp::{
            BitGatherHandle, LocalBus, LocalRank, TpLayout, TpRuntime, Wire,
        };

        let checkpoint = directory.path().to_path_buf();
        let context = config.max_position_embeddings;
        let bus = Arc::new(LocalBus::new(world).with_wire(Wire::F32));
        let open = move |rank: usize| {
            let failed =
                |error: String| BackendError::new("fixture", format!("rank {rank}: {error}"));
            let local = || LocalRank {
                rank,
                bus: Arc::clone(&bus),
            };
            let layout = TpLayout {
                rank,
                world_size: world,
            };
            let tp = TpRuntime::new(layout, Box::new(local()))
                .map_err(|error| failed(error.to_string()))?;
            let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
                .map_err(|error| failed(format!("CUDA: {error}")))?;
            crate::Glm5NextDeviceBackend::open(
                Arc::new(crate::device::CudaHandle::new(cuda)),
                config.clone(),
                &checkpoint,
                layout,
                tp,
                BitGatherHandle::new(Box::new(local())),
                context,
            )?
            .walk_arm()
        };
        let wire = move || -> Box<dyn sparkflash_runtime::GroupWire> {
            Box::new(sparkflash_runtime::TextWire::new("glm", "GLM", context))
        };
        let (walk, group, workers) = sparkflash_runtime::local_group(world, open, wire)?;
        let keep: Box<dyn std::any::Any> = Box::new((workers, directory));
        Ok(Fixture::new(
            walk,
            format!("the GLM-5.3 synthetic small checkpoint at TP {world}"),
            0,
            0,
            Vec::new(),
            Some(keep),
        )
        .in_group(group))
    }

    /// The arm as the record and the generated checks name it: a marker with the walk's own
    /// name, since the walk is the runtime's type.
    pub struct GlmWalk;

    impl ArmFixture for GlmWalk {
        type Walk = super::GlmWalk;

        fn fixture(world: usize) -> Result<Fixture<super::GlmWalk>, BackendError> {
            let config = crate::Glm5NextConfig::small();
            let directory = tempfile::tempdir()
                .map_err(|error| BackendError::new("fixture", error.to_string()))?;
            crate::checkpoint::write_synthetic_checkpoint(directory.path(), &config, 7, 1)
                .map_err(|error| BackendError::new("fixture", error))?;
            if world > 1 {
                return group_fixture(world, config, directory);
            }
            let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
                .map_err(|error| BackendError::new("fixture", format!("CUDA: {error}")))?;
            let backend = crate::Glm5NextDeviceBackend::open_single_rank(
                std::sync::Arc::new(crate::device::CudaHandle::new(cuda)),
                config.clone(),
                directory.path(),
                config.max_position_embeddings,
            )?;
            let walk = backend.walk_arm()?;
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
                config.n_routed_experts,
                config.num_experts_per_tok,
            );
            Ok(Fixture::new(
                walk,
                "the GLM-5.3 synthetic small checkpoint",
                bytes,
                floor,
                FormatRule::classes(&config.architecture.quantization),
                Some(Box::new(directory)),
            ))
        }

        /// GLM's single-process path serves on the tick engine, so its generated checks drive
        /// that engine exactly as dense's do. Its group is still the fused engine's; the TP leg
        /// of the check skips because the fixture is one rank.
        fn tick_engine() -> bool {
            true
        }
    }

    /// The CLI's profile check and the ledger's token row are one constant: version 3 states the
    /// tiled row GEMV, so `decode.tensor_cores = true` is refused with the row's reason and the
    /// boot line says which arithmetic runs.
    #[test]
    fn the_token_row_and_the_profile_check_are_one_source() {
        let State::Off(detail) = crate::TOKEN_DECODE else {
            panic!("GLM version 3 decodes on the tiled row GEMV; the CLI reads this row");
        };
        assert!(detail.contains("row GEMV"), "{detail}");
        assert_eq!(
            <super::GlmWalk as Walk>::CAPABILITIES.compute.token,
            crate::TOKEN_DECODE
        );
        let mut ledger = ledger_for::<super::GlmWalk>();
        ledger.set(Strategy::Lanes, Mode::native("one segment a walk"));
        let line = ledger.line();
        assert!(
            line.contains(&format!("tensor-core decode off ({detail})")),
            "the boot line states the row GEMV: {line}"
        );
        assert!(
            !line.contains("does not state it"),
            "the runtime derives every row GLM states: {line}"
        );
    }

    /// The layout follows the plan: a positional latent and indexer region and a recurrent KDA
    /// one.
    #[test]
    fn the_layout_is_the_plans_two_kinds_of_state() {
        let positional = StateLayout::new(regions_of(sparkflash_model::KvPlan::new(1_920, 0)));
        assert_eq!(positional.kind(), StateKind::Positional);
        let both = StateLayout::new(regions_of(sparkflash_model::KvPlan::new(20_160, 4_096)));
        assert_eq!(both.kind(), StateKind::Both);
        assert!(both.needs_checkpoint());
        assert_eq!(both.bytes_per_row(), 20_160);
        assert_eq!(both.bytes_at(2), 40_320 + 4_096);
    }

    /// The member cap is the rank's verify width, not the prompt chunk: a verify is one rank
    /// step, so the engine may hand the walk no more members than that.
    #[test]
    fn the_member_cap_is_the_rank_verify_width() {
        let limits = walk_limits(crate::device::DEFAULT_PREFILL_ROWS, 1_048_576);
        assert_eq!(limits.rows, crate::device::DEFAULT_PREFILL_ROWS);
        assert_eq!(limits.members, MAX_ROWS);
        assert_eq!(limits.context, 1_048_576);
        assert_ne!(limits.members, limits.rows, "the two widths are separate");
    }

    /// What the arm states for the generated profile and round tests: the default prefill rows
    /// as the pass width and the rank's verify width as the member cap, as [walk_limits] states
    /// them; the MTP chain's three drafts under the arm's budget and wall; the plan's two kinds
    /// of state; and the chunks a 5,000-row prompt runs in.
    fn stated() -> ArmProfile {
        let limits = walk_limits(crate::device::DEFAULT_PREFILL_ROWS, 1_048_576);
        ArmProfile::new(
            "GLM",
            StateLayout::new(regions_of(sparkflash_model::KvPlan::new(20_160, 4_096))),
            limits,
        )
        .with_code("glm5_shape")
        .drafting(3, DRAFT_BUDGET)
        .with_wall(JOINT_WALL)
        .prompt(limits, 5_000, &[4_096, 904])
    }

    sparkflash_runtime::conformance!(super::GlmWalk, profile: stated(), rounds);
}
