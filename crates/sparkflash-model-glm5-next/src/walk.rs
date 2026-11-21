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

//! GLM-5.3-Flash as a Walk.
//!
//! The runtime drives an arm: it owns the prompt parts, the history, reuse, rewind, the lane
//! pool, give-back and the ledger, and this walk supplies only what GLM alone knows. GLM's
//! serving construction opens [sparkflash_runtime::Fusion] over a
//! [`sparkflash_runtime::Runtime`] of this walk
//! ([crate::serve::Glm5NextDeviceBackend::enable_fusion]); nothing of the arm's own session,
//! prompt-part or round glue is left.
//!
//! * [Walk::layout] is what the rank's caches are: every sparse (DSA) layer holds a positional
//!   latent row and its indexer's rows a position, and every linear (KDA) layer holds a
//!   convolution history and a recurrent memory that are one live value. The bytes come from the
//!   rank's own plan ([DeviceRank::kv_plan]), not a compiled-in estimate.
//! * [Walk::limits] is the rank's bulk chunk width ([DeviceRank::prefill_rows], the profile's
//!   4096 by default), the served context, and the rank's verify width ([MAX_ROWS]) as the
//!   member cap. The runtime's [sparkflash_runtime::ChunkPolicy] quantises the first, so a GLM
//!   prompt still runs the 4096-row chunks its kernels and scratch were sized for.
//! * [Walk::walk] runs the same calls the old serving loop ran: a bulk prompt through
//!   [DeviceRank::begin_prefill]/[DeviceRank::prefill_chunk] (the tensor-core chunk path), a
//!   token row through [DeviceRank::forward_batch_greedy] (the row-GEMV decode path), and a
//!   verify through [DeviceRank::verify_batch].
//! * [Walk::head] is the arm's existing head: the host sampler over the row the greedy step left
//!   on the device, or a verified chain's [sparkflash_model::verdict] and [DeviceRank::commit].
//! * [Walk::Drafter] is [GlmDrafter], the arm's one round: the MTP layer's or the DFlash 2
//!   block drafter's chain ([GlmSequence::draft_after]), the shared verify
//!   ([DeviceRank::verify_batch]) and the settlement ([GlmSequence::settle]). The runtime decides
//!   which members share a pass, when a lane walks alone, and how many rows a lone member keeps;
//!   the arm states only its depth, its verify width and its cost ([Drafter::budget],
//!   [Drafter::wall]).

use std::sync::{Arc, Mutex, OnceLock};

use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_model::{DepthPolicy, Queue, Speculator, kept_to_end, verdict};
use sparkflash_ops::sampler::Sampler;
use sparkflash_runtime::{
    ComputeDeclaration, Draft, DraftBudget, Drafter, ExpertUnionWall, HeadRow, LaneState, Limits,
    NoVision, OutputConstraint, Pass, PassKind, PassMask, Region, Residual, Segment, Segments,
    StateKind, StateLayout, StepWall, Walk, WalkCapabilities, check_checkpoint, check_rewind,
};
use sparkflash_weights::tp::{BitGatherHandle, TpRuntime};

use crate::device::{DeviceRank, DeviceSequence, MAX_ROWS, Prefill};

/// How a session drafts, once the backend has been told to.
#[derive(Debug, Clone)]
pub(crate) struct SpecConfig {
    /// The most drafts a step verifies.
    pub(crate) drafts: usize,
    /// The depth policy, when the drafter says how sure it is of each draft (the block drafter).
    pub(crate) policy: Option<DepthPolicy>,
}

/// The backend's rank and the collectives every walk runs through.
pub(crate) struct Shared {
    pub(crate) rank: Mutex<DeviceRank>,
    pub(crate) tp: TpRuntime,
    pub(crate) gather: BitGatherHandle,
    pub(crate) eos: Vec<u32>,
    /// Set once, before the first sequence opens: a sequence's sequence is a drafting one or it
    /// is not, so the choice cannot change under a live one.
    pub(crate) spec: OnceLock<SpecConfig>,
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

/// The rows a verified chain has, its anchor included.
type Chain = Vec<u32>;

/// A round the walk verified for one sequence, waiting for [Walk::head] or [GlmDrafter] to keep
/// its leading run.
struct Verified {
    /// The anchor followed by the drafts the share left.
    chunk: Chain,
    /// The drafter's confidence in each draft it proposed.
    confidence: Vec<f32>,
    /// One row of full-vocabulary logits for a row of `chunk`.
    rows: Vec<Vec<f32>>,
}

/// What one segment of a pass left on the host for its head.
struct Left {
    /// The pass's row after this segment's last, so a head row finds its segment.
    end: usize,
    /// The round a verify pass ran for the segment's sequence.
    verified: Option<Verified>,
    /// A bulk chunk's last row of logits, which [DeviceRank::prefill_chunk] leaves on the host.
    prompt_logits: Option<Vec<f32>>,
}

/// What a GLM pass hands its head: the row count, and what each segment left on the host.
///
/// A token row leaves nothing here: the rank keeps the last kept row's logits and the head
/// reads them back. A prompt chunk that is not the prompt's last is never headed, and its
/// logits go when its value does.
pub struct GlmHidden {
    rows: usize,
    /// One entry a segment of a bulk or verify pass, in order; empty for a token pass.
    left: Vec<Left>,
}

impl GlmHidden {
    /// What the segment holding `row` left, when the pass left anything.
    fn left(&mut self, row: usize) -> Option<&mut Left> {
        self.left.iter_mut().find(|left| row < left.end)
    }
}

impl Residual for GlmHidden {
    fn rows(&self) -> usize {
        self.rows
    }
}

/// One sequence's device state, as the runtime drives it.
///
/// The runtime owns the history; this owns the rank's cache and the request's sampler. A
/// speculative sequence also owns its queue of kept answers, because a verify produces more than
/// the one token a step answers with.
pub struct GlmSequence {
    shared: Arc<Shared>,
    sequence: DeviceSequence,
    sampler: Sampler,
    sampling: Sampling,
    /// The draft depth and its accounting, when this sequence drafts.
    spec: Option<Speculator>,
    /// Tokens a verify already produced, oldest first, each with the token that follows it.
    pending: Queue,
    /// The drafter's confidence in each of the last round's drafts.
    confidence: Vec<f32>,
    /// Whether a prompt is running a chunk at a time ([Self::bulk_chunk]).
    prefill_open: bool,
}

impl GlmSequence {
    fn new(shared: Arc<Shared>, sequence: DeviceSequence) -> Self {
        let spec = shared
            .spec
            .get()
            .map(|spec| Speculator::new(spec.drafts, spec.policy.clone()));
        Self {
            shared,
            sequence,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
            spec,
            pending: Queue::new(),
            confidence: Vec::new(),
            prefill_open: false,
        }
    }

    /// Whether the sequence already holds the answer for a step fed `token`.
    #[must_use]
    pub fn holds(&self, token: u32) -> bool {
        self.pending
            .front()
            .is_some_and(|front| front.token == token)
    }

    /// The caller fed `token`: it joins the sampler history, and the answer a verify already
    /// holds for it is returned. `None` says the token still needs a round.
    ///
    /// # Errors
    /// A token other than the one the verify kept: the caches have consumed that one already.
    fn take_held(&mut self, token: u32) -> Result<Option<StepOutput>, BackendError> {
        self.sampler.record(token);
        let Some(front) = self.pending.front() else {
            return Ok(None);
        };
        let expected = front.token;
        let Some(next) = self.pending.pop(token) else {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "the engine fed token {token} where the verify produced {expected}; a check that did not accept the drafts must be re-drafted"
                ),
            ));
        };
        Ok(Some(StepOutput {
            logits: next.logits,
            next_token: next.next_token,
            draft: Vec::new(),
        }))
    }

    /// The chain a round verifies after `anchor`: the anchor and the drafts the depth allows
    /// ([DeviceRank::draft_chain]), with the drafter's confidence in each draft it proposed,
    /// which the depth was chosen from and the round's verdict calibrates.
    fn draft_after(
        &mut self,
        rank: &DeviceRank,
        anchor: u32,
    ) -> Result<(Chain, Vec<f32>), BackendError> {
        // A draft writes the MTP caches past what the sequence keeps, so a chain longer than the
        // room left would draft past the window.
        let room = self
            .sequence
            .capacity()
            .saturating_sub(self.sequence.position() + 1);
        let (drafts, confidence) = match self.spec.as_ref() {
            Some(spec) => rank.draft_chain(
                &self.shared.tp,
                &self.shared.gather,
                &mut self.sequence,
                anchor,
                spec,
                room,
            )?,
            None => (Vec::new(), Vec::new()),
        };
        self.confidence.clone_from(&confidence);
        let mut chunk = Vec::with_capacity(drafts.len() + 1);
        chunk.push(anchor);
        chunk.extend_from_slice(&drafts);
        Ok((chunk, confidence))
    }

    /// A verify of `verified` answered its rows: keep what the model chose, give the rest back
    /// before the rank's next step, and queue the kept rows for the calls that feed them.
    /// [GlmDrafter]'s round settles through here, and [Walk::head] calls it for a
    /// [PassKind::Verify] pass a caller drove directly.
    fn settle(
        &mut self,
        rank: &DeviceRank,
        verified: Verified,
    ) -> Result<StepOutput, BackendError> {
        let Verified {
            chunk,
            confidence,
            rows,
        } = verified;
        // Rows past the first disagreement draw nothing, so a seeded answer does not depend on
        // how many drafts the round verified.
        let picks =
            self.sampler
                .choose_verify(rows.iter().map(Vec::as_slice), &chunk[1..], &self.sampling);
        let kept = verdict(&chunk, &picks);
        let eos = &self.shared.eos;
        // No row past the reply's end stays in the caches, so a reply that ends inside the round
        // leaves its sequence at the history its caller saw, which the next turn extends.
        let held = kept_to_end(kept, &picks, eos);
        if chunk.len() > 1 {
            // Give back the rows the sequence does not keep, before this rank's next step.
            rank.commit(&mut self.sequence, held)?;
        }
        if let Some(spec) = self.spec.as_mut() {
            spec.observe(&confidence, chunk.len() - 1, kept);
        }
        self.pending.extend_round(&chunk, &picks, &rows, held, eos);
        let next_token = picks.first().copied().filter(|token| !eos.contains(token));
        Ok(StepOutput {
            logits: rows.first().cloned().unwrap_or_default(),
            next_token,
            draft: Vec::new(),
        })
    }

    /// Forwards one bulk prompt chunk and answers its last row's logits, for the head.
    ///
    /// The first chunk asks [DeviceRank::begin_prefill] which path the prompt takes: the token
    /// path whole (a prompt below [MIN_PREFILL_ROWS], or a rank with no chunk scratch), or the
    /// bulk chunks, which is the tensor-core prompt path. Every later chunk is a
    /// [DeviceRank::prefill_chunk] of the prompt [DeviceRank::begin_prefill] started, as the old
    /// serving loop's `prompt_part` ran it.
    ///
    /// `after` is the prompt's remaining rows ([Pass::prompt_end]): the old loop passed the
    /// prompt's whole end to every chunk, and the block drafter's window commits read it, so a
    /// chunk that used its own end instead would move the drafter's rows.
    fn bulk_chunk(
        &mut self,
        rank: &mut DeviceRank,
        rows: &[u32],
        after: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let first = !self.prefill_open;
        if first {
            // A new prompt's tokens replace any chain the last one left queued.
            self.pending.clear();
            self.confidence.clear();
        }
        let end = self.sequence.position() + rows.len() + after;
        let logits = if first {
            match rank.begin_prefill(
                &self.shared.tp,
                &self.shared.gather,
                &mut self.sequence,
                rows,
            )? {
                Prefill::Whole(logits) => logits,
                Prefill::Chunks { .. } => {
                    self.prefill_open = true;
                    rank.prefill_chunk(
                        &self.shared.tp,
                        &self.shared.gather,
                        &mut self.sequence,
                        rows,
                        end,
                    )?
                }
            }
        } else {
            // The lane must back the rows this chunk writes before it runs; the first chunk's
            // begin_prefill backed only its own.
            rank.back(&mut self.sequence, end)?;
            rank.prefill_chunk(
                &self.shared.tp,
                &self.shared.gather,
                &mut self.sequence,
                rows,
                end,
            )?
        };
        Ok(logits)
    }
}

impl LaneState for GlmSequence {
    fn position(&self) -> usize {
        self.sequence.position()
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        check_checkpoint("glm5_device", at, self.sequence.position())?;
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        rank.keep_point(&mut self.sequence)
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        check_rewind("glm5_device", to, self.sequence.position())?;
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        if to == 0 {
            rank.reset(&mut self.sequence)?;
        } else {
            rank.rewind(&mut self.sequence, to)?;
        }
        drop(rank);
        self.pending.clear();
        self.confidence.clear();
        self.prefill_open = false;
        Ok(())
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        // The runtime asks about the history's positions; the state is at least there, and a
        // drafting step also writes its chain's rows past it.
        let drafts = self.spec.as_ref().map_or(0, Speculator::max_depth);
        let wanted = positions.max(self.sequence.position() + 1 + drafts);
        self.sequence.room(wanted, reset)
    }
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

/// One member's verify answer: the drafter's confidence in each draft and the full-vocabulary
/// logits of every row the verify ran.
pub struct VerifyRows {
    confidence: Vec<f32>,
    rows: Vec<Vec<f32>>,
}

/// The arm's draft source: the checkpoint's MTP layer, or a loaded DFlash 2 block drafter.
///
/// This is the round the arm's old serving fusion ran, expressed on the runtime's
/// [Drafter] contract: [GlmSequence::draft_after] proposes the chain,
/// [DeviceRank::verify_batch] verifies every member's anchor and drafts in one pass, and
/// [GlmSequence::settle] keeps the leading run. The runtime decides when rows are shared
/// ([Drafter::budget], [Drafter::wall]), which member walks alone, and how deep a chain is; the
/// arm states only what its rank can verify in one step.
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

impl Drafter<GlmWalk> for GlmDrafter {
    const PRESENT: bool = true;
    /// The verify's logits and the confidence the settle needs; the runtime never reads logits.
    type Rows = VerifyRows;

    fn wall(&self) -> &dyn StepWall {
        &self.wall
    }

    /// The member's held answer, or its next chain. The token joins the sampler history either
    /// way, exactly as the old round's intake did.
    fn open_round(&self, seq: &mut GlmSequence, token: u32) -> Result<Draft, BackendError> {
        if let Some(step) = seq.take_held(token)? {
            return Ok(Draft::Held(step));
        }
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        let (chunk, confidence) = seq.draft_after(&rank, token)?;
        let drafts = chunk[1..].to_vec();
        let chain = seq
            .spec
            .as_ref()
            .map_or_else(Vec::new, |spec| spec.chain(&confidence, drafts.len()));
        Ok(Draft::Chain {
            confidence: chain,
            drafts,
        })
    }

    /// One [DeviceRank::verify_batch] over every segment's anchor and drafts, as the old serving
    /// fusion ran it. The walk hands each member's rows and confidence back in its
    /// [GlmHidden]; this takes them for [Drafter::settle].
    fn verify(
        &self,
        walk: &GlmWalk,
        segments: &mut [Segment<'_, GlmSequence>],
        _depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        let members = segments.len();
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        let mut hidden = walk.walk(
            segments,
            Pass::verify(rows, OutputConstraint::new(&[], rows)),
        )?;
        let mut answers = Vec::with_capacity(members);
        for member in 0..members {
            let verified = hidden
                .left
                .get_mut(member)
                .and_then(|left| left.verified.take())
                .ok_or_else(|| {
                    BackendError::new("glm5_device", "a member walks a round it never verified")
                })?;
            answers.push(VerifyRows {
                confidence: verified.confidence,
                rows: verified.rows,
            });
        }
        if answers.len() != members {
            return Err(BackendError::new(
                "glm5_device",
                format!("a verify of {members} member(s) answered {}", answers.len()),
            ));
        }
        Ok(answers)
    }

    /// The verify's settlement: the verdict, the commit and the queue of kept answers, exactly as
    /// the old fusion's round settled.
    fn settle(
        &self,
        seq: &mut GlmSequence,
        chunk: &[u32],
        rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        seq.settle(
            &rank,
            Verified {
                chunk: chunk.to_vec(),
                confidence: rows.confidence,
                rows: rows.rows,
            },
        )
    }

    /// One rank step holds at most [MAX_ROWS] rows: the anchors take one a member, and every
    /// member's drafts share the rest.
    fn budget(&self, seqs: usize) -> usize {
        DRAFT_BUDGET.rows(seqs, self.depth)
    }

    fn ready(&self, seq: &GlmSequence, token: u32) -> bool {
        seq.holds(token)
    }

    /// The caches have already consumed the answers; the history is cut by the runtime, and the
    /// next prompt goes back to the rewind point. Only the queued answers and the confidence go.
    fn give_back(&self, seq: &mut GlmSequence, _to: usize) -> Result<(), BackendError> {
        seq.pending.clear();
        seq.confidence.clear();
        Ok(())
    }

    fn speculation(&self, seq: &GlmSequence) -> Speculation {
        seq.spec
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

/// The GLM-5.3-Flash arm, exposed to the shared runtime as a Walk.
pub struct GlmWalk {
    shared: Arc<Shared>,
    context_limit: usize,
    layout: StateLayout,
    limits: Limits,
    /// The arm's draft source, behind the runtime's round contract.
    drafter: GlmDrafter,
}

impl GlmWalk {
    /// The walk over `shared`'s rank and `context_limit` positions a sequence.
    ///
    /// # Errors
    /// A poisoned rank lock, or a rank whose cache plan cannot be read.
    pub(crate) fn new(shared: Arc<Shared>, context_limit: usize) -> Result<Self, BackendError> {
        let (plan, chunk, wall, block) = {
            let rank = shared
                .rank
                .lock()
                .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
            let config = rank.config();
            let plan = rank.kv_plan().map_err(|error| {
                BackendError::new("glm5_device", format!("the rank's cache plan: {error}"))
            })?;
            (
                plan,
                rank.prefill_rows(),
                JOINT_WALL.with_experts(config.n_routed_experts, config.num_experts_per_tok),
                rank.has_block_drafter(),
            )
        };
        let spec = shared.spec.get();
        let drafter = GlmDrafter {
            shared: Arc::clone(&shared),
            depth: spec.map_or(0, |spec| spec.drafts),
            wall,
            kind: if block { "DFlash 2" } else { "MTP" },
            adaptive: spec.is_some_and(|spec| spec.policy.is_some()),
        };
        Ok(Self {
            shared,
            context_limit,
            layout: layout_of(plan),
            // The runtime's ChunkPolicy quantises this, so a profile's 4096 stays 4096 and the
            // rank's scratch can always hold the chunk it is handed. A verify is one rank step,
            // so the member cap is the rank's verify width, not the prompt chunk.
            limits: walk_limits(chunk, context_limit),
            drafter,
        })
    }
}

/// The arm's state, as the runtime needs to see it: the sparse layers' latent and indexer rows
/// grow a position at a time, and the linear layers' convolution history and recurrent memory
/// are one live value a layer.
fn layout_of(plan: sparkflash_model::KvPlan) -> StateLayout {
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
    StateLayout::new(regions)
}

impl Walk for GlmWalk {
    type Seq = GlmSequence;
    type Hidden = GlmHidden;
    /// The checkpoint's MTP layer or the loaded DFlash 2 block drafter, as the runtime's round.
    type Drafter = GlmDrafter;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        // A prompt runs in bulk chunks and a step runs token rows; a verify pass is the round
        // the walk runs itself, for a runtime caller that has rows to check.
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

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self) -> Result<Self::Seq, BackendError> {
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        // A sequence is a drafting one or it is not for its whole life: the rank's drafter is
        // chosen here, before the first prompt, exactly as the old fusion's slot did.
        let sequence = match self.shared.spec.get() {
            Some(_) if rank.has_block_drafter() => {
                rank.new_block_drafting_sequence(self.context_limit)?
            }
            Some(_) => rank.new_drafting_sequence(self.context_limit)?,
            None => rank.new_sequence(self.context_limit)?,
        };
        drop(rank);
        Ok(GlmSequence::new(Arc::clone(&self.shared), sequence))
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn drafter(&self) -> &Self::Drafter {
        &self.drafter
    }

    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling) {
        // The penalties, temperature, top-k and top-p run on the host over the row this step
        // already copies back, so a non-greedy request is served rather than approximated.
        seq.sampling = sampling;
        seq.sampler.reset();
        // The adapter calls this once per request, and a kept sequence carries the last one's
        // counts: a request reports its own speculation, not the session's lifetime.
        if let Some(spec) = seq.spec.as_mut() {
            spec.reset();
        }
        seq.confidence.clear();
        seq.pending.clear();
    }

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        pass: Pass<'_>,
    ) -> Result<GlmHidden, BackendError> {
        let after = pass.prompt_end.unwrap_or(0);
        let mut rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        match pass.kind {
            PassKind::Bulk { .. } | PassKind::Paired { .. } => {
                let mut rows = 0;
                let mut left = Vec::with_capacity(segments.len());
                for segment in segments.iter_mut() {
                    rows += segment.rows.len();
                    let words = segment.rows;
                    let seq = &mut *segment.seq;
                    let logits = seq.bulk_chunk(&mut rank, words, after)?;
                    left.push(Left {
                        end: rows,
                        verified: None,
                        prompt_logits: Some(logits),
                    });
                }
                Ok(GlmHidden { rows, left })
            }
            PassKind::Verify { .. } => {
                let mut rows = 0;
                let mut batch: Vec<(&mut DeviceSequence, &[u32])> = segments
                    .iter_mut()
                    .map(|segment| {
                        rows += segment.rows.len();
                        (&mut segment.seq.sequence, segment.rows)
                    })
                    .collect();
                let verified =
                    rank.verify_batch(&self.shared.tp, &self.shared.gather, &mut batch)?;
                drop(batch);
                let mut end = 0;
                let left = segments
                    .iter()
                    .zip(verified)
                    .map(|(segment, rows)| {
                        end += segment.rows.len();
                        Left {
                            end,
                            verified: Some(Verified {
                                chunk: segment.rows.to_vec(),
                                confidence: segment.seq.confidence.clone(),
                                rows,
                            }),
                            prompt_logits: None,
                        }
                    })
                    .collect();
                Ok(GlmHidden { rows, left })
            }
            PassKind::TokenRows => self.token_rows(&mut rank, segments),
        }
    }

    fn head(
        &self,
        hidden: &mut GlmHidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        let mut steps = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            if row.row >= hidden.rows {
                return Err(BackendError::new(
                    "glm5_device",
                    format!("the head was asked for row {} of {}", row.row, hidden.rows),
                ));
            }
            let seq = &mut *row.seq;
            let fed = row.input.record(&mut seq.sampler);
            if fed.is_none() {
                // The prompt's chunks are done: the next prompt starts a new one.
                seq.prefill_open = false;
            }
            // A verify already consumed this token and produced the answer: hand back the row it
            // sampled, exactly as the one-token step it stands in for would have.
            if let Some(token) = fed
                && seq
                    .pending
                    .front()
                    .is_some_and(|front| front.token == token)
            {
                let entry = seq.pending.pop(token).ok_or_else(|| {
                    BackendError::new("glm5_device", "the queue changed under its front")
                })?;
                steps.push(StepOutput {
                    logits: entry.logits,
                    next_token: entry.next_token,
                    draft: Vec::new(),
                });
                continue;
            }
            let mut left = hidden.left(row.row);
            if let Some(verified) = left.as_mut().and_then(|left| left.verified.take()) {
                steps.push(seq.settle(&rank, verified)?);
                continue;
            }
            // The greedy path leaves each row's pick on the device and the last kept row's
            // logits behind it, so a step reads back one vocabulary row, not one a row. A bulk
            // prompt's last row came back from prefill_chunk instead.
            let logits = match left.and_then(|left| left.prompt_logits.take()) {
                Some(logits) => logits,
                None => rank.last_logits(&seq.sequence)?,
            };
            let token = seq.sampler.choose(&logits, &seq.sampling);
            let next_token = if self.shared.eos.contains(&token) {
                None
            } else {
                Some(token)
            };
            steps.push(StepOutput {
                logits,
                next_token,
                draft: Vec::new(),
            });
        }
        Ok(steps)
    }
}

impl GlmWalk {
    /// One token a segment: every member's row through the rank's batched greedy path, whose
    /// last kept row's logits [Walk::head] reads back.
    ///
    /// The round path is [GlmDrafter]'s; this is the plain pass the runtime's own plain walk
    /// and a conformance comparison still use. A batch wider than [MAX_ROWS] rows runs as one
    /// call: [DeviceRank::forward_batch_greedy] splits it a step at a time.
    fn token_rows(
        &self,
        rank: &mut DeviceRank,
        segments: &mut [Segment<'_, GlmSequence>],
    ) -> Result<GlmHidden, BackendError> {
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        for segment in segments.iter_mut() {
            // A step ends any prompt that was running a chunk at a time.
            segment.seq.prefill_open = false;
        }
        let mut batch: Vec<(&mut DeviceSequence, &[u32])> = segments
            .iter_mut()
            .map(|segment| (&mut segment.seq.sequence, segment.rows))
            .collect();
        rank.forward_batch_greedy(&self.shared.tp, &self.shared.gather, &mut batch)?;
        Ok(GlmHidden {
            rows,
            left: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_model::ledger::Strategy;
    use sparkflash_runtime::testing::{ShapeWalk, format_rule, token_bytes};
    use sparkflash_runtime::{Fixture, Mode, Runtime, Walk, WalkFixture, ledger_for};

    use super::*;

    /// The stored bytes a token pass must read at this rank, and the bytes the fixture's plan
    /// says a token reads once the routed experts pay their top-k share.
    fn fixture_token_bytes(
        architecture: &sparkflash_model::Architecture,
        experts: usize,
        top_k: usize,
    ) -> Result<(u64, u64), BackendError> {
        let tp = sparkflash_model::TpPlan::new(1, 0)
            .map_err(|error| BackendError::new("fixture", error.detail))?;
        let tensors =
            sparkflash_ops::tensor_plan::ArchitectureTensorPlan::served_and_mtp(architecture, tp)
                .map_err(|error| BackendError::new("fixture", error.to_string()))?;
        let total = u64::try_from(tensors.rank_bytes()).unwrap_or(u64::MAX);
        let expert = u64::try_from(tensors.role_bytes(sparkflash_ops::tensor_plan::Role::Experts))
            .unwrap_or(u64::MAX);
        Ok(token_bytes(total, expert, experts, top_k))
    }

    /// The GLM fixture: the synthetic `small` checkpoint, opened through the arm's own TP=1
    /// constructor. It is a real checkpoint in the released storage formats, not a served image.
    impl WalkFixture for GlmWalk {
        fn fixture(world: usize) -> Result<Fixture<Self>, BackendError> {
            if world > 1 {
                return Err(BackendError::new(
                    "group",
                    "the GLM fixture is one rank; TP 2 and TP 4 need a four-rank group",
                ));
            }
            let config = crate::Glm5NextConfig::small();
            let directory = tempfile::tempdir()
                .map_err(|error| BackendError::new("fixture", error.to_string()))?;
            crate::checkpoint::write_synthetic_checkpoint(directory.path(), &config, 7, 1)
                .map_err(|error| BackendError::new("fixture", error))?;
            let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
                .map_err(|error| BackendError::new("fixture", format!("CUDA: {error}")))?;
            let backend = crate::Glm5NextDeviceBackend::open_single_rank(
                std::sync::Arc::new(crate::device::CudaHandle::new(cuda)),
                config.clone(),
                directory.path(),
                config.max_position_embeddings,
            )?;
            let walk = backend.walk_arm()?;
            let (bytes, floor) = fixture_token_bytes(
                &config.architecture,
                config.n_routed_experts,
                config.num_experts_per_tok,
            )?;
            let formats = vec![
                format_rule(config.architecture.quantization.dense_format),
                format_rule(config.architecture.quantization.expert_format),
            ];
            Ok(Fixture::new(
                walk,
                "the GLM-5.3 synthetic small checkpoint",
                bytes,
                floor,
                formats,
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
            <GlmWalk as Walk>::CAPABILITIES.compute.token,
            crate::TOKEN_DECODE
        );
        let mut ledger = ledger_for::<GlmWalk>();
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
        let positional = layout_of(sparkflash_model::KvPlan::new(1_920, 0));
        assert_eq!(positional.kind(), StateKind::Positional);
        let both = layout_of(sparkflash_model::KvPlan::new(20_160, 4_096));
        assert_eq!(both.kind(), StateKind::Both);
        assert!(both.needs_checkpoint());
        assert_eq!(both.bytes_per_row(), 20_160);
        assert_eq!(both.bytes_at(2), 40_320 + 4_096);
    }

    /// The walk declares its drafter, so the ledger's three round rows are the path's and the
    /// seventh generated check is claimed rather than skipped.
    #[test]
    fn the_walk_declares_the_round_path_and_claims_its_check() {
        const {
            assert!(
                <<GlmWalk as Walk>::Drafter as Drafter<GlmWalk>>::PRESENT,
                "the GLM walk drafts through the runtime's round path"
            );
        }
        let ledger = ledger_for::<GlmWalk>();
        assert!(
            matches!(ledger.mode(Strategy::Drafting), Mode::Native(_)),
            "the runtime ledger states GLM's drafting: {}",
            ledger.line()
        );
        assert!(
            matches!(ledger.mode(Strategy::Runs), Mode::Native(_)),
            "an accepted run reaches the request: {}",
            ledger.line()
        );
        let joint = sparkflash_runtime::joint_rounds().unwrap_or(true);
        assert_eq!(
            matches!(ledger.mode(Strategy::JointRounds), Mode::Native(_)),
            joint,
            "the joint row follows the switch: {}",
            ledger.line()
        );
        assert!(
            !ledger.line().contains("does not state it"),
            "the runtime derives every row GLM states: {}",
            ledger.line()
        );
        let checks = sparkflash_runtime::checks_for::<GlmWalk>();
        assert!(
            checks.contains(&sparkflash_runtime::ConformanceCheck::DraftedEqualsPlain),
            "a drafting arm claims drafted = plain"
        );
        sparkflash_runtime::check_claims::<GlmWalk>(checks)
            .expect("every generated check is supported");
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

    /// A profile's 4096-row chunk survives the runtime's ChunkPolicy quantization.
    #[test]
    fn the_runtime_chunks_a_glm_prompt_4096_rows_at_a_time() {
        let limits = walk_limits(crate::device::DEFAULT_PREFILL_ROWS, 1_048_576);
        assert_eq!(
            sparkflash_runtime::ChunkPolicy::for_limits(limits).rows(),
            crate::device::DEFAULT_PREFILL_ROWS
        );
    }

    /// The budget is one rank step: anchors plus drafts never exceed [MAX_ROWS].
    #[test]
    fn the_budget_caps_anchors_and_drafts_to_one_verify() {
        for seqs in 1..=MAX_ROWS {
            for depth in 0..MAX_ROWS {
                let drafts = DRAFT_BUDGET.rows(seqs, depth);
                assert!(
                    seqs + drafts <= MAX_ROWS,
                    "{seqs} member(s) at depth {depth} budget {drafts} rows"
                );
                assert_eq!(drafts, (seqs * depth).min(MAX_ROWS.saturating_sub(seqs)));
            }
        }
        // Four members and a three-deep chain share four draft rows, so the verify is eight.
        assert_eq!(DRAFT_BUDGET.rows(4, 3), 4);
        assert_eq!(DRAFT_BUDGET.rows(1, 3), 3);
    }

    // The runtime's CPU walk shaped like GLM's: its verify width is [MAX_ROWS], its budget is
    // [DRAFT_BUDGET], its wall is the arm's and its chains are as deep as the arm proposes.
    // The runtime's policies can be checked with no device; the device path is compiled and
    // stays `#[ignore]`d.

    /// A shape walk at `limits` whose drafter proposes `depth` drafts a member, every one the
    /// model's own pick, under the arm's budget and the arm's joint wall.
    fn shape_walk(depth: usize, limits: Limits) -> ShapeWalk {
        ShapeWalk::new(
            StateLayout::new(vec![Region::Positional {
                name: "shape",
                bytes_per_row: 4,
            }]),
            limits,
            depth,
        )
        .with_right_picks()
        .with_budget(DRAFT_BUDGET)
        .with_wall(Box::new(JOINT_WALL))
    }

    fn shape_runtime(depth: usize) -> Runtime<ShapeWalk> {
        Runtime::new(shape_walk(depth, walk_limits(64, 4_096)), 0)
            .expect("the shape runtime starts")
    }

    fn prompted(
        runtime: &Runtime<ShapeWalk>,
        token: u32,
    ) -> sparkflash_runtime::RuntimeSlot<ShapeWalk> {
        let mut slot = runtime.open_slot().expect("a slot");
        runtime
            .prefill_slot(&mut slot, &[token])
            .expect("the prompt");
        slot
    }

    fn verifies(runtime: &Runtime<ShapeWalk>) -> Vec<Vec<usize>> {
        runtime.walk().verifies()
    }

    /// One member keeps the drafter's own depth in one verify pass.
    #[test]
    fn a_lone_member_verifies_its_own_depth_in_one_pass() {
        let runtime = shape_runtime(3);
        let mut members = [(prompted(&runtime, 10), 11)];
        runtime.step_slots(&mut members).expect("the round");
        assert_eq!(verifies(&runtime), vec![vec![4]]);
    }

    /// Four members share one verify pass, and their anchors and drafts fit one rank step.
    #[test]
    fn four_members_share_one_verify_pass() {
        let runtime = shape_runtime(3);
        let mut members: Vec<_> = (0..4)
            .map(|at| (prompted(&runtime, at + 1), at + 11))
            .collect();
        runtime.step_slots(&mut members).expect("the joint round");
        let passes = verifies(&runtime);
        assert_eq!(passes.len(), 1, "one verify pass for four members");
        let rows: usize = passes[0].iter().sum();
        assert_eq!(passes[0].len(), 4);
        assert!(
            passes[0].iter().all(|rows| *rows >= 1),
            "every member steps"
        );
        assert!(
            rows <= MAX_ROWS,
            "the shared verify is one rank step: {rows} rows"
        );
        assert!(
            rows < 4 * 4,
            "the budget shared the drafts: {} rows for four chains of three",
            rows
        );
    }

    /// With joint rounds off, each lane's round walks alone at the drafter's own depth.
    #[test]
    fn with_joint_off_each_lane_walks_alone() {
        let runtime = shape_runtime(3).with_joint_rounds(false);
        let mut members: Vec<_> = (0..2)
            .map(|at| (prompted(&runtime, at + 1), at + 11))
            .collect();
        runtime.step_slots(&mut members).expect("two lone rounds");
        assert_eq!(verifies(&runtime), vec![vec![4], vec![4]]);
    }

    sparkflash_runtime::profile_checks!(
        super::GlmWalk,
        table: "GLM",
        depth: 3,
        chunks: [crate::device::DEFAULT_PREFILL_ROWS],
        shape: |chunk, depth| shape_walk(depth, walk_limits(chunk, 1_048_576))
    );

    sparkflash_runtime::conformance!(super::GlmWalk);
}
